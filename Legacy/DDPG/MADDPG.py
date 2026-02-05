from DDPG.Legacy.DDPG import *
import torch
from torch import nn
from DisturbanceDetectorLLM import (
    TransformerDisturbanceDetector,
    HistoryBuffer,
    AdaptiveControllerBlender,
    DisturbanceDetectorTrainer
)

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device=torch.device("cpu")):
        super().__init__()
        self.q = mlp([obs_dim + 2*act_dim] + list(hidden_sizes) + [1], activation, device=device)

    def forward(self, obs, act, act_d): # act_d is the action of the other agent
        q = self.q(torch.cat([obs, act, act_d], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MAMLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, disturbance_ratio=0.1, hidden_sizes=(256, 256), activation=nn.Tanh, device=torch.device("cpu")):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, device).to(device)
        self.pi_d = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit * disturbance_ratio, device).to(device)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, device).to(device)

    def act(self, obs) -> np.ndarray:
        """Return action for given observation, for exploration."""
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()
    def act_d(self, obs) -> np.ndarray:
        """Return disturbance action for given observation and action, for exploration."""
        with torch.no_grad():
            return self.pi_d(obs).cpu().numpy()

class MAReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device=torch.device("cpu")):
        if device == torch.device("cuda"):
            self.obs_buf = torch.zeros((size, *obs_dim), dtype=torch.float32, device=device)
            self.obs2_buf = torch.zeros((size, *obs_dim), dtype=torch.float32, device=device)
            self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
            self.act_d_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
            self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
            self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        else:
            self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
            self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
            self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
            self.act_d_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
            self.rew_buf = np.zeros(size, dtype=np.float32)
            self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, act_d, rew, next_obs, done):
        if isinstance(self.obs_buf, torch.Tensor):
            self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            self.obs2_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
            self.act_d_buf[self.ptr] = torch.as_tensor(act_d, dtype=torch.float32, device=self.device)
            self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
            self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        else:
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.act_buf[self.ptr] = act
            self.act_d_buf[self.ptr] = act_d
            self.rew_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        idxes = np.random.randint(0, self.size, size=batch_size)
        if isinstance(self.obs_buf, torch.Tensor):
            batch = dict(
                obs=self.obs_buf[idxes],
                obs2=self.obs2_buf[idxes],
                act=self.act_buf[idxes],
                act_d=self.act_d_buf[idxes],
                rew=self.rew_buf[idxes],
                done=self.done_buf[idxes],
            )
            return {k: v.clone().detach() for k, v in batch.items()}
        else:
            batch = dict(
                obs=self.obs_buf[idxes],
                obs2=self.obs2_buf[idxes],
                act=self.act_buf[idxes],
                act_d=self.act_d_buf[idxes],
                rew=self.rew_buf[idxes],
                done=self.done_buf[idxes],
            )
            return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}

class MADDPGAgent:
    ##################################################################################
    # DDPG Agent
    ##################################################################################
    def __init__(
        self,
        env_fn,
        actor_critic,
        ac_kwargs,
        seed=0,
        steps_per_epoch=4000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=1024,
        start_steps=5000,
        update_after=10000,
        update_every=50,
        act_noise=0.1,
        num_test_episodes=10,
        max_ep_len=1000,
        save_freq=10,
        device="auto",
        log_dir="logs",
        plot_freq=1,
        second_agent_probability=0.5,
        disturbance_ratio=0.1,
        use_transformer_blending=True,  # NEW
        transformer_sequence_length=20,  # NEW
        transformer_lr=1e-4  # NEW
    ):
        """DDPG Agent initialization."""
        # Map underscored params to clean internal names (backward compatibility).
        seed, epochs, gamma, device, ac_kwargs = seed, epochs, gamma, device, ac_kwargs
        torch.manual_seed(seed); np.random.seed(seed)

        # Device resolution
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        print(f"[DDPG] Using device: {self.device}")

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]
        self.act_limit = float(self.env.action_space.high[0])

        # Build networks on resolved device
        self.actor_critic = actor_critic(
            self.env.observation_space,
            self.env.action_space,
            **ac_kwargs,
            device=self.device
        ).to(self.device)

        self.target_actor_critic = deepcopy(self.actor_critic).to(self.device)
        for p in self.target_actor_critic.parameters():
            p.requires_grad = False

        self.replay_buffer = MAReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=self.device)
        self.gamma = gamma
        self.polyak = polyak
        self.actor_optimizer = Adam(self.actor_critic.pi.parameters(), lr=pi_lr)
        self.disturbance_optimizer = Adam(self.actor_critic.pi_d.parameters(), lr=pi_lr)
        self.critic_optimizer = Adam(self.actor_critic.q.parameters(), lr=q_lr)

        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        # logging additions
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_returns = []
        self.episode_steps = []
        self._csv_path = os.path.join(self.log_dir, "training_returns.csv")
        self.second_agent_probability = second_agent_probability
        self.disturbance_ratio = disturbance_ratio

        # NEW: Initialize transformer-based controller blending
        self.use_transformer_blending = use_transformer_blending
        if self.use_transformer_blending:
            obs_dim = self.env.observation_space.shape[0]
            self.transformer_detector = TransformerDisturbanceDetector(
                obs_dim=obs_dim,
                sequence_length=transformer_sequence_length,
                d_model=128,
                nhead=4,
                num_layers=3,
                device=self.device
            )
            self.history_buffer = HistoryBuffer(
                obs_dim=obs_dim,
                max_length=transformer_sequence_length,
                device=self.device
            )
            self.controller_blender = AdaptiveControllerBlender(
                detector=self.transformer_detector,
                history_buffer=self.history_buffer,
                smoothing_window=5
            )
            self.transformer_trainer = DisturbanceDetectorTrainer(
                model=self.transformer_detector,
                learning_rate=transformer_lr,
                device=self.device
            )
            
            # Buffer for transformer training data
            self.transformer_replay_buffer = []
            self.alpha_history_log = []
        
        actor_params, critic_params = (count_vars(self.actor_critic.pi), count_vars(self.actor_critic.q))
        print(f"\nNumber of parameters: \t actor: {actor_params}, \t critic: {critic_params}\n")

    def _compute_critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obs, act, act_d, rew, next_obs, done = batch["obs"], batch["act"], batch["act_d"], batch["rew"], batch["obs2"], batch["done"]
        q_val = self.actor_critic.q(obs, act, act_d)
        with torch.no_grad():
            target_q = self.target_actor_critic.q(next_obs, self.target_actor_critic.pi(next_obs),
                                                            self.target_actor_critic.pi_d(next_obs))
            backup = rew + self.gamma * (1 - done) * target_q
        loss_q = ((q_val - backup) ** 2).mean()
        return loss_q, {"q_values": q_val.detach().cpu().numpy()}

    def _compute_actor_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = batch["obs"]
        return -self.actor_critic.q(obs, self.actor_critic.pi(obs), self.actor_critic.pi_d(obs)).mean()


    def _update(self, batch: Dict[str, torch.Tensor]) -> None:
        # Critic update
        self.critic_optimizer.zero_grad()
        critic_loss, _ = self._compute_critic_loss(batch)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic, update actor
        for p in self.actor_critic.q.parameters():
            p.requires_grad = False
        self.actor_optimizer.zero_grad()
        self.disturbance_optimizer.zero_grad()
        actor_loss = self._compute_actor_loss(batch)
        actor_loss.backward()
        for p in self.actor_critic.pi_d.parameters():
            if p.grad is not None:
                p.grad.mul_(-1)
        self.actor_optimizer.step()
        self.disturbance_optimizer.step()
        # Unfreeze critic
        for p in self.actor_critic.q.parameters():
            p.requires_grad = True

        # Polyak averaging
        with torch.no_grad():
            for p, p_t in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
                p_t.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)

    def get_action(self, obs: np.ndarray, noise_scale: float) -> np.ndarray:
        # ensure obs tensor is created on the active device
        action = self.actor_critic.act(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        action += noise_scale * np.random.randn(self.env.action_space.shape[0])
        return np.clip(action, -self.act_limit, self.act_limit)
    ##################################################################################
    def get_disturbance_action(self, obs: np.ndarray, noise_scale: float) -> np.ndarray:
        # ensure obs tensor is created on the active device
        action = self.actor_critic.act_d(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        action += noise_scale * np.random.randn(self.env.action_space.shape[0])
        return np.clip(action, -self.act_limit * self.disturbance_ratio, self.act_limit * self.disturbance_ratio)
    ##################################################################################

    def get_blended_action(self, obs: np.ndarray, noise_scale: float) -> Tuple[np.ndarray, float, float]:
        """
        Get action using transformer-based blending between optimal and robust controllers.
        
        Returns:
            action: Blended action
            alpha: Blending weight (0=robust, 1=optimal)
            disturbance_prob: Predicted disturbance probability
        """
        # Get optimal controller action (without disturbance)
        action_optimal = self.get_action(obs, noise_scale)
        
        # Get robust controller action (with disturbance consideration)
        action_robust = self.get_action(obs, noise_scale)
        # Note: In practice, the robust action comes from training with disturbances
        
        # Blend using transformer
        blended_action, alpha, disturbance_prob = self.controller_blender.get_blended_action(
            obs, action_optimal, action_robust
        )
        
        return blended_action, alpha, disturbance_prob

    def _collect_transformer_training_data(
        self,
        obs_sequence: np.ndarray,
        has_disturbance: bool,
        episode_performance: float
    ):
        """
        Collect data for training the transformer.
        
        Args:
            obs_sequence: Recent observation history
            has_disturbance: Whether disturbance was present
            episode_performance: Metric to compute optimal blending weight
        """
        # Compute optimal blending weight based on performance
        # Lower performance with disturbance → lower α (more robust)
        # Higher performance without disturbance → higher α (more optimal)
        if has_disturbance:
            # Map performance to α: worse performance → lower α
            optimal_alpha = np.clip(episode_performance / 100.0, 0.0, 0.5)
        else:
            # No disturbance: prefer optimal controller
            optimal_alpha = np.clip(0.5 + episode_performance / 200.0, 0.5, 1.0)
        
        self.transformer_replay_buffer.append({
            'obs_sequence': obs_sequence,
            'has_disturbance': float(has_disturbance),
            'optimal_alpha': optimal_alpha
        })
        
        # Limit buffer size
        if len(self.transformer_replay_buffer) > 10000:
            self.transformer_replay_buffer.pop(0)

    def _train_transformer(self, batch_size: int = 64):
        """Train the transformer disturbance detector."""
        if len(self.transformer_replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        indices = np.random.choice(len(self.transformer_replay_buffer), batch_size, replace=False)
        batch = [self.transformer_replay_buffer[i] for i in indices]
        
        # Prepare tensors
        obs_sequences = torch.stack([
            torch.as_tensor(item['obs_sequence'], dtype=torch.float32, device=self.device)
            for item in batch
        ])
        disturbance_labels = torch.tensor(
            [item['has_disturbance'] for item in batch],
            dtype=torch.float32,
            device=self.device
        )
        optimal_alphas = torch.tensor(
            [item['optimal_alpha'] for item in batch],
            dtype=torch.float32,
            device=self.device
        )
        
        # Train
        info = self.transformer_trainer.train_step(obs_sequences, disturbance_labels, optimal_alphas)
        return info

    def _record_episode(self, global_step: int, ep_return: float):
        self.episode_steps.append(global_step)
        self.episode_returns.append(ep_return)
        # append to csv
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([global_step, ep_return])

    def _plot_returns(self, epoch: int):
        if not self.episode_returns:
            return
        plt.figure(figsize=(6,4))
        plt.plot(self.episode_steps, self.episode_returns, label="Episode Return")
        plt.xlabel("Env Steps")
        plt.ylabel("Return")
        plt.title("Return vs Steps")
        plt.grid(True, alpha=0.3)
        plt.legend()
        latest_path = os.path.join(self.log_dir, "return_vs_step_latest.png")
        plt.savefig(latest_path, dpi=150, bbox_inches="tight")
        per_epoch_path = os.path.join(self.log_dir, f"return_vs_step_epoch_{epoch}.png")
        plt.savefig(per_epoch_path, dpi=150, bbox_inches="tight")
        plt.close()

    def train(self, epochs: int = None):
        if epochs is None:
            epochs = self.epochs
        
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["global_step", "episode_return"])
        
        # NEW: CSV for alpha tracking
        alpha_csv_path = os.path.join(self.log_dir, "blending_alpha.csv")
        if self.use_transformer_blending and not os.path.isfile(alpha_csv_path):
            with open(alpha_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["global_step", "alpha", "disturbance_prob", "has_disturbance"])
        
        total_steps = self.steps_per_epoch * epochs
        start_time = time.time()
        obs, _ = self.env.reset()
        episode_return, episode_length = 0.0, 0
        disturbance = False
        
        # NEW: Episode observation history for transformer
        episode_obs_history = []

        for t in range(total_steps):
            episode_obs_history.append(obs.copy())
            
            if t > self.start_steps:
                if self.use_transformer_blending:
                    # Use transformer-based blending
                    act, alpha, disturbance_prob = self.get_blended_action(obs, self.act_noise)
                    
                    # Log alpha
                    self.alpha_history_log.append((t, alpha, disturbance_prob, float(disturbance)))
                    if len(self.alpha_history_log) % 100 == 0:
                        with open(alpha_csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            for entry in self.alpha_history_log[-100:]:
                                writer.writerow(entry)
                else:
                    # Original behavior
                    act = self.get_action(obs, self.act_noise)
                
                if disturbance:
                    act_d = self.get_disturbance_action(obs, self.act_noise)
                else:
                    act_d = np.zeros_like(act)
            else:
                act = self.env.action_space.sample()
                if disturbance:
                    act_d = self.env.action_space.sample() * self.disturbance_ratio
                else:
                    act_d = np.zeros_like(act)

            next_obs, reward, done, _, _ = self.env.step(act+act_d)
            self.replay_buffer.store(obs, act, act_d, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            episode_length += 1

            if done or (episode_length == self.max_ep_len):
                # NEW: Collect transformer training data
                if self.use_transformer_blending and len(episode_obs_history) >= 20:
                    # Take last sequence_length observations
                    seq_len = self.transformer_detector.sequence_length
                    obs_seq = np.array(episode_obs_history[-seq_len:])
                    self._collect_transformer_training_data(obs_seq, disturbance, episode_return)
                
                self._record_episode(t, episode_return)
                obs, _ = self.env.reset()
                episode_return, episode_length = 0.0, 0
                episode_obs_history = []
                
                disturbance = np.random.rand() < self.second_agent_probability
                if disturbance:
                    print("Disturbance agent activated for next episode.")
                else:
                    print("No disturbance agent for next episode.")
                
                # Reset blender history
                if self.use_transformer_blending:
                    self.controller_blender.reset()

            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(500):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self._update(batch)
                
                # NEW: Train transformer
                if self.use_transformer_blending and t % (self.update_every * 10) == 0:
                    for _ in range(50):
                        transformer_info = self._train_transformer(batch_size=64)
                        if transformer_info:
                            print(f"Transformer training - Loss: {transformer_info.get('total_loss', 0):.4f}, "
                                  f"Alpha MSE: {transformer_info.get('loss_blending', 0):.4f}")

            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch
                print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")

            # save model
            if (t + 1) % (self.steps_per_epoch * self.save_freq) == 0:
                self.save()

    def save(self, filepath: str = "model/"):
        os.makedirs(filepath, exist_ok=True)
        device_suffix = "cuda" if self.device == torch.device("cuda") else "cpu"
        
        torch.save(self.actor_critic.pi.state_dict(), os.path.join(filepath, f"actor_{device_suffix}.pth"))
        torch.save(self.actor_critic.pi_d.state_dict(), os.path.join(filepath, f"disturbance_{device_suffix}.pth"))
        torch.save(self.actor_critic.q.state_dict(), os.path.join(filepath, f"q_{device_suffix}.pth"))
        
        # NEW: Save transformer
        if self.use_transformer_blending:
            torch.save(
                self.transformer_detector.state_dict(),
                os.path.join(filepath, f"transformer_{device_suffix}.pth")
            )
        
        print(colorize("Model saved.", "blue", bold=True))

    def load(self, filepath: str = "model/", load_device: torch.device = torch.device("cpu"), from_device_to_load: str = "cpu"):
        actor_file = f"actor_{from_device_to_load}.pth"
        disturbance_file = f"disturbance_{from_device_to_load}.pth"
        critic_file = f"q_{from_device_to_load}.pth"
        actor_path = os.path.join(filepath, actor_file)
        disturbance_path = os.path.join(filepath, disturbance_file)
        critic_path = os.path.join(filepath, critic_file)

        if not (os.path.isfile(actor_path) and os.path.isfile(critic_path)):
            print(colorize("Model not found.", "red", bold=True))
            return

        map_loc = load_device
        self.actor_critic.pi.load_state_dict(torch.load(actor_path, map_location=map_loc))
        self.actor_critic.pi_d.load_state_dict(torch.load(disturbance_path, map_location=map_loc))
        self.actor_critic.q.load_state_dict(torch.load(critic_path, map_location=map_loc))
        print(colorize(f"Model loaded on {load_device}.", "blue", bold=True))
        
        # NEW: Load transformer
        if self.use_transformer_blending:
            transformer_file = f"transformer_{from_device_to_load}.pth"
            transformer_path = os.path.join(filepath, transformer_file)
            if os.path.isfile(transformer_path):
                self.transformer_detector.load_state_dict(
                    torch.load(transformer_path, map_location=load_device)
                )
                print(colorize("Transformer loaded.", "blue", bold=True))
