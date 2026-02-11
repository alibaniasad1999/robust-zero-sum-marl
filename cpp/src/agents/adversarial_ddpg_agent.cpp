#include <rzsm/agents/adversarial_ddpg_agent.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace fs = std::filesystem;

namespace rzsm {

AdversarialDDPGAgent::AdversarialDDPGAgent(
    std::function<MuJoCoEnv()> env_fn,
    const AdversarialDDPGConfig& cfg)
    : env_(env_fn()), device_(select_device(cfg.device)),
      buffer_(cfg.replay_size, 0, 0, device_),
      dataset_logger_(cfg.log_dir + "/dataset"),
      use_transformer_(cfg.use_transformer), cfg_(cfg)
{
    set_seed(cfg.seed);

    auto obs_info = env_.observation_space();
    auto act_info = env_.action_space();
    obs_dim_ = obs_info.dim;
    act_dim_ = act_info.dim;
    act_limit_ = act_info.high;
    dist_limit_ = act_limit_ * cfg.disturbance_ratio;

    // Reinit buffer with act_dim * 2
    buffer_ = ReplayBuffer(cfg.replay_size, obs_dim_, act_dim_ * 2, device_);

    // Networks
    pi_rob_ = MLPActor(obs_dim_, act_dim_, cfg.hidden_sizes,
                        cfg.activation, act_limit_);
    pi_adv_ = MLPActor(obs_dim_, act_dim_, cfg.hidden_sizes,
                        cfg.activation, dist_limit_);
    q_ = AdversarialMLPQFunction(obs_dim_, act_dim_, act_dim_,
                                  cfg.hidden_sizes, cfg.activation);
    pi_rob_->to(device_);
    pi_adv_->to(device_);
    q_->to(device_);

    // Targets
    pi_rob_targ_ = MLPActor(obs_dim_, act_dim_, cfg.hidden_sizes,
                             cfg.activation, act_limit_);
    pi_adv_targ_ = MLPActor(obs_dim_, act_dim_, cfg.hidden_sizes,
                             cfg.activation, dist_limit_);
    q_targ_ = AdversarialMLPQFunction(obs_dim_, act_dim_, act_dim_,
                                       cfg.hidden_sizes, cfg.activation);
    copy_params(*pi_rob_, *pi_rob_targ_);
    copy_params(*pi_adv_, *pi_adv_targ_);
    copy_params(*q_, *q_targ_);
    pi_rob_targ_->to(device_); pi_adv_targ_->to(device_); q_targ_->to(device_);
    freeze(*pi_rob_targ_); freeze(*pi_adv_targ_); freeze(*q_targ_);

    // Optional pi_opt
    if (!cfg.pi_opt_path.empty()) {
        pi_opt_ = MLPActor(obs_dim_, act_dim_, cfg.hidden_sizes,
                           cfg.activation, act_limit_);
        torch::load(*pi_opt_, cfg.pi_opt_path + "/pi.pt");
        (*pi_opt_)->to(device_);
        (*pi_opt_)->eval();
        freeze(**pi_opt_);
        std::cout << "[AdversarialDDPG] Loaded pi_opt from "
                  << cfg.pi_opt_path << std::endl;
    }

    // Optimizers
    pi_rob_optim_ = std::make_unique<torch::optim::Adam>(
        pi_rob_->parameters(), torch::optim::AdamOptions(cfg.pi_lr));
    pi_adv_optim_ = std::make_unique<torch::optim::Adam>(
        pi_adv_->parameters(), torch::optim::AdamOptions(cfg.pi_lr));
    q_optim_ = std::make_unique<torch::optim::Adam>(
        q_->parameters(), torch::optim::AdamOptions(cfg.q_lr));

    // Transformer detector
    if (use_transformer_) {
        detector_ = std::make_unique<TransformerDisturbanceDetector>(
            obs_dim_, cfg.transformer_seq_len, cfg.transformer_d_model,
            cfg.transformer_nhead, cfg.transformer_layers);
        (*detector_)->to(device_);
        history_buf_ = std::make_unique<HistoryBuffer>(
            obs_dim_, cfg.transformer_seq_len, device_);
        blender_ = std::make_unique<AdaptiveControllerBlender>(
            *detector_, *history_buf_, 5);
        det_trainer_ = std::make_unique<DisturbanceDetectorTrainer>(
            *detector_, cfg.transformer_lr);
    }

    // Logging
    fs::create_directories(cfg.log_dir);
    csv_path_ = cfg.log_dir + "/training_returns.csv";

    std::cout << "[AdversarialDDPG] device=" << device_
              << "  pi_rob=" << count_vars(*pi_rob_)
              << "  pi_adv=" << count_vars(*pi_adv_)
              << "  q=" << count_vars(*q_) << std::endl;
}

std::vector<float> AdversarialDDPGAgent::get_ctrl_action(
    const std::vector<float>& obs, float noise)
{
    torch::NoGradGuard no_grad;
    auto obs_t = torch::from_blob(const_cast<float*>(obs.data()),
                                   {obs_dim_}, torch::kFloat32).to(device_);
    auto a = pi_rob_->forward(obs_t).to(torch::kCPU);
    auto ptr = a.data_ptr<float>();
    std::normal_distribution<float> dist(0.0f, noise);
    std::vector<float> action(act_dim_);
    for (int64_t i = 0; i < act_dim_; ++i)
        action[i] = std::clamp(ptr[i] + dist(global_rng()),
                               -act_limit_, act_limit_);
    return action;
}

std::vector<float> AdversarialDDPGAgent::get_adv_action(
    const std::vector<float>& obs, float noise)
{
    torch::NoGradGuard no_grad;
    auto obs_t = torch::from_blob(const_cast<float*>(obs.data()),
                                   {obs_dim_}, torch::kFloat32).to(device_);
    auto a = pi_adv_->forward(obs_t).to(torch::kCPU);
    auto ptr = a.data_ptr<float>();
    std::normal_distribution<float> dist(0.0f, noise);
    std::vector<float> action(act_dim_);
    for (int64_t i = 0; i < act_dim_; ++i)
        action[i] = std::clamp(ptr[i] + dist(global_rng()),
                               -dist_limit_, dist_limit_);
    return action;
}

std::vector<float> AdversarialDDPGAgent::get_opt_action(
    const std::vector<float>& obs, float noise)
{
    if (!pi_opt_) return get_ctrl_action(obs, noise);
    torch::NoGradGuard no_grad;
    auto obs_t = torch::from_blob(const_cast<float*>(obs.data()),
                                   {obs_dim_}, torch::kFloat32).to(device_);
    auto a = (*pi_opt_)->forward(obs_t).to(torch::kCPU);
    auto ptr = a.data_ptr<float>();
    std::normal_distribution<float> dist(0.0f, noise);
    std::vector<float> action(act_dim_);
    for (int64_t i = 0; i < act_dim_; ++i)
        action[i] = std::clamp(ptr[i] + dist(global_rng()),
                               -act_limit_, act_limit_);
    return action;
}

void AdversarialDDPGAgent::update(const ReplayBatch& batch) {
    auto o = batch.obs, o2 = batch.next_obs;
    auto r = batch.rew.squeeze(-1);
    auto d = batch.done().to(torch::kFloat32).squeeze(-1);
    auto a_ctrl = batch.act.slice(/*dim=*/1, 0, act_dim_);
    auto a_dist = batch.act.slice(/*dim=*/1, act_dim_);

    // Critic
    torch::Tensor backup;
    {
        torch::NoGradGuard no_grad;
        auto a2_ctrl = pi_rob_targ_->forward(o2);
        auto a2_dist = pi_adv_targ_->forward(o2);
        auto q_next = q_targ_->forward(o2, a2_ctrl, a2_dist);
        backup = r + cfg_.gamma * (1.0f - d) * q_next;
    }
    auto q_val = q_->forward(o, a_ctrl, a_dist);
    auto loss_q = ((q_val - backup).pow(2)).mean();
    q_optim_->zero_grad();
    loss_q.backward();
    q_optim_->step();

    // Actor + adversary (gradient flip)
    for (auto& p : q_->parameters()) p.set_requires_grad(false);

    auto a_rob = pi_rob_->forward(o);
    auto a_adv = pi_adv_->forward(o);
    auto q_for_actors = q_->forward(o, a_rob, a_adv);
    auto loss_pi = -q_for_actors.mean();

    pi_rob_optim_->zero_grad();
    pi_adv_optim_->zero_grad();
    loss_pi.backward();

    // Flip adversary gradients
    for (auto& p : pi_adv_->parameters()) {
        if (p.grad().defined()) p.grad().mul_(-1.0);
    }

    pi_rob_optim_->step();
    pi_adv_optim_->step();

    for (auto& p : q_->parameters()) p.set_requires_grad(true);
}

void AdversarialDDPGAgent::polyak_update() {
    torch::NoGradGuard no_grad;
    float tau = cfg_.polyak;

    auto update = [tau](const std::vector<torch::Tensor>& src,
                        std::vector<torch::Tensor>& tgt) {
        for (size_t i = 0; i < src.size(); ++i)
            tgt[i].data().mul_(tau).add_(src[i].data(), 1.0 - tau);
    };

    auto pr = pi_rob_->parameters(), prt = pi_rob_targ_->parameters();
    auto pa = pi_adv_->parameters(), pat = pi_adv_targ_->parameters();
    auto qp = q_->parameters(), qt = q_targ_->parameters();
    update(pr, prt);
    update(pa, pat);
    update(qp, qt);
}

void AdversarialDDPGAgent::collect_detector_data(
    const std::vector<std::vector<float>>& obs_seq,
    bool has_dist, float ep_return)
{
    float alpha_target;
    if (has_dist) {
        alpha_target = std::clamp(ep_return / 100.0f, 0.0f, 0.5f);
    } else {
        alpha_target = std::clamp(0.5f + ep_return / 200.0f, 0.5f, 1.0f);
    }
    transformer_data_.push_back({obs_seq, has_dist ? 1.0f : 0.0f, alpha_target});
    if (transformer_data_.size() > 10000) {
        transformer_data_.erase(transformer_data_.begin());
    }
}

void AdversarialDDPGAgent::train_detector(int batch_size) {
    if (static_cast<int>(transformer_data_.size()) < batch_size) return;

    std::uniform_int_distribution<int> dist(0, transformer_data_.size() - 1);
    int seq_len = cfg_.transformer_seq_len;

    auto obs_t = torch::zeros({batch_size, seq_len, obs_dim_}, torch::kFloat32);
    auto dist_t = torch::zeros({batch_size}, torch::kFloat32);
    auto alpha_t = torch::zeros({batch_size}, torch::kFloat32);

    for (int i = 0; i < batch_size; ++i) {
        int idx = dist(global_rng());
        auto& sample = transformer_data_[idx];
        dist_t[i] = sample.dist;
        alpha_t[i] = sample.alpha;
        for (int64_t t = 0; t < seq_len && t < static_cast<int64_t>(sample.obs_seq.size()); ++t) {
            for (int64_t j = 0; j < obs_dim_; ++j) {
                obs_t[i][t][j] = sample.obs_seq[t][j];
            }
        }
    }

    det_trainer_->train_step(obs_t.to(device_), dist_t.to(device_),
                              alpha_t.to(device_));
}

void AdversarialDDPGAgent::train(std::optional<int> epochs) {
    int total_epochs = epochs.value_or(cfg_.epochs);
    int total_steps = cfg_.steps_per_epoch * total_epochs;

    if (!fs::exists(csv_path_)) {
        std::ofstream csv(csv_path_);
        csv << "step,episode_return,has_disturbance\n";
    }

    auto obs = env_.reset(cfg_.seed);
    float ep_ret = 0.0f;
    int ep_len = 0;
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    bool has_dist = prob_dist(global_rng()) < cfg_.disturbance_probability;
    std::vector<std::vector<float>> ep_obs_history;
    int episode_id = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < total_steps; ++t) {
        ep_obs_history.push_back(obs);

        std::vector<float> act_ctrl, act_dist;
        if (t < cfg_.start_steps) {
            act_ctrl = env_.sample_action();
            if (has_dist) {
                act_dist = env_.sample_action();
                for (auto& v : act_dist) v *= cfg_.disturbance_ratio;
            } else {
                act_dist.assign(act_dim_, 0.0f);
            }
        } else {
            act_ctrl = get_ctrl_action(obs, cfg_.act_noise);
            if (has_dist) {
                act_dist = get_adv_action(obs, cfg_.act_noise);
            } else {
                act_dist.assign(act_dim_, 0.0f);
            }
        }

        // Total action
        std::vector<float> act_total(act_dim_);
        for (int64_t i = 0; i < act_dim_; ++i)
            act_total[i] = act_ctrl[i] + act_dist[i];

        auto result = env_.step(act_total.data());

        // Combined action for buffer
        std::vector<float> combined(act_dim_ * 2);
        std::copy(act_ctrl.begin(), act_ctrl.end(), combined.begin());
        std::copy(act_dist.begin(), act_dist.end(), combined.begin() + act_dim_);
        buffer_.add(obs.data(), result.obs.data(), combined.data(),
                     result.reward, result.terminated, result.truncated);

        // Dataset logger
        TransitionRecord rec;
        rec.t = ep_len;
        rec.episode_id = episode_id;
        rec.obs = obs;
        rec.action_ctrl = act_ctrl;
        rec.action_dist = act_dist;
        rec.action_total = act_total;
        rec.reward = result.reward;
        rec.terminated = result.terminated;
        rec.truncated = result.truncated;
        rec.disturbance_params = {has_dist ? 1.0f : 0.0f, cfg_.disturbance_ratio};
        rec.disturbance_tag = has_dist ? "adversary" : "none";
        dataset_logger_.log(rec);

        obs = result.obs;
        ep_ret += result.reward;
        ep_len++;

        // Episode boundary
        if (result.terminated || result.truncated || ep_len >= cfg_.max_ep_len) {
            std::ofstream csv(csv_path_, std::ios::app);
            csv << t << "," << ep_ret << "," << (has_dist ? 1 : 0) << "\n";
            std::cout << "  step=" << t + 1 << "  ret=" << ep_ret
                      << "  len=" << ep_len << "  dist=" << has_dist << std::endl;
            dataset_logger_.end_episode();

            if (use_transformer_) {
                int seq_len = cfg_.transformer_seq_len;
                if (static_cast<int>(ep_obs_history.size()) >= seq_len) {
                    std::vector<std::vector<float>> obs_seq(
                        ep_obs_history.end() - seq_len, ep_obs_history.end());
                    collect_detector_data(obs_seq, has_dist, ep_ret);
                }
                blender_->reset();
            }

            obs = env_.reset();
            ep_ret = 0.0f;
            ep_len = 0;
            ep_obs_history.clear();
            has_dist = prob_dist(global_rng()) < cfg_.disturbance_probability;
            episode_id++;
        }

        // Network updates
        if (t >= cfg_.update_after && t % cfg_.update_every == 0) {
            for (int u = 0; u < cfg_.n_updates; ++u) {
                auto batch = buffer_.sample(cfg_.batch_size);
                update(batch);
                polyak_update();
            }
            if (use_transformer_ && t % cfg_.transformer_train_interval == 0) {
                for (int u = 0; u < 50; ++u) train_detector(64);
            }
        }

        // Epoch boundary
        if ((t + 1) % cfg_.steps_per_epoch == 0) {
            int epoch = (t + 1) / cfg_.steps_per_epoch;
            auto elapsed = std::chrono::steady_clock::now() - t0;
            auto sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            std::cout << "[AdversarialDDPG] Epoch " << epoch << "/" << total_epochs
                      << "  elapsed=" << sec << "s" << std::endl;
            if (epoch % cfg_.save_freq == 0) save();
        }
    }
}

void AdversarialDDPGAgent::save(const std::string& path) {
    std::string dir = path.empty() ? cfg_.log_dir + "/checkpoints" : path;
    fs::create_directories(dir);
    torch::save(pi_rob_, dir + "/pi_rob.pt");
    torch::save(pi_adv_, dir + "/pi_adv.pt");
    torch::save(q_, dir + "/q.pt");
    if (use_transformer_) {
        torch::save(*detector_, dir + "/detector.pt");
    }
    std::cout << "  [saved] " << dir << std::endl;
}

void AdversarialDDPGAgent::load(const std::string& path) {
    torch::load(pi_rob_, path + "/pi_rob.pt");
    torch::load(pi_adv_, path + "/pi_adv.pt");
    torch::load(q_, path + "/q.pt");
    pi_rob_->to(device_); pi_adv_->to(device_); q_->to(device_);
    copy_params(*pi_rob_, *pi_rob_targ_);
    copy_params(*pi_adv_, *pi_adv_targ_);
    copy_params(*q_, *q_targ_);
    pi_rob_targ_->to(device_); pi_adv_targ_->to(device_); q_targ_->to(device_);
    freeze(*pi_rob_targ_); freeze(*pi_adv_targ_); freeze(*q_targ_);
    if (use_transformer_) {
        std::string det_path = path + "/detector.pt";
        if (fs::exists(det_path)) {
            torch::load(*detector_, det_path);
            (*detector_)->to(device_);
        }
    }
    std::cout << "  [loaded] " << path << std::endl;
}

} // namespace rzsm
