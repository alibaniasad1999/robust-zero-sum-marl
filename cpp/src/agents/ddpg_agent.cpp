#include <rzsm/agents/ddpg_agent.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

namespace rzsm {

DDPGAgent::DDPGAgent(std::function<MuJoCoEnv()> env_fn, const DDPGConfig& cfg)
    : env_(env_fn()), device_(select_device(cfg.device)),
      buffer_(cfg.replay_size, 0, 0, device_), cfg_(cfg)
{
    set_seed(cfg.seed);

    auto obs_info = env_.observation_space();
    auto act_info = env_.action_space();
    obs_dim_ = obs_info.dim;
    act_dim_ = act_info.dim;
    act_limit_ = act_info.high;

    // Reinitialize buffer with correct dims
    buffer_ = ReplayBuffer(cfg.replay_size, obs_dim_, act_dim_, device_);

    // Networks
    pi_ = MLPActor(obs_dim_, act_dim_, cfg.hidden_sizes,
                    cfg.activation, act_limit_);
    q_ = MLPQFunction(obs_dim_, act_dim_, cfg.hidden_sizes, cfg.activation);
    pi_->to(device_);
    q_->to(device_);

    // Target copies
    pi_targ_ = MLPActor(obs_dim_, act_dim_, cfg.hidden_sizes,
                         cfg.activation, act_limit_);
    q_targ_ = MLPQFunction(obs_dim_, act_dim_, cfg.hidden_sizes, cfg.activation);
    copy_params(*pi_, *pi_targ_);
    copy_params(*q_, *q_targ_);
    pi_targ_->to(device_);
    q_targ_->to(device_);
    freeze(*pi_targ_);
    freeze(*q_targ_);

    // Optimizers
    pi_optim_ = std::make_unique<torch::optim::Adam>(
        pi_->parameters(), torch::optim::AdamOptions(cfg.pi_lr));
    q_optim_ = std::make_unique<torch::optim::Adam>(
        q_->parameters(), torch::optim::AdamOptions(cfg.q_lr));

    // Logging
    fs::create_directories(cfg.log_dir);
    csv_path_ = cfg.log_dir + "/training_returns.csv";

    std::cout << "[DDPGAgent] device=" << device_
              << "  pi_params=" << count_vars(*pi_)
              << "  q_params=" << count_vars(*q_) << std::endl;
}

std::vector<float> DDPGAgent::get_action(const std::vector<float>& obs,
                                          float noise) {
    torch::NoGradGuard no_grad;
    auto obs_t = torch::from_blob(const_cast<float*>(obs.data()),
                                   {obs_dim_}, torch::kFloat32)
                     .to(device_);
    auto a_t = pi_->forward(obs_t).to(torch::kCPU);
    auto a_ptr = a_t.data_ptr<float>();

    std::normal_distribution<float> dist(0.0f, noise);
    std::vector<float> action(act_dim_);
    for (int64_t i = 0; i < act_dim_; ++i) {
        action[i] = std::clamp(a_ptr[i] + dist(global_rng()),
                               -act_limit_, act_limit_);
    }
    return action;
}

void DDPGAgent::update(const ReplayBatch& batch) {
    auto o = batch.obs, a = batch.act, o2 = batch.next_obs;
    auto r = batch.rew.squeeze(-1);
    auto d = batch.done().to(torch::kFloat32).squeeze(-1);

    // Compute backup (no grad for targets)
    torch::Tensor backup;
    {
        torch::NoGradGuard no_grad;
        auto a2 = pi_targ_->forward(o2);
        auto q_next = q_targ_->forward(o2, a2);
        backup = r + cfg_.gamma * (1.0f - d) * q_next;
    }

    // Critic update
    auto q_val = q_->forward(o, a);
    auto loss_q = ((q_val - backup).pow(2)).mean();
    q_optim_->zero_grad();
    loss_q.backward();
    q_optim_->step();

    // Actor update — freeze Q grads
    for (auto& p : q_->parameters()) p.set_requires_grad(false);
    auto loss_pi = -q_->forward(o, pi_->forward(o)).mean();
    pi_optim_->zero_grad();
    loss_pi.backward();
    pi_optim_->step();
    for (auto& p : q_->parameters()) p.set_requires_grad(true);
}

void DDPGAgent::polyak_update() {
    torch::NoGradGuard no_grad;
    auto tau = cfg_.polyak;
    auto pi_p = pi_->parameters();
    auto pi_t = pi_targ_->parameters();
    for (size_t i = 0; i < pi_p.size(); ++i)
        pi_t[i].data().mul_(tau).add_(pi_p[i].data(), 1.0 - tau);

    auto q_p = q_->parameters();
    auto q_t = q_targ_->parameters();
    for (size_t i = 0; i < q_p.size(); ++i)
        q_t[i].data().mul_(tau).add_(q_p[i].data(), 1.0 - tau);
}

void DDPGAgent::train(std::optional<int> epochs) {
    int total_epochs = epochs.value_or(cfg_.epochs);
    int total_steps = cfg_.steps_per_epoch * total_epochs;

    // CSV header
    if (!fs::exists(csv_path_)) {
        std::ofstream csv(csv_path_);
        csv << "step,episode_return\n";
    }

    auto obs = env_.reset(cfg_.seed);
    float ep_ret = 0.0f;
    int ep_len = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < total_steps; ++t) {
        std::vector<float> act;
        if (t < cfg_.start_steps) {
            act = env_.sample_action();
        } else {
            act = get_action(obs, cfg_.act_noise);
        }

        auto result = env_.step(act.data());
        buffer_.add(obs.data(), result.obs.data(), act.data(),
                     result.reward, result.terminated, result.truncated);
        obs = result.obs;
        ep_ret += result.reward;
        ep_len++;

        if (result.terminated || result.truncated || ep_len >= cfg_.max_ep_len) {
            std::ofstream csv(csv_path_, std::ios::app);
            csv << t << "," << ep_ret << "\n";
            std::cout << "  step=" << t + 1 << "  ret=" << ep_ret
                      << "  len=" << ep_len << std::endl;
            obs = env_.reset();
            ep_ret = 0.0f;
            ep_len = 0;
        }

        if (t >= cfg_.update_after && t % cfg_.update_every == 0) {
            for (int u = 0; u < cfg_.n_updates; ++u) {
                auto batch = buffer_.sample(cfg_.batch_size);
                update(batch);
                polyak_update();
            }
        }

        if ((t + 1) % cfg_.steps_per_epoch == 0) {
            int epoch = (t + 1) / cfg_.steps_per_epoch;
            auto elapsed = std::chrono::steady_clock::now() - t0;
            auto sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            std::cout << "[DDPGAgent] Epoch " << epoch << "/" << total_epochs
                      << "  elapsed=" << sec << "s" << std::endl;
            if (epoch % cfg_.save_freq == 0) save();
        }
    }
}

void DDPGAgent::save(const std::string& path) {
    std::string dir = path.empty() ? cfg_.log_dir + "/checkpoints" : path;
    fs::create_directories(dir);
    torch::save(pi_, dir + "/pi.pt");
    torch::save(q_, dir + "/q.pt");
    std::cout << "  [saved] " << dir << std::endl;
}

void DDPGAgent::load(const std::string& path) {
    torch::load(pi_, path + "/pi.pt");
    torch::load(q_, path + "/q.pt");
    pi_->to(device_);
    q_->to(device_);
    copy_params(*pi_, *pi_targ_);
    copy_params(*q_, *q_targ_);
    pi_targ_->to(device_);
    q_targ_->to(device_);
    freeze(*pi_targ_);
    freeze(*q_targ_);
    std::cout << "  [loaded] " << path << std::endl;
}

} // namespace rzsm
