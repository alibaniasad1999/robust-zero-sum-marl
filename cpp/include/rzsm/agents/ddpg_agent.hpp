#pragma once
#include <torch/torch.h>
#include <rzsm/common.hpp>
#include <rzsm/networks/mlp.hpp>
#include <rzsm/utils/replay_buffer.hpp>
#include <rzsm/envs/mujoco_env.hpp>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace rzsm {

struct DDPGConfig {
    std::vector<int64_t> hidden_sizes = {256, 256};
    ActivationType activation = ActivationType::Tanh;
    int seed = 0;
    int steps_per_epoch = 4000;
    int epochs = 100;
    int64_t replay_size = 1000000;
    float gamma = 0.99f;
    float polyak = 0.995f;
    float pi_lr = 1e-3f;
    float q_lr = 1e-3f;
    int batch_size = 1024;
    int start_steps = 5000;
    int update_after = 10000;
    int update_every = 50;
    int n_updates = 500;
    float act_noise = 0.1f;
    int max_ep_len = 1000;
    int save_freq = 10;
    std::string device = "auto";
    std::string log_dir = "logs/nominal";
};

class DDPGAgent {
public:
    DDPGAgent(std::function<MuJoCoEnv()> env_fn, const DDPGConfig& cfg = {});

    void train(std::optional<int> epochs = std::nullopt);
    void save(const std::string& path = "");
    void load(const std::string& path);

private:
    std::vector<float> get_action(const std::vector<float>& obs, float noise);
    void update(const ReplayBatch& batch);
    void polyak_update();

    MuJoCoEnv env_;
    torch::Device device_;
    float act_limit_;
    int64_t obs_dim_, act_dim_;

    MLPActor pi_{nullptr}, pi_targ_{nullptr};
    MLPQFunction q_{nullptr}, q_targ_{nullptr};
    std::unique_ptr<torch::optim::Adam> pi_optim_, q_optim_;

    ReplayBuffer buffer_;
    DDPGConfig cfg_;
    std::string csv_path_;
};

} // namespace rzsm
