#pragma once
#include <torch/torch.h>
#include <rzsm/common.hpp>
#include <rzsm/networks/mlp.hpp>
#include <rzsm/utils/replay_buffer.hpp>
#include <rzsm/utils/logger.hpp>
#include <rzsm/detector/transformer.hpp>
#include <rzsm/detector/history_buffer.hpp>
#include <rzsm/detector/trainer.hpp>
#include <rzsm/detector/blender.hpp>
#include <rzsm/envs/mujoco_env.hpp>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rzsm {

struct AdversarialDDPGConfig {
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
    std::string log_dir = "logs/adversarial";
    // Adversary
    float disturbance_ratio = 0.1f;
    float disturbance_probability = 0.5f;
    // Transformer
    bool use_transformer = true;
    int transformer_seq_len = 20;
    int transformer_d_model = 128;
    int transformer_nhead = 4;
    int transformer_layers = 3;
    float transformer_lr = 1e-4f;
    int transformer_train_interval = 500;
    // Pre-trained optimal policy
    std::string pi_opt_path = "";
};

class AdversarialDDPGAgent {
public:
    AdversarialDDPGAgent(std::function<MuJoCoEnv()> env_fn,
                         const AdversarialDDPGConfig& cfg = {});

    void train(std::optional<int> epochs = std::nullopt);
    void save(const std::string& path = "");
    void load(const std::string& path);

private:
    std::vector<float> get_ctrl_action(const std::vector<float>& obs, float noise);
    std::vector<float> get_adv_action(const std::vector<float>& obs, float noise);
    std::vector<float> get_opt_action(const std::vector<float>& obs, float noise);
    void update(const ReplayBatch& batch);
    void polyak_update();

    // Transformer data
    struct DetectorSample {
        std::vector<std::vector<float>> obs_seq;
        float dist;
        float alpha;
    };
    void collect_detector_data(const std::vector<std::vector<float>>& obs_seq,
                               bool has_dist, float ep_return);
    void train_detector(int batch_size = 64);

    MuJoCoEnv env_;
    torch::Device device_;
    float act_limit_, dist_limit_;
    int64_t obs_dim_, act_dim_;

    MLPActor pi_rob_{nullptr}, pi_adv_{nullptr};
    MLPActor pi_rob_targ_{nullptr}, pi_adv_targ_{nullptr};
    AdversarialMLPQFunction q_{nullptr}, q_targ_{nullptr};
    std::optional<MLPActor> pi_opt_;

    std::unique_ptr<torch::optim::Adam> pi_rob_optim_, pi_adv_optim_, q_optim_;

    ReplayBuffer buffer_;
    DatasetLogger dataset_logger_;

    // Transformer components
    bool use_transformer_;
    std::unique_ptr<TransformerDisturbanceDetector> detector_;
    std::unique_ptr<HistoryBuffer> history_buf_;
    std::unique_ptr<AdaptiveControllerBlender> blender_;
    std::unique_ptr<DisturbanceDetectorTrainer> det_trainer_;
    std::vector<DetectorSample> transformer_data_;

    AdversarialDDPGConfig cfg_;
    std::string csv_path_;
};

} // namespace rzsm
