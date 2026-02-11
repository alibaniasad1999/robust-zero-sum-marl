#pragma once
#include <torch/torch.h>
#include <rzsm/detector/transformer.hpp>
#include <string>
#include <unordered_map>

namespace rzsm {

class DisturbanceDetectorTrainer {
public:
    DisturbanceDetectorTrainer(TransformerDisturbanceDetector model,
                               double learning_rate = 1e-4);

    std::unordered_map<std::string, float> train_step(
        torch::Tensor obs_seq, torch::Tensor true_dist,
        torch::Tensor true_alpha);

    std::unordered_map<std::string, float> evaluate(
        torch::Tensor obs_seq, torch::Tensor true_dist,
        torch::Tensor true_alpha);

private:
    TransformerDisturbanceDetector model_;
    torch::optim::Adam optimizer_;
};

} // namespace rzsm
