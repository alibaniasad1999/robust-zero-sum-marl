#include <rzsm/detector/trainer.hpp>

namespace rzsm {

DisturbanceDetectorTrainer::DisturbanceDetectorTrainer(
    TransformerDisturbanceDetector model, double learning_rate)
    : model_(model),
      optimizer_(model_->parameters(),
                 torch::optim::AdamOptions(learning_rate))
{}

std::unordered_map<std::string, float>
DisturbanceDetectorTrainer::train_step(
    torch::Tensor obs_seq, torch::Tensor true_dist,
    torch::Tensor true_alpha)
{
    model_->train();
    optimizer_.zero_grad();

    auto [pred_dist, pred_alpha] = model_->forward(obs_seq);

    auto loss_dist = torch::nn::functional::binary_cross_entropy(
        pred_dist, true_dist);
    auto loss_alpha = torch::nn::functional::mse_loss(
        pred_alpha, true_alpha);
    auto loss = loss_dist + 2.0 * loss_alpha;

    loss.backward();
    torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
    optimizer_.step();

    return {
        {"loss_disturbance", loss_dist.item<float>()},
        {"loss_blending", loss_alpha.item<float>()},
        {"total_loss", loss.item<float>()},
    };
}

std::unordered_map<std::string, float>
DisturbanceDetectorTrainer::evaluate(
    torch::Tensor obs_seq, torch::Tensor true_dist,
    torch::Tensor true_alpha)
{
    model_->eval();
    torch::NoGradGuard no_grad;

    auto [pred_dist, pred_alpha] = model_->forward(obs_seq);

    auto loss_dist = torch::nn::functional::binary_cross_entropy(
        pred_dist, true_dist);
    auto loss_alpha = torch::nn::functional::mse_loss(
        pred_alpha, true_alpha);
    auto loss = loss_dist + 2.0 * loss_alpha;

    return {
        {"loss_disturbance", loss_dist.item<float>()},
        {"loss_blending", loss_alpha.item<float>()},
        {"total_loss", loss.item<float>()},
    };
}

} // namespace rzsm
