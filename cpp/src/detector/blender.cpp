#include <rzsm/detector/blender.hpp>

namespace rzsm {

AdaptiveControllerBlender::AdaptiveControllerBlender(
    TransformerDisturbanceDetector detector,
    HistoryBuffer& history_buffer,
    int smoothing_window)
    : detector_(detector), history_buffer_(history_buffer),
      smoothing_window_(smoothing_window)
{}

std::tuple<std::vector<float>, float, float>
AdaptiveControllerBlender::get_blended_action(
    const float* obs,
    const float* action_optimal,
    const float* action_robust,
    int64_t act_dim,
    bool use_smoothing)
{
    history_buffer_.add(obs);
    detector_->eval();

    float alpha_val, dist_val;
    {
        torch::NoGradGuard no_grad;
        auto obs_seq = history_buffer_.get_sequence();
        auto [dist_prob, alpha] = detector_->forward(obs_seq);
        alpha_val = alpha.item<float>();
        dist_val = dist_prob.item<float>();
    }

    if (use_smoothing) {
        alpha_history_.push_back(alpha_val);
        if (static_cast<int>(alpha_history_.size()) > smoothing_window_) {
            alpha_history_.pop_front();
        }
        float sum = std::accumulate(alpha_history_.begin(),
                                     alpha_history_.end(), 0.0f);
        alpha_val = sum / static_cast<float>(alpha_history_.size());
    }

    std::vector<float> blended(act_dim);
    for (int64_t i = 0; i < act_dim; ++i) {
        blended[i] = alpha_val * action_optimal[i] +
                      (1.0f - alpha_val) * action_robust[i];
    }

    return {blended, alpha_val, dist_val};
}

void AdaptiveControllerBlender::reset() {
    history_buffer_.reset();
    alpha_history_.clear();
}

} // namespace rzsm
