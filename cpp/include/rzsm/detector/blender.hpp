#pragma once
#include <torch/torch.h>
#include <rzsm/detector/transformer.hpp>
#include <rzsm/detector/history_buffer.hpp>
#include <deque>
#include <numeric>
#include <tuple>
#include <vector>

namespace rzsm {

class AdaptiveControllerBlender {
public:
    AdaptiveControllerBlender(TransformerDisturbanceDetector detector,
                              HistoryBuffer& history_buffer,
                              int smoothing_window = 5);

    // Returns: {blended_action, alpha_val, dist_val}
    std::tuple<std::vector<float>, float, float> get_blended_action(
        const float* obs,
        const float* action_optimal,
        const float* action_robust,
        int64_t act_dim,
        bool use_smoothing = true);

    void reset();

private:
    TransformerDisturbanceDetector detector_;
    HistoryBuffer& history_buffer_;
    std::deque<float> alpha_history_;
    int smoothing_window_;
};

} // namespace rzsm
