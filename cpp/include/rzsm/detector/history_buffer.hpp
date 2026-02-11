#pragma once
#include <torch/torch.h>
#include <deque>
#include <vector>

namespace rzsm {

class HistoryBuffer {
public:
    HistoryBuffer(int64_t obs_dim, int64_t max_length = 20,
                  torch::Device device = torch::kCPU);

    void reset();
    void add(const float* obs);
    torch::Tensor get_sequence(); // (1, L, obs_dim)

private:
    int64_t obs_dim_, max_length_;
    torch::Device device_;
    std::deque<std::vector<float>> buffer_;
};

} // namespace rzsm
