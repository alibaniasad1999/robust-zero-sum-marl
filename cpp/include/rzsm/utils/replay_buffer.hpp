#pragma once
#include <torch/torch.h>
#include <cassert>
#include <cstdint>
#include <vector>

namespace rzsm {

struct ReplayBatch {
    torch::Tensor obs;        // [B, obs_dim]
    torch::Tensor next_obs;   // [B, obs_dim]
    torch::Tensor act;        // [B, total_act]
    torch::Tensor rew;        // [B, 1]
    torch::Tensor terminated; // [B, 1] bool
    torch::Tensor truncated;  // [B, 1] bool

    torch::Tensor done() const { return terminated.logical_or(truncated); }
};

class ReplayBuffer {
public:
    ReplayBuffer(int64_t capacity, int64_t obs_dim, int64_t total_act,
                 torch::Device device);

    void add(const float* obs, const float* next_obs, const float* act,
             float rew, bool terminated, bool truncated);

    ReplayBatch sample(int64_t batch_size);
    int64_t size() const { return size_; }
    int64_t total_act() const { return total_act_; }

private:
    int64_t capacity_, obs_dim_, total_act_;
    torch::Device device_;

    std::vector<float> obs_buf_, next_obs_buf_, act_buf_, rew_buf_;
    std::vector<uint8_t> term_buf_, trunc_buf_;

    int64_t ptr_ = 0;
    int64_t size_ = 0;
};

} // namespace rzsm
