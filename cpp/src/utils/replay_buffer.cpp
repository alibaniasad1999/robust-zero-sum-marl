#include <rzsm/utils/replay_buffer.hpp>
#include <cstring>

namespace rzsm {

ReplayBuffer::ReplayBuffer(int64_t capacity, int64_t obs_dim,
                           int64_t total_act, torch::Device device)
    : capacity_(capacity), obs_dim_(obs_dim), total_act_(total_act),
      device_(device),
      obs_buf_(capacity * obs_dim, 0.0f),
      next_obs_buf_(capacity * obs_dim, 0.0f),
      act_buf_(capacity * total_act, 0.0f),
      rew_buf_(capacity, 0.0f),
      term_buf_(capacity, 0),
      trunc_buf_(capacity, 0)
{
    assert(capacity > 0);
}

void ReplayBuffer::add(const float* obs, const float* next_obs,
                       const float* act, float rew,
                       bool terminated, bool truncated) {
    int64_t p = ptr_;
    std::memcpy(&obs_buf_[p * obs_dim_], obs, obs_dim_ * sizeof(float));
    std::memcpy(&next_obs_buf_[p * obs_dim_], next_obs, obs_dim_ * sizeof(float));
    std::memcpy(&act_buf_[p * total_act_], act, total_act_ * sizeof(float));
    rew_buf_[p] = rew;
    term_buf_[p] = terminated ? 1 : 0;
    trunc_buf_[p] = truncated ? 1 : 0;

    ptr_ = (p + 1) % capacity_;
    size_ = std::min(capacity_, size_ + 1);
}

ReplayBatch ReplayBuffer::sample(int64_t batch_size) {
    assert(batch_size > 0);
    assert(size_ >= batch_size);

    auto idx = torch::randint(0, size_, {batch_size}, torch::kLong);
    auto idx_a = idx.accessor<int64_t, 1>();

    auto obs_t = torch::zeros({batch_size, obs_dim_}, torch::kFloat32);
    auto next_obs_t = torch::zeros({batch_size, obs_dim_}, torch::kFloat32);
    auto act_t = torch::zeros({batch_size, total_act_}, torch::kFloat32);
    auto rew_t = torch::zeros({batch_size, 1}, torch::kFloat32);
    auto term_t = torch::zeros({batch_size, 1}, torch::kBool);
    auto trunc_t = torch::zeros({batch_size, 1}, torch::kBool);

    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t j = idx_a[i];
        std::memcpy(obs_t.data_ptr<float>() + i * obs_dim_,
                     &obs_buf_[j * obs_dim_], obs_dim_ * sizeof(float));
        std::memcpy(next_obs_t.data_ptr<float>() + i * obs_dim_,
                     &next_obs_buf_[j * obs_dim_], obs_dim_ * sizeof(float));
        std::memcpy(act_t.data_ptr<float>() + i * total_act_,
                     &act_buf_[j * total_act_], total_act_ * sizeof(float));
        rew_t[i][0] = rew_buf_[j];
        term_t[i][0] = term_buf_[j] != 0;
        trunc_t[i][0] = trunc_buf_[j] != 0;
    }

    return ReplayBatch{
        obs_t.to(device_),
        next_obs_t.to(device_),
        act_t.to(device_),
        rew_t.to(device_),
        term_t.to(device_),
        trunc_t.to(device_),
    };
}

} // namespace rzsm
