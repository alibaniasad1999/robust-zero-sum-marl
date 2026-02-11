#include <rzsm/detector/history_buffer.hpp>
#include <cstring>

namespace rzsm {

HistoryBuffer::HistoryBuffer(int64_t obs_dim, int64_t max_length,
                             torch::Device device)
    : obs_dim_(obs_dim), max_length_(max_length), device_(device)
{
    reset();
}

void HistoryBuffer::reset() {
    buffer_.clear();
    for (int64_t i = 0; i < max_length_; ++i) {
        buffer_.emplace_back(obs_dim_, 0.0f);
    }
}

void HistoryBuffer::add(const float* obs) {
    std::vector<float> v(obs, obs + obs_dim_);
    buffer_.push_back(std::move(v));
    if (static_cast<int64_t>(buffer_.size()) > max_length_) {
        buffer_.pop_front();
    }
}

torch::Tensor HistoryBuffer::get_sequence() {
    auto seq = torch::zeros({1, max_length_, obs_dim_}, torch::kFloat32);
    auto acc = seq.accessor<float, 3>();
    for (int64_t i = 0; i < max_length_; ++i) {
        for (int64_t j = 0; j < obs_dim_; ++j) {
            acc[0][i][j] = buffer_[i][j];
        }
    }
    return seq.to(device_);
}

} // namespace rzsm
