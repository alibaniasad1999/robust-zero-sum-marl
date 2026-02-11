#pragma once
#include <torch/torch.h>
#include <tuple>

namespace rzsm {

struct TransformerDisturbanceDetectorImpl : torch::nn::Module {
    TransformerDisturbanceDetectorImpl(
        int64_t obs_dim, int64_t sequence_length = 20,
        int64_t d_model = 128, int64_t nhead = 4,
        int64_t num_layers = 3, int64_t dim_feedforward = 256,
        double dropout = 0.1);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor obs_sequence);

    int64_t sequence_length() const { return sequence_length_; }
    int64_t obs_dim() const { return obs_dim_; }

private:
    static torch::Tensor sinusoidal_pe(int64_t max_len, int64_t d_model);

    int64_t obs_dim_, sequence_length_, d_model_;
    torch::nn::Linear input_projection_{nullptr};
    torch::Tensor positional_encoding_;
    torch::nn::TransformerEncoder transformer_encoder_{nullptr};
    torch::nn::Sequential disturbance_head_{nullptr};
    torch::nn::Sequential blending_head_{nullptr};
};
TORCH_MODULE(TransformerDisturbanceDetector);

} // namespace rzsm
