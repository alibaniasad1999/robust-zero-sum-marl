#include <rzsm/detector/transformer.hpp>
#include <cmath>

namespace rzsm {

torch::Tensor TransformerDisturbanceDetectorImpl::sinusoidal_pe(
    int64_t max_len, int64_t d_model)
{
    auto pe = torch::zeros({max_len, d_model});
    auto position = torch::arange(0, max_len, torch::kFloat32).unsqueeze(1);
    auto div_term = torch::exp(
        torch::arange(0, d_model, 2, torch::kFloat32) *
        (-std::log(10000.0) / d_model));

    using namespace torch::indexing;
    pe.index_put_({Slice(), Slice(0, None, 2)}, torch::sin(position * div_term));
    pe.index_put_({Slice(), Slice(1, None, 2)}, torch::cos(position * div_term));
    return pe.unsqueeze(0); // (1, L, D)
}

TransformerDisturbanceDetectorImpl::TransformerDisturbanceDetectorImpl(
    int64_t obs_dim, int64_t sequence_length,
    int64_t d_model, int64_t nhead,
    int64_t num_layers, int64_t dim_feedforward,
    double dropout)
    : obs_dim_(obs_dim), sequence_length_(sequence_length), d_model_(d_model)
{
    input_projection_ = register_module("input_projection",
        torch::nn::Linear(obs_dim, d_model));

    positional_encoding_ = register_buffer("positional_encoding",
        sinusoidal_pe(sequence_length, d_model));

    auto layer_opts = torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
        .dim_feedforward(dim_feedforward)
        .dropout(dropout)
        .batch_first(true);
    auto encoder_layer = torch::nn::TransformerEncoderLayer(layer_opts);
    auto enc_opts = torch::nn::TransformerEncoderOptions(encoder_layer, num_layers);
    transformer_encoder_ = register_module("transformer_encoder",
        torch::nn::TransformerEncoder(enc_opts));

    disturbance_head_ = register_module("disturbance_head",
        torch::nn::Sequential(
            torch::nn::Linear(d_model, 64),
            torch::nn::ReLU(),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout)),
            torch::nn::Linear(64, 1),
            torch::nn::Sigmoid()));

    blending_head_ = register_module("blending_head",
        torch::nn::Sequential(
            torch::nn::Linear(d_model, 64),
            torch::nn::ReLU(),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout)),
            torch::nn::Linear(64, 32),
            torch::nn::ReLU(),
            torch::nn::Linear(32, 1),
            torch::nn::Sigmoid()));
}

std::tuple<torch::Tensor, torch::Tensor>
TransformerDisturbanceDetectorImpl::forward(torch::Tensor obs_sequence) {
    int64_t seq_len = obs_sequence.size(1);
    auto x = input_projection_->forward(obs_sequence);

    using namespace torch::indexing;
    x = x + positional_encoding_.index({Slice(), Slice(None, seq_len), Slice()});

    auto encoded = transformer_encoder_->forward(x);
    auto final_token = encoded.index({Slice(), -1, Slice()}); // (B, d_model)

    auto p = disturbance_head_->forward(final_token).squeeze(-1);
    auto alpha = blending_head_->forward(final_token).squeeze(-1);
    return {p, alpha};
}

} // namespace rzsm
