#pragma once
#include <torch/torch.h>
#include <rzsm/common.hpp>
#include <vector>

namespace rzsm {

// Build an MLP from layer sizes and activation types
torch::nn::Sequential build_mlp(
    const std::vector<int64_t>& sizes,
    ActivationType activation,
    ActivationType output_activation = ActivationType::Identity);

// ── MLPActor ─────────────────────────────────────────────────────
// Deterministic policy: obs -> tanh-squashed action * act_limit
struct MLPActorImpl : torch::nn::Module {
    MLPActorImpl(int64_t obs_dim, int64_t act_dim,
                 const std::vector<int64_t>& hidden_sizes,
                 ActivationType activation, double act_limit);

    torch::Tensor forward(torch::Tensor obs);

    // Expose for cloning
    int64_t obs_dim_, act_dim_;
    std::vector<int64_t> hidden_sizes_;
    ActivationType activation_;
    double act_limit_;
    torch::nn::Sequential pi_{nullptr};
};
TORCH_MODULE(MLPActor);

// ── MLPQFunction ─────────────────────────────────────────────────
// Q(obs, act) -> scalar
struct MLPQFunctionImpl : torch::nn::Module {
    MLPQFunctionImpl(int64_t obs_dim, int64_t act_dim,
                     const std::vector<int64_t>& hidden_sizes,
                     ActivationType activation);

    torch::Tensor forward(torch::Tensor obs, torch::Tensor act);

    int64_t obs_dim_, act_dim_;
    std::vector<int64_t> hidden_sizes_;
    ActivationType activation_;
    torch::nn::Sequential q_{nullptr};
};
TORCH_MODULE(MLPQFunction);

// ── AdversarialMLPQFunction ──────────────────────────────────────
// Q(obs, act_ctrl, act_dist) -> scalar
struct AdversarialMLPQFunctionImpl : torch::nn::Module {
    AdversarialMLPQFunctionImpl(int64_t obs_dim, int64_t act_dim,
                                 int64_t dist_dim,
                                 const std::vector<int64_t>& hidden_sizes,
                                 ActivationType activation);

    torch::Tensor forward(torch::Tensor obs, torch::Tensor act,
                          torch::Tensor act_d);

    int64_t obs_dim_, act_dim_, dist_dim_;
    std::vector<int64_t> hidden_sizes_;
    ActivationType activation_;
    torch::nn::Sequential q_{nullptr};
};
TORCH_MODULE(AdversarialMLPQFunction);

} // namespace rzsm
