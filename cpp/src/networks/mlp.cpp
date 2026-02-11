#include <rzsm/networks/mlp.hpp>

namespace rzsm {

torch::nn::Sequential build_mlp(
    const std::vector<int64_t>& sizes,
    ActivationType activation,
    ActivationType output_activation)
{
    torch::nn::Sequential seq;
    for (size_t j = 0; j + 1 < sizes.size(); ++j) {
        seq->push_back(torch::nn::Linear(sizes[j], sizes[j + 1]));
        ActivationType act = (j < sizes.size() - 2) ? activation : output_activation;
        push_activation(seq, act);
    }
    return seq;
}

// ── MLPActor ─────────────────────────────────────────────────────

MLPActorImpl::MLPActorImpl(
    int64_t obs_dim, int64_t act_dim,
    const std::vector<int64_t>& hidden_sizes,
    ActivationType activation, double act_limit)
    : obs_dim_(obs_dim), act_dim_(act_dim),
      hidden_sizes_(hidden_sizes), activation_(activation),
      act_limit_(act_limit)
{
    std::vector<int64_t> sizes = {obs_dim};
    sizes.insert(sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
    sizes.push_back(act_dim);
    pi_ = build_mlp(sizes, activation, ActivationType::Tanh);
    register_module("pi", pi_);
}

torch::Tensor MLPActorImpl::forward(torch::Tensor obs) {
    return act_limit_ * pi_->forward(obs);
}

// ── MLPQFunction ─────────────────────────────────────────────────

MLPQFunctionImpl::MLPQFunctionImpl(
    int64_t obs_dim, int64_t act_dim,
    const std::vector<int64_t>& hidden_sizes,
    ActivationType activation)
    : obs_dim_(obs_dim), act_dim_(act_dim),
      hidden_sizes_(hidden_sizes), activation_(activation)
{
    std::vector<int64_t> sizes = {obs_dim + act_dim};
    sizes.insert(sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
    sizes.push_back(1);
    q_ = build_mlp(sizes, activation);
    register_module("q", q_);
}

torch::Tensor MLPQFunctionImpl::forward(torch::Tensor obs, torch::Tensor act) {
    return q_->forward(torch::cat({obs, act}, /*dim=*/-1)).squeeze(-1);
}

// ── AdversarialMLPQFunction ──────────────────────────────────────

AdversarialMLPQFunctionImpl::AdversarialMLPQFunctionImpl(
    int64_t obs_dim, int64_t act_dim, int64_t dist_dim,
    const std::vector<int64_t>& hidden_sizes,
    ActivationType activation)
    : obs_dim_(obs_dim), act_dim_(act_dim), dist_dim_(dist_dim),
      hidden_sizes_(hidden_sizes), activation_(activation)
{
    std::vector<int64_t> sizes = {obs_dim + act_dim + dist_dim};
    sizes.insert(sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
    sizes.push_back(1);
    q_ = build_mlp(sizes, activation);
    register_module("q", q_);
}

torch::Tensor AdversarialMLPQFunctionImpl::forward(
    torch::Tensor obs, torch::Tensor act, torch::Tensor act_d)
{
    return q_->forward(torch::cat({obs, act, act_d}, /*dim=*/-1)).squeeze(-1);
}

} // namespace rzsm
