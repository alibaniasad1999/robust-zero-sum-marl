#pragma once
#include <torch/torch.h>
#include <random>
#include <sstream>
#include <string>

namespace rzsm {

// Activation type enum (replaces Python's dynamic class passing)
enum class ActivationType { ReLU, Tanh, Sigmoid, Identity };

// Device selection: "auto" picks CUDA if available
inline torch::Device select_device(const std::string& device_str = "auto") {
    if (device_str == "auto") {
        return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    }
    return torch::Device(device_str);
}

// Global RNG for exploration noise
inline std::mt19937& global_rng() {
    static std::mt19937 rng(0);
    return rng;
}

inline void set_seed(int seed) {
    torch::manual_seed(seed);
    global_rng().seed(seed);
}

// Count total parameters in a module
inline int64_t count_vars(const torch::nn::Module& module) {
    int64_t total = 0;
    for (const auto& p : module.parameters()) {
        total += p.numel();
    }
    return total;
}

// Add activation module to a Sequential
inline void push_activation(torch::nn::Sequential& seq, ActivationType act) {
    switch (act) {
        case ActivationType::ReLU:     seq->push_back(torch::nn::ReLU()); break;
        case ActivationType::Tanh:     seq->push_back(torch::nn::Tanh()); break;
        case ActivationType::Sigmoid:  seq->push_back(torch::nn::Sigmoid()); break;
        case ActivationType::Identity: seq->push_back(torch::nn::Identity()); break;
    }
}

// Clone module parameters from src into dst (must have same architecture)
inline void copy_params(const torch::nn::Module& src, torch::nn::Module& dst) {
    auto src_params = src.parameters();
    auto dst_params = dst.parameters();
    assert(src_params.size() == dst_params.size());
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < src_params.size(); ++i) {
        dst_params[i].data().copy_(src_params[i].data());
    }
    // Also copy buffers
    auto src_bufs = src.buffers();
    auto dst_bufs = dst.buffers();
    for (size_t i = 0; i < src_bufs.size(); ++i) {
        dst_bufs[i].data().copy_(src_bufs[i].data());
    }
}

// Freeze all parameters (set requires_grad = false)
inline void freeze(torch::nn::Module& module) {
    for (auto& p : module.parameters()) {
        p.set_requires_grad(false);
    }
}

} // namespace rzsm
