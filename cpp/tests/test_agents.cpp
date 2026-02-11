// Tests for rzsm::DDPGAgent and rzsm::AdversarialDDPGAgent
// Mirrors Python tests/test_agents.py (construction and config only — full
// training requires MuJoCo assets and is tested via integration tests)

#include <gtest/gtest.h>
#include <rzsm/agents/ddpg_agent.hpp>
#include <rzsm/agents/adversarial_ddpg_agent.hpp>
#include <rzsm/common.hpp>
#include <filesystem>

namespace fs = std::filesystem;
using namespace rzsm;

// ── DDPGConfig ──────────────────────────────────────────────────────

TEST(DDPGConfig, DefaultValues) {
    DDPGConfig cfg;
    EXPECT_EQ(cfg.epochs, 100);
    EXPECT_FLOAT_EQ(cfg.gamma, 0.99f);
    EXPECT_FLOAT_EQ(cfg.polyak, 0.995f);
    EXPECT_FLOAT_EQ(cfg.pi_lr, 1e-3f);
    EXPECT_FLOAT_EQ(cfg.q_lr, 1e-3f);
    EXPECT_EQ(cfg.batch_size, 1024);
    EXPECT_FLOAT_EQ(cfg.act_noise, 0.1f);
    EXPECT_EQ(cfg.start_steps, 5000);
    EXPECT_EQ(cfg.update_after, 10000);
    EXPECT_EQ(cfg.device, "auto");
}

TEST(DDPGConfig, CustomValues) {
    DDPGConfig cfg;
    cfg.epochs = 50;
    cfg.gamma = 0.95f;
    cfg.hidden_sizes = {128, 128};
    EXPECT_EQ(cfg.epochs, 50);
    EXPECT_FLOAT_EQ(cfg.gamma, 0.95f);
    EXPECT_EQ(cfg.hidden_sizes.size(), 2u);
    EXPECT_EQ(cfg.hidden_sizes[0], 128);
}

// ── AdversarialDDPGConfig ───────────────────────────────────────────

TEST(AdversarialDDPGConfig, DefaultValues) {
    AdversarialDDPGConfig cfg;
    EXPECT_EQ(cfg.epochs, 100);
    EXPECT_FLOAT_EQ(cfg.disturbance_ratio, 0.1f);
    EXPECT_FLOAT_EQ(cfg.disturbance_probability, 0.5f);
    EXPECT_TRUE(cfg.use_transformer);
    EXPECT_EQ(cfg.transformer_seq_len, 20);
    EXPECT_EQ(cfg.transformer_d_model, 128);
    EXPECT_EQ(cfg.transformer_nhead, 4);
    EXPECT_EQ(cfg.transformer_layers, 3);
    EXPECT_TRUE(cfg.pi_opt_path.empty());
}

TEST(AdversarialDDPGConfig, TransformerCanBeDisabled) {
    AdversarialDDPGConfig cfg;
    cfg.use_transformer = false;
    EXPECT_FALSE(cfg.use_transformer);
}

// ── DDPGAgent Construction ──────────────────────────────────────────

TEST(DDPGAgent, ConstructsWithAnt) {
    DDPGConfig cfg;
    cfg.device = "cpu";
    cfg.epochs = 0;  // no training

    auto env_fn = []() { return MuJoCoEnv("Ant"); };
    EXPECT_NO_THROW(DDPGAgent agent(env_fn, cfg));
}

TEST(DDPGAgent, ConstructsWithHumanoid) {
    DDPGConfig cfg;
    cfg.device = "cpu";
    cfg.epochs = 0;

    auto env_fn = []() { return MuJoCoEnv("Humanoid"); };
    EXPECT_NO_THROW(DDPGAgent agent(env_fn, cfg));
}

// ── AdversarialDDPGAgent Construction ───────────────────────────────

TEST(AdversarialDDPGAgent, ConstructsWithAnt) {
    AdversarialDDPGConfig cfg;
    cfg.device = "cpu";
    cfg.epochs = 0;
    cfg.use_transformer = false;  // avoid extra alloc for simple test

    auto env_fn = []() { return MuJoCoEnv("Ant"); };
    EXPECT_NO_THROW(AdversarialDDPGAgent agent(env_fn, cfg));
}

TEST(AdversarialDDPGAgent, ConstructsWithTransformer) {
    AdversarialDDPGConfig cfg;
    cfg.device = "cpu";
    cfg.epochs = 0;
    cfg.use_transformer = true;

    auto env_fn = []() { return MuJoCoEnv("Ant"); };
    EXPECT_NO_THROW(AdversarialDDPGAgent agent(env_fn, cfg));
}

// ── DDPGAgent Save/Load ─────────────────────────────────────────────

TEST(DDPGAgent, SaveCreatesFiles) {
    DDPGConfig cfg;
    cfg.device = "cpu";
    cfg.epochs = 0;

    auto tmp = fs::temp_directory_path() / ("rzsm_ddpg_save_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count()));
    cfg.log_dir = tmp.string();

    auto env_fn = []() { return MuJoCoEnv("Ant"); };
    DDPGAgent agent(env_fn, cfg);
    agent.save(tmp.string());

    // Should have created .pt files
    bool found_files = false;
    if (fs::exists(tmp)) {
        for (auto& e : fs::recursive_directory_iterator(tmp)) {
            if (e.path().extension() == ".pt") {
                found_files = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found_files);

    // Cleanup
    if (fs::exists(tmp)) fs::remove_all(tmp);
}

// ── Seed Reproducibility ────────────────────────────────────────────

TEST(Seed, SetSeedReproducible) {
    set_seed(42);
    auto t1 = torch::randn({3, 3});
    set_seed(42);
    auto t2 = torch::randn({3, 3});
    EXPECT_TRUE(torch::allclose(t1, t2));
}

TEST(Seed, DifferentSeedsDifferent) {
    set_seed(42);
    auto t1 = torch::randn({3, 3});
    set_seed(99);
    auto t2 = torch::randn({3, 3});
    EXPECT_FALSE(torch::allclose(t1, t2));
}

// ── Device Selection ────────────────────────────────────────────────

TEST(SelectDevice, CpuString) {
    auto dev = select_device("cpu");
    EXPECT_EQ(dev.type(), torch::kCPU);
}

TEST(SelectDevice, AutoDoesNotThrow) {
    EXPECT_NO_THROW(select_device("auto"));
}
