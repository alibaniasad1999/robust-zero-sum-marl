// Tests for rzsm::MuJoCoEnv
// Tests the Ant and Humanoid environment wrappers

#include <gtest/gtest.h>
#include <rzsm/envs/mujoco_env.hpp>
#include <cmath>

using namespace rzsm;

// ── Ant Environment ─────────────────────────────────────────────────

class AntEnvTest : public ::testing::Test {
protected:
    std::unique_ptr<MuJoCoEnv> env;
    void SetUp() override {
        env = std::make_unique<MuJoCoEnv>("Ant");
    }
};

TEST_F(AntEnvTest, ResetReturnsObs) {
    auto obs = env->reset(42);
    auto space = env->observation_space();
    EXPECT_EQ(static_cast<int64_t>(obs.size()), space.dim);
}

TEST_F(AntEnvTest, ObservationSpaceDim) {
    auto space = env->observation_space();
    EXPECT_GT(space.dim, 0);
}

TEST_F(AntEnvTest, ActionSpaceDim) {
    auto space = env->action_space();
    EXPECT_GT(space.dim, 0);
    EXPECT_FLOAT_EQ(space.low, -1.0f);
    EXPECT_FLOAT_EQ(space.high, 1.0f);
}

TEST_F(AntEnvTest, StepReturnsValidResult) {
    env->reset(42);
    auto action = env->sample_action();
    auto result = env->step(action.data());

    auto space = env->observation_space();
    EXPECT_EQ(static_cast<int64_t>(result.obs.size()), space.dim);
    EXPECT_TRUE(std::isfinite(result.reward));
}

TEST_F(AntEnvTest, SampleActionInBounds) {
    auto action = env->sample_action();
    auto space = env->action_space();
    EXPECT_EQ(static_cast<int64_t>(action.size()), space.dim);
    for (float a : action) {
        EXPECT_GE(a, -1.0f);
        EXPECT_LE(a, 1.0f);
    }
}

TEST_F(AntEnvTest, MultipleSteps) {
    env->reset(0);
    for (int i = 0; i < 10; ++i) {
        auto action = env->sample_action();
        auto result = env->step(action.data());
        if (result.terminated || result.truncated) break;
        EXPECT_TRUE(std::isfinite(result.reward));
    }
}

TEST_F(AntEnvTest, ResetDeterminsticWithSeed) {
    auto obs1 = env->reset(123);
    auto obs2 = env->reset(123);
    EXPECT_EQ(obs1.size(), obs2.size());
    for (size_t i = 0; i < obs1.size(); ++i)
        EXPECT_FLOAT_EQ(obs1[i], obs2[i]);
}

TEST_F(AntEnvTest, DifferentSeedsGiveDifferentObs) {
    auto obs1 = env->reset(1);
    auto obs2 = env->reset(2);
    bool any_different = false;
    for (size_t i = 0; i < obs1.size(); ++i) {
        if (std::abs(obs1[i] - obs2[i]) > 1e-6f) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST_F(AntEnvTest, EnvName) {
    EXPECT_EQ(env->env_name(), "Ant");
}

// ── Humanoid Environment ────────────────────────────────────────────

class HumanoidEnvTest : public ::testing::Test {
protected:
    std::unique_ptr<MuJoCoEnv> env;
    void SetUp() override {
        env = std::make_unique<MuJoCoEnv>("Humanoid");
    }
};

TEST_F(HumanoidEnvTest, ResetReturnsObs) {
    auto obs = env->reset(42);
    auto space = env->observation_space();
    EXPECT_EQ(static_cast<int64_t>(obs.size()), space.dim);
}

TEST_F(HumanoidEnvTest, ObsDimDifferentFromAnt) {
    MuJoCoEnv ant("Ant");
    EXPECT_NE(env->observation_space().dim, ant.observation_space().dim);
}

TEST_F(HumanoidEnvTest, ActionDimDifferentFromAnt) {
    MuJoCoEnv ant("Ant");
    EXPECT_NE(env->action_space().dim, ant.action_space().dim);
}

TEST_F(HumanoidEnvTest, StepReturnsValidResult) {
    env->reset(42);
    auto action = env->sample_action();
    auto result = env->step(action.data());

    auto space = env->observation_space();
    EXPECT_EQ(static_cast<int64_t>(result.obs.size()), space.dim);
    EXPECT_TRUE(std::isfinite(result.reward));
}

TEST_F(HumanoidEnvTest, EnvName) {
    EXPECT_EQ(env->env_name(), "Humanoid");
}

// ── Invalid Environment ─────────────────────────────────────────────

TEST(MuJoCoEnvInvalid, ThrowsOnUnknownEnv) {
    EXPECT_THROW(MuJoCoEnv("UnknownRobot"), std::runtime_error);
}

// ── Move Semantics ──────────────────────────────────────────────────

TEST(MuJoCoEnvMove, MoveConstructor) {
    MuJoCoEnv env1("Ant");
    auto obs1 = env1.reset(42);

    MuJoCoEnv env2(std::move(env1));
    auto obs2 = env2.reset(42);

    EXPECT_EQ(obs1.size(), obs2.size());
    for (size_t i = 0; i < obs1.size(); ++i)
        EXPECT_FLOAT_EQ(obs1[i], obs2[i]);
}

TEST(MuJoCoEnvMove, MoveAssignment) {
    MuJoCoEnv env1("Ant");
    MuJoCoEnv env2("Humanoid");

    env2 = std::move(env1);
    EXPECT_EQ(env2.env_name(), "Ant");
    auto obs = env2.reset(0);
    EXPECT_GT(obs.size(), 0u);
}
