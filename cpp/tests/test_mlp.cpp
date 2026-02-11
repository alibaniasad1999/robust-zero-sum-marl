// Tests for rzsm::networks — MLP building blocks
// Mirrors Python tests/test_networks.py

#include <gtest/gtest.h>
#include <rzsm/networks/mlp.hpp>
#include <rzsm/common.hpp>

using namespace rzsm;

// ── build_mlp ───────────────────────────────────────────────────────

TEST(BuildMLP, OutputShape) {
    auto net = build_mlp({4, 32, 16, 2}, ActivationType::ReLU);
    auto x = torch::randn({8, 4});
    auto out = net->forward(x);
    EXPECT_EQ(out.size(0), 8);
    EXPECT_EQ(out.size(1), 2);
}

TEST(BuildMLP, SingleHidden) {
    auto net = build_mlp({3, 5, 1}, ActivationType::Tanh);
    auto x = torch::randn({1, 3});
    auto out = net->forward(x);
    EXPECT_EQ(out.size(0), 1);
    EXPECT_EQ(out.size(1), 1);
}

TEST(BuildMLP, OutputActivation) {
    auto net = build_mlp({4, 8, 1}, ActivationType::ReLU, ActivationType::Sigmoid);
    auto x = torch::randn({10, 4});
    auto out = net->forward(x);
    EXPECT_TRUE((out >= 0).all().item<bool>());
    EXPECT_TRUE((out <= 1).all().item<bool>());
}

// ── MLPActor ────────────────────────────────────────────────────────

class MLPActorTest : public ::testing::Test {
protected:
    MLPActor actor{nullptr};
    void SetUp() override {
        actor = MLPActor(8, 3, {64, 64}, ActivationType::Tanh, 2.0);
    }
};

TEST_F(MLPActorTest, OutputShape) {
    auto obs = torch::randn({4, 8});
    auto out = actor->forward(obs);
    EXPECT_EQ(out.size(0), 4);
    EXPECT_EQ(out.size(1), 3);
}

TEST_F(MLPActorTest, OutputBounded) {
    auto obs = torch::randn({100, 8});
    auto out = actor->forward(obs);
    EXPECT_LE(out.abs().max().item<float>(), 2.0f + 1e-6f);
}

TEST_F(MLPActorTest, SingleObs) {
    auto obs = torch::randn({8});
    auto out = actor->forward(obs);
    EXPECT_EQ(out.size(0), 3);
}

TEST_F(MLPActorTest, Deterministic) {
    auto obs = torch::randn({1, 8});
    auto a1 = actor->forward(obs);
    auto a2 = actor->forward(obs);
    EXPECT_TRUE(torch::allclose(a1, a2));
}

// ── MLPQFunction ────────────────────────────────────────────────────

class MLPQFunctionTest : public ::testing::Test {
protected:
    MLPQFunction q{nullptr};
    void SetUp() override {
        q = MLPQFunction(8, 3, {64, 64}, ActivationType::ReLU);
    }
};

TEST_F(MLPQFunctionTest, OutputShape) {
    auto obs = torch::randn({4, 8});
    auto act = torch::randn({4, 3});
    auto out = q->forward(obs, act);
    EXPECT_EQ(out.size(0), 4);
}

TEST_F(MLPQFunctionTest, ScalarOutput) {
    auto obs = torch::randn({1, 8});
    auto act = torch::randn({1, 3});
    auto out = q->forward(obs, act);
    EXPECT_EQ(out.size(0), 1);
}

TEST_F(MLPQFunctionTest, GradientFlows) {
    auto obs = torch::randn({4, 8}).requires_grad_(true);
    auto act = torch::randn({4, 3}).requires_grad_(true);
    auto loss = q->forward(obs, act).mean();
    loss.backward();
    EXPECT_TRUE(obs.grad().defined());
    EXPECT_TRUE(act.grad().defined());
}

// ── AdversarialMLPQFunction ─────────────────────────────────────────

class AdversarialQTest : public ::testing::Test {
protected:
    AdversarialMLPQFunction q{nullptr};
    void SetUp() override {
        q = AdversarialMLPQFunction(8, 3, 3, {64, 64}, ActivationType::ReLU);
    }
};

TEST_F(AdversarialQTest, OutputShape) {
    auto obs = torch::randn({4, 8});
    auto act = torch::randn({4, 3});
    auto act_d = torch::randn({4, 3});
    auto out = q->forward(obs, act, act_d);
    EXPECT_EQ(out.size(0), 4);
}

TEST_F(AdversarialQTest, DifferentDistDim) {
    auto q2 = AdversarialMLPQFunction(8, 3, 5, {32}, ActivationType::ReLU);
    auto out = q2->forward(torch::randn({2, 8}), torch::randn({2, 3}), torch::randn({2, 5}));
    EXPECT_EQ(out.size(0), 2);
}

TEST_F(AdversarialQTest, GradientFlowsToAllInputs) {
    auto obs = torch::randn({4, 8}).requires_grad_(true);
    auto act = torch::randn({4, 3}).requires_grad_(true);
    auto act_d = torch::randn({4, 3}).requires_grad_(true);
    auto loss = q->forward(obs, act, act_d).mean();
    loss.backward();
    EXPECT_TRUE(obs.grad().defined());
    EXPECT_TRUE(act.grad().defined());
    EXPECT_TRUE(act_d.grad().defined());
}

// ── count_vars ──────────────────────────────────────────────────────

TEST(CountVars, Linear) {
    torch::nn::Linear layer(4, 3);
    EXPECT_EQ(count_vars(*layer), 15);  // 4*3 + 3 = 15
}

TEST(CountVars, Actor) {
    MLPActor actor(4, 2, {16}, ActivationType::Tanh, 1.0);
    EXPECT_GT(count_vars(*actor), 0);
}
