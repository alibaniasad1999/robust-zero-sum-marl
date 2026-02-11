// Tests for rzsm::detector — Transformer, HistoryBuffer, Trainer, Blender
// Mirrors Python tests/test_detector.py

#include <gtest/gtest.h>
#include <rzsm/detector/transformer.hpp>
#include <rzsm/detector/history_buffer.hpp>
#include <rzsm/detector/trainer.hpp>
#include <rzsm/detector/blender.hpp>
#include <vector>
#include <cmath>

using namespace rzsm;

// ── TransformerDisturbanceDetector ──────────────────────────────────

class DetectorTest : public ::testing::Test {
protected:
    TransformerDisturbanceDetector det{nullptr};
    void SetUp() override {
        det = TransformerDisturbanceDetector(8, 10, 32, 2, 1, 64, 0.0);
    }
};

TEST_F(DetectorTest, OutputShapes) {
    auto x = torch::randn({4, 10, 8});
    auto [dist_prob, alpha] = det->forward(x);
    EXPECT_EQ(dist_prob.size(0), 4);
    EXPECT_EQ(alpha.size(0), 4);
}

TEST_F(DetectorTest, OutputsIn01) {
    auto x = torch::randn({16, 10, 8});
    det->eval();
    torch::NoGradGuard no_grad;
    auto [dist_prob, alpha] = det->forward(x);
    EXPECT_TRUE((dist_prob >= 0).all().item<bool>());
    EXPECT_TRUE((dist_prob <= 1).all().item<bool>());
    EXPECT_TRUE((alpha >= 0).all().item<bool>());
    EXPECT_TRUE((alpha <= 1).all().item<bool>());
}

TEST_F(DetectorTest, ShorterSequence) {
    auto x = torch::randn({2, 5, 8});
    auto [dist_prob, alpha] = det->forward(x);
    EXPECT_EQ(dist_prob.size(0), 2);
}

TEST_F(DetectorTest, SingleSample) {
    auto x = torch::randn({1, 10, 8});
    auto [dist_prob, alpha] = det->forward(x);
    EXPECT_EQ(dist_prob.size(0), 1);
}

TEST_F(DetectorTest, GradientFlows) {
    auto x = torch::randn({2, 10, 8}).requires_grad_(true);
    auto [dist_prob, alpha] = det->forward(x);
    auto loss = dist_prob.sum() + alpha.sum();
    loss.backward();
    EXPECT_TRUE(x.grad().defined());
}

TEST_F(DetectorTest, DifferentObsDim) {
    auto det2 = TransformerDisturbanceDetector(27, 20, 64, 4, 2);
    auto x = torch::randn({3, 20, 27});
    auto [dist_prob, alpha] = det2->forward(x);
    EXPECT_EQ(dist_prob.size(0), 3);
}

// ── HistoryBuffer ───────────────────────────────────────────────────

TEST(HistoryBuffer, InitialStateAllZeros) {
    HistoryBuffer hb(4, 10);
    auto seq = hb.get_sequence();
    EXPECT_EQ(seq.size(0), 1);
    EXPECT_EQ(seq.size(1), 10);
    EXPECT_EQ(seq.size(2), 4);
    EXPECT_TRUE((seq == 0).all().item<bool>());
}

TEST(HistoryBuffer, AddFillsBuffer) {
    HistoryBuffer hb(3, 5);
    for (int i = 0; i < 5; ++i) {
        float val = static_cast<float>(i + 1);
        std::vector<float> obs = {val, val, val};
        hb.add(obs.data());
    }
    auto seq = hb.get_sequence();
    EXPECT_EQ(seq.size(0), 1);
    EXPECT_EQ(seq.size(1), 5);
    EXPECT_EQ(seq.size(2), 3);
    // Last entry should be [5, 5, 5]
    auto last = seq.index({0, -1});
    EXPECT_TRUE(torch::allclose(last, torch::tensor({5.0f, 5.0f, 5.0f})));
}

TEST(HistoryBuffer, OverflowKeepsRecent) {
    HistoryBuffer hb(2, 3);
    for (int i = 0; i < 10; ++i) {
        float val = static_cast<float>(i);
        std::vector<float> obs = {val, val};
        hb.add(obs.data());
    }
    auto seq = hb.get_sequence();
    // Should contain last 3: [7,7], [8,8], [9,9]
    EXPECT_TRUE(torch::allclose(seq.index({0, 0}), torch::tensor({7.0f, 7.0f})));
    EXPECT_TRUE(torch::allclose(seq.index({0, 2}), torch::tensor({9.0f, 9.0f})));
}

TEST(HistoryBuffer, ResetClearsBuffer) {
    HistoryBuffer hb(4, 5);
    std::vector<float> obs(4, 1.0f);
    hb.add(obs.data());
    hb.reset();
    auto seq = hb.get_sequence();
    EXPECT_TRUE((seq == 0).all().item<bool>());
}

TEST(HistoryBuffer, AddCopiesData) {
    HistoryBuffer hb(3, 5);
    std::vector<float> obs = {1.0f, 2.0f, 3.0f};
    hb.add(obs.data());
    obs[0] = 0.0f;  // mutate original
    auto seq = hb.get_sequence();
    EXPECT_FLOAT_EQ(seq.index({0, -1, 0}).item<float>(), 1.0f);
}

// ── DisturbanceDetectorTrainer ──────────────────────────────────────

class TrainerTest : public ::testing::Test {
protected:
    TransformerDisturbanceDetector det{nullptr};
    std::unique_ptr<DisturbanceDetectorTrainer> trainer;

    void SetUp() override {
        det = TransformerDisturbanceDetector(4, 8, 16, 2, 1, 32, 0.0);
        trainer = std::make_unique<DisturbanceDetectorTrainer>(det, 1e-3);
    }
};

TEST_F(TrainerTest, TrainStepReturnsLosses) {
    auto obs = torch::randn({8, 8, 4});
    auto dist = torch::randint(0, 2, {8}).to(torch::kFloat32);
    auto alpha = torch::rand({8});
    auto result = trainer->train_step(obs, dist, alpha);

    EXPECT_TRUE(result.count("loss_disturbance") > 0);
    EXPECT_TRUE(result.count("loss_blending") > 0);
    EXPECT_TRUE(result.count("total_loss") > 0);
    for (auto& [k, v] : result) {
        EXPECT_GE(v, 0.0f);
    }
}

TEST_F(TrainerTest, TrainStepReducesLoss) {
    auto obs = torch::randn({16, 8, 4});
    auto dist = torch::ones({16});
    auto alpha = torch::ones({16}) * 0.8f;

    float loss_before = trainer->evaluate(obs, dist, alpha)["total_loss"];
    for (int i = 0; i < 50; ++i)
        trainer->train_step(obs, dist, alpha);
    float loss_after = trainer->evaluate(obs, dist, alpha)["total_loss"];
    EXPECT_LT(loss_after, loss_before);
}

TEST_F(TrainerTest, EvaluateNoGrad) {
    auto obs = torch::randn({4, 8, 4});
    auto dist = torch::zeros({4});
    auto alpha = torch::rand({4});
    auto result = trainer->evaluate(obs, dist, alpha);

    for (auto& p : det->parameters()) {
        EXPECT_TRUE(!p.grad().defined() || (p.grad() == 0).all().item<bool>());
    }
}

// ── AdaptiveControllerBlender ───────────────────────────────────────

class BlenderTest : public ::testing::Test {
protected:
    TransformerDisturbanceDetector det{nullptr};
    std::unique_ptr<HistoryBuffer> hb;
    std::unique_ptr<AdaptiveControllerBlender> blender;

    void SetUp() override {
        det = TransformerDisturbanceDetector(4, 5, 16, 2, 1, 32, 0.0);
        hb = std::make_unique<HistoryBuffer>(4, 5);
        blender = std::make_unique<AdaptiveControllerBlender>(det, *hb, 3);
    }
};

TEST_F(BlenderTest, ReturnsTupleOfThree) {
    std::vector<float> obs = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> a_opt = {1.0f, 1.0f, 1.0f};
    std::vector<float> a_rob = {0.0f, 0.0f, 0.0f};

    auto [blended, alpha_val, dist_val] = blender->get_blended_action(
        obs.data(), a_opt.data(), a_rob.data(), 3);
    EXPECT_EQ(blended.size(), 3u);
}

TEST_F(BlenderTest, AlphaInRange) {
    std::vector<float> obs = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> a_opt = {1.0f, 1.0f, 1.0f};
    std::vector<float> a_rob = {0.0f, 0.0f, 0.0f};

    auto [blended, alpha_val, dist_val] = blender->get_blended_action(
        obs.data(), a_opt.data(), a_rob.data(), 3);
    EXPECT_GE(alpha_val, 0.0f);
    EXPECT_LE(alpha_val, 1.0f);
    EXPECT_GE(dist_val, 0.0f);
    EXPECT_LE(dist_val, 1.0f);
}

TEST_F(BlenderTest, BlendingFormula) {
    std::vector<float> obs = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> a_opt = {10.0f, 20.0f, 30.0f};
    std::vector<float> a_rob = {1.0f, 2.0f, 3.0f};

    auto [blended, alpha, dist_val] = blender->get_blended_action(
        obs.data(), a_opt.data(), a_rob.data(), 3, false);

    for (size_t i = 0; i < 3; ++i) {
        float expected = alpha * a_opt[i] + (1.0f - alpha) * a_rob[i];
        EXPECT_NEAR(blended[i], expected, 1e-5f);
    }
}

TEST_F(BlenderTest, SmoothingProducesValues) {
    std::vector<float> a_opt = {1.0f, 1.0f, 1.0f};
    std::vector<float> a_rob = {0.0f, 0.0f, 0.0f};
    std::vector<float> alphas;

    for (int i = 0; i < 5; ++i) {
        std::vector<float> obs = {static_cast<float>(i) * 0.1f, 0.2f, 0.3f, 0.4f};
        auto [blended, alpha, dist] = blender->get_blended_action(
            obs.data(), a_opt.data(), a_rob.data(), 3, true);
        alphas.push_back(alpha);
    }
    EXPECT_EQ(alphas.size(), 5u);
}

TEST_F(BlenderTest, ResetClearsState) {
    std::vector<float> obs = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> a_opt = {1.0f, 1.0f, 1.0f};
    std::vector<float> a_rob = {0.0f, 0.0f, 0.0f};

    blender->get_blended_action(obs.data(), a_opt.data(), a_rob.data(), 3);
    blender->reset();

    auto seq = hb->get_sequence();
    EXPECT_TRUE((seq == 0).all().item<bool>());
}

TEST_F(BlenderTest, NoSmoothing) {
    std::vector<float> obs = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> a_opt = {10.0f, 10.0f, 10.0f};
    std::vector<float> a_rob = {0.0f, 0.0f, 0.0f};

    auto [blended, alpha, dist] = blender->get_blended_action(
        obs.data(), a_opt.data(), a_rob.data(), 3, false);

    for (size_t i = 0; i < 3; ++i) {
        float expected = alpha * a_opt[i] + (1.0f - alpha) * a_rob[i];
        EXPECT_NEAR(blended[i], expected, 1e-5f);
    }
}
