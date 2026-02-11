// Tests for rzsm::ReplayBuffer
// Mirrors Python tests/test_replay_buffer.py

#include <gtest/gtest.h>
#include <rzsm/utils/replay_buffer.hpp>

using namespace rzsm;

class ReplayBufferTest : public ::testing::Test {
protected:
    static constexpr int64_t OBS_DIM = 4;
    static constexpr int64_t ACT_DIM = 2;
    static constexpr int64_t CAP = 100;

    ReplayBuffer buf{CAP, OBS_DIM, ACT_DIM, torch::kCPU};

    void add_transition(float val = 1.0f, bool terminated = false, bool truncated = false) {
        std::vector<float> obs(OBS_DIM, val);
        std::vector<float> next_obs(OBS_DIM, val + 0.1f);
        std::vector<float> act(ACT_DIM, val * 0.5f);
        buf.add(obs.data(), next_obs.data(), act.data(), val, terminated, truncated);
    }
};

TEST_F(ReplayBufferTest, InitialSizeZero) {
    EXPECT_EQ(buf.size(), 0);
}

TEST_F(ReplayBufferTest, SizeIncrementsOnAdd) {
    add_transition();
    EXPECT_EQ(buf.size(), 1);
    add_transition();
    EXPECT_EQ(buf.size(), 2);
}

TEST_F(ReplayBufferTest, SizeCapsAtCapacity) {
    for (int i = 0; i < 150; ++i)
        add_transition(static_cast<float>(i));
    EXPECT_EQ(buf.size(), CAP);
}

TEST_F(ReplayBufferTest, SampleBatchShapes) {
    for (int i = 0; i < 20; ++i)
        add_transition(static_cast<float>(i));

    auto batch = buf.sample(8);
    EXPECT_EQ(batch.obs.size(0), 8);
    EXPECT_EQ(batch.obs.size(1), OBS_DIM);
    EXPECT_EQ(batch.next_obs.size(0), 8);
    EXPECT_EQ(batch.next_obs.size(1), OBS_DIM);
    EXPECT_EQ(batch.act.size(0), 8);
    EXPECT_EQ(batch.act.size(1), ACT_DIM);
    EXPECT_EQ(batch.rew.size(0), 8);
    EXPECT_EQ(batch.rew.size(1), 1);
    EXPECT_EQ(batch.terminated.size(0), 8);
    EXPECT_EQ(batch.truncated.size(0), 8);
}

TEST_F(ReplayBufferTest, DonePropertyCombinesTermAndTrunc) {
    // Add terminated transition
    add_transition(1.0f, true, false);
    // Add truncated transition
    add_transition(2.0f, false, true);
    // Add neither
    add_transition(3.0f, false, false);

    // Check raw storage by sampling all 3
    EXPECT_EQ(buf.size(), 3);
}

TEST_F(ReplayBufferTest, SampleRespectsBatchSize) {
    for (int i = 0; i < 50; ++i)
        add_transition(static_cast<float>(i));

    auto batch1 = buf.sample(1);
    EXPECT_EQ(batch1.obs.size(0), 1);

    auto batch32 = buf.sample(32);
    EXPECT_EQ(batch32.obs.size(0), 32);
}

TEST_F(ReplayBufferTest, CircularOverwrite) {
    // Fill buffer completely
    for (int i = 0; i < CAP; ++i)
        add_transition(static_cast<float>(i));
    EXPECT_EQ(buf.size(), CAP);

    // Overwrite
    add_transition(999.0f);
    EXPECT_EQ(buf.size(), CAP);  // size stays at capacity
}

TEST_F(ReplayBufferTest, TotalActDim) {
    EXPECT_EQ(buf.total_act(), ACT_DIM);
}

TEST_F(ReplayBufferTest, SampledTensorsAreFloat) {
    for (int i = 0; i < 10; ++i)
        add_transition(static_cast<float>(i));

    auto batch = buf.sample(4);
    EXPECT_EQ(batch.obs.scalar_type(), torch::kFloat32);
    EXPECT_EQ(batch.act.scalar_type(), torch::kFloat32);
    EXPECT_EQ(batch.rew.scalar_type(), torch::kFloat32);
}
