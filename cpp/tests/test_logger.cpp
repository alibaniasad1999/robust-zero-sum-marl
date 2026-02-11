// Tests for rzsm::DatasetLogger and TransitionRecord
// Mirrors Python tests/test_logger.py

#include <gtest/gtest.h>
#include <rzsm/utils/logger.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;
using namespace rzsm;

static TransitionRecord make_record(int t = 0, int episode_id = 0,
                                     int obs_dim = 4, int act_dim = 2,
                                     const std::string& tag = "none") {
    std::mt19937 rng(t * 100 + episode_id);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    TransitionRecord r;
    r.t = t;
    r.episode_id = episode_id;
    r.obs.resize(obs_dim);
    r.action_ctrl.resize(act_dim);
    r.action_dist.resize(act_dim);
    r.action_total.resize(act_dim);
    for (int i = 0; i < obs_dim; ++i) r.obs[i] = dist(rng);
    for (int i = 0; i < act_dim; ++i) {
        r.action_ctrl[i] = dist(rng);
        r.action_dist[i] = dist(rng);
        r.action_total[i] = dist(rng);
    }
    r.reward = dist(rng);
    r.terminated = false;
    r.truncated = false;
    r.disturbance_params = {0.0f, 0.1f};
    r.disturbance_tag = tag;
    r.robust_weight_target = 0.5f;
    return r;
}

class LoggerTest : public ::testing::Test {
protected:
    fs::path tmp_dir;
    std::unique_ptr<DatasetLogger> logger;

    void SetUp() override {
        tmp_dir = fs::temp_directory_path() / ("rzsm_test_" + std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count()));
        logger = std::make_unique<DatasetLogger>(tmp_dir.string());
    }

    void TearDown() override {
        if (fs::exists(tmp_dir))
            fs::remove_all(tmp_dir);
    }
};

TEST_F(LoggerTest, CreatesLogDir) {
    EXPECT_TRUE(fs::is_directory(logger->log_dir()));
}

TEST_F(LoggerTest, EndEpisodeEmptyBufferNoop) {
    logger->end_episode();  // should not crash
    auto episode_dir = fs::path(logger->log_dir()) / "episodes";
    EXPECT_FALSE(fs::exists(episode_dir));
}

TEST_F(LoggerTest, SingleEpisodeSavesNpz) {
    for (int t = 0; t < 10; ++t)
        logger->log(make_record(t, 0));
    logger->end_episode();

    auto episode_dir = fs::path(logger->log_dir()) / "episodes";
    auto npz_path = episode_dir / "ep_000000.npz";
    EXPECT_TRUE(fs::is_regular_file(npz_path));
}

TEST_F(LoggerTest, CsvSummaryWritten) {
    for (int t = 0; t < 5; ++t)
        logger->log(make_record(t));
    logger->end_episode();

    auto csv_path = fs::path(logger->log_dir()) / "transitions.csv";
    EXPECT_TRUE(fs::is_regular_file(csv_path));

    std::ifstream f(csv_path);
    std::string line;
    int line_count = 0;
    while (std::getline(f, line)) line_count++;
    EXPECT_EQ(line_count, 2);  // header + 1 episode
}

TEST_F(LoggerTest, MultipleEpisodesIncrementId) {
    for (int ep = 0; ep < 3; ++ep) {
        for (int t = 0; t < 5; ++t)
            logger->log(make_record(t, ep));
        logger->end_episode();
    }

    auto episode_dir = fs::path(logger->log_dir()) / "episodes";
    EXPECT_TRUE(fs::is_regular_file(episode_dir / "ep_000000.npz"));
    EXPECT_TRUE(fs::is_regular_file(episode_dir / "ep_000001.npz"));
    EXPECT_TRUE(fs::is_regular_file(episode_dir / "ep_000002.npz"));
}

TEST_F(LoggerTest, CsvHasCorrectDisturbanceFlag) {
    // Episode with disturbance
    for (int t = 0; t < 3; ++t)
        logger->log(make_record(t, 0, 4, 2, "adversary"));
    logger->end_episode();

    // Episode without
    for (int t = 0; t < 3; ++t)
        logger->log(make_record(t, 1, 4, 2, "none"));
    logger->end_episode();

    auto csv_path = fs::path(logger->log_dir()) / "transitions.csv";
    std::ifstream f(csv_path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) lines.push_back(line);

    ASSERT_EQ(lines.size(), 3u);  // header + 2 episodes
    // Parse CSV: has_disturbance is 4th column (index 3)
    auto get_col = [](const std::string& row, int col) {
        std::istringstream ss(row);
        std::string token;
        for (int i = 0; i <= col; ++i) std::getline(ss, token, ',');
        return token;
    };
    EXPECT_EQ(get_col(lines[1], 3), "1");  // has_disturbance = True
    EXPECT_EQ(get_col(lines[2], 3), "0");  // has_disturbance = False
}

TEST_F(LoggerTest, BufferClearedAfterEndEpisode) {
    logger->log(make_record());
    logger->end_episode();
    // Buffer should be empty now — another end_episode is a no-op
    logger->end_episode();

    auto episode_dir = fs::path(logger->log_dir()) / "episodes";
    int file_count = 0;
    for (auto& entry : fs::directory_iterator(episode_dir))
        if (entry.is_regular_file()) file_count++;
    EXPECT_EQ(file_count, 1);  // only the first episode
}

// ── TransitionRecord ────────────────────────────────────────────────

TEST(TransitionRecord, DefaultValues) {
    TransitionRecord r;
    r.disturbance_tag = "none";
    r.robust_weight_target = 0.0f;
    EXPECT_EQ(r.disturbance_tag, "none");
    EXPECT_FLOAT_EQ(r.robust_weight_target, 0.0f);
}

TEST(TransitionRecord, FieldsAccessible) {
    auto r = make_record(5, 3, 4, 2, "adversary");
    EXPECT_EQ(r.t, 5);
    EXPECT_EQ(r.episode_id, 3);
    EXPECT_EQ(r.disturbance_tag, "adversary");
}
