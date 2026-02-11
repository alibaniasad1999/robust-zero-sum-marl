#pragma once
#include <mujoco/mujoco.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rzsm {

struct StepResult {
    std::vector<float> obs;
    float reward;
    bool terminated;
    bool truncated;
};

struct SpaceInfo {
    int64_t dim;
    float low;
    float high;
};

class MuJoCoEnv {
public:
    // env_name: "Ant" or "Humanoid"
    // xml_path: explicit path, or "" to use bundled assets
    MuJoCoEnv(const std::string& env_name,
              const std::string& xml_path = "",
              int max_episode_steps = 1000);
    ~MuJoCoEnv();

    // Move-only (owns mjModel/mjData)
    MuJoCoEnv(const MuJoCoEnv&) = delete;
    MuJoCoEnv& operator=(const MuJoCoEnv&) = delete;
    MuJoCoEnv(MuJoCoEnv&& o) noexcept;
    MuJoCoEnv& operator=(MuJoCoEnv&& o) noexcept;

    std::vector<float> reset(int seed = -1);
    StepResult step(const float* action);
    std::vector<float> sample_action();

    SpaceInfo observation_space() const;
    SpaceInfo action_space() const;

    const std::string& env_name() const { return env_name_; }

private:
    void compute_obs(std::vector<float>& out);
    float compute_reward(const float* action);
    bool check_terminated();

    std::string env_name_;
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    int max_episode_steps_;
    int current_step_ = 0;
    std::mt19937 rng_;

    // Per-environment constants
    int frameskip_ = 5;
    float ctrl_cost_weight_ = 0.5f;
    float healthy_z_min_ = 0.2f;
    float healthy_z_max_ = 1.0f;

    double prev_x_ = 0.0;
};

} // namespace rzsm
