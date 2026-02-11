#include <rzsm/envs/mujoco_env.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef RZSM_ASSETS_DIR
#define RZSM_ASSETS_DIR "assets"
#endif

namespace rzsm {

MuJoCoEnv::MuJoCoEnv(const std::string& env_name,
                       const std::string& xml_path,
                       int max_episode_steps)
    : env_name_(env_name), max_episode_steps_(max_episode_steps)
{
    // Set per-environment constants
    if (env_name == "Ant") {
        frameskip_ = 5;
        ctrl_cost_weight_ = 0.5f;
        healthy_z_min_ = 0.2f;
        healthy_z_max_ = 1.0f;
    } else if (env_name == "Humanoid") {
        frameskip_ = 5;
        ctrl_cost_weight_ = 0.1f;
        healthy_z_min_ = 1.0f;
        healthy_z_max_ = 2.0f;
    } else {
        throw std::runtime_error("Unknown environment: " + env_name);
    }

    std::string path = xml_path;
    if (path.empty()) {
        std::string lower_name = env_name;
        std::transform(lower_name.begin(), lower_name.end(),
                       lower_name.begin(), ::tolower);
        path = std::string(RZSM_ASSETS_DIR) + "/" + lower_name + ".xml";
    }

    char error[1000] = "";
    model_ = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
    if (!model_) {
        throw std::runtime_error("Failed to load MuJoCo model '" + path +
                                 "': " + std::string(error));
    }
    data_ = mj_makeData(model_);
}

MuJoCoEnv::~MuJoCoEnv() {
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

MuJoCoEnv::MuJoCoEnv(MuJoCoEnv&& o) noexcept
    : env_name_(std::move(o.env_name_)), model_(o.model_), data_(o.data_),
      max_episode_steps_(o.max_episode_steps_),
      current_step_(o.current_step_), rng_(std::move(o.rng_)),
      frameskip_(o.frameskip_), ctrl_cost_weight_(o.ctrl_cost_weight_),
      healthy_z_min_(o.healthy_z_min_), healthy_z_max_(o.healthy_z_max_),
      prev_x_(o.prev_x_)
{
    o.model_ = nullptr;
    o.data_ = nullptr;
}

MuJoCoEnv& MuJoCoEnv::operator=(MuJoCoEnv&& o) noexcept {
    if (this != &o) {
        if (data_) mj_deleteData(data_);
        if (model_) mj_deleteModel(model_);
        env_name_ = std::move(o.env_name_);
        model_ = o.model_; data_ = o.data_;
        max_episode_steps_ = o.max_episode_steps_;
        current_step_ = o.current_step_;
        rng_ = std::move(o.rng_);
        frameskip_ = o.frameskip_;
        ctrl_cost_weight_ = o.ctrl_cost_weight_;
        healthy_z_min_ = o.healthy_z_min_;
        healthy_z_max_ = o.healthy_z_max_;
        prev_x_ = o.prev_x_;
        o.model_ = nullptr; o.data_ = nullptr;
    }
    return *this;
}

std::vector<float> MuJoCoEnv::reset(int seed) {
    if (seed >= 0) rng_.seed(seed);

    mj_resetData(model_, data_);

    // Apply small noise to initial state (matching Gymnasium)
    std::uniform_real_distribution<double> qpos_noise(-0.1, 0.1);
    std::uniform_real_distribution<double> qvel_noise(-0.01, 0.01);

    for (int i = 0; i < model_->nq; ++i)
        data_->qpos[i] = model_->qpos0[i] + qpos_noise(rng_);
    for (int i = 0; i < model_->nv; ++i)
        data_->qvel[i] = qvel_noise(rng_);

    mj_forward(model_, data_);
    current_step_ = 0;
    prev_x_ = data_->qpos[0];

    std::vector<float> obs;
    compute_obs(obs);
    return obs;
}

StepResult MuJoCoEnv::step(const float* action) {
    prev_x_ = data_->qpos[0];

    // Set controls
    for (int i = 0; i < model_->nu; ++i)
        data_->ctrl[i] = static_cast<double>(action[i]);

    // Step with frameskip
    for (int i = 0; i < frameskip_; ++i)
        mj_step(model_, data_);

    current_step_++;

    StepResult result;
    compute_obs(result.obs);
    result.reward = compute_reward(action);
    result.terminated = check_terminated();
    result.truncated = (current_step_ >= max_episode_steps_);
    return result;
}

void MuJoCoEnv::compute_obs(std::vector<float>& out) {
    out.clear();
    // qpos[2:] — skip x,y of torso
    for (int i = 2; i < model_->nq; ++i)
        out.push_back(static_cast<float>(data_->qpos[i]));
    // qvel[:]
    for (int i = 0; i < model_->nv; ++i)
        out.push_back(static_cast<float>(data_->qvel[i]));
}

float MuJoCoEnv::compute_reward(const float* action) {
    double dt = model_->opt.timestep * frameskip_;
    double x_after = data_->qpos[0];
    double forward_reward = (x_after - prev_x_) / dt;

    double ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i)
        ctrl_cost += action[i] * action[i];
    ctrl_cost *= ctrl_cost_weight_;

    bool healthy = !check_terminated();
    double healthy_reward = healthy ? 1.0 : 0.0;

    return static_cast<float>(forward_reward - ctrl_cost + healthy_reward);
}

bool MuJoCoEnv::check_terminated() {
    double z = data_->qpos[2];
    if (z < healthy_z_min_ || z > healthy_z_max_) return true;

    for (int i = 0; i < model_->nq; ++i)
        if (!std::isfinite(data_->qpos[i])) return true;
    for (int i = 0; i < model_->nv; ++i)
        if (!std::isfinite(data_->qvel[i])) return true;

    return false;
}

std::vector<float> MuJoCoEnv::sample_action() {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> action(model_->nu);
    for (int i = 0; i < model_->nu; ++i)
        action[i] = dist(rng_);
    return action;
}

SpaceInfo MuJoCoEnv::observation_space() const {
    int64_t obs_dim = (model_->nq - 2) + model_->nv;
    return {obs_dim, -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
}

SpaceInfo MuJoCoEnv::action_space() const {
    return {model_->nu, -1.0f, 1.0f};
}

} // namespace rzsm
