#include <rzsm/utils/logger.hpp>
#include <cnpy.h>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;

namespace rzsm {

DatasetLogger::DatasetLogger(const std::string& log_dir)
    : log_dir_(log_dir)
{
    fs::create_directories(log_dir);
    csv_path_ = log_dir + "/transitions.csv";
    csv_header_written_ = fs::exists(csv_path_);
}

void DatasetLogger::log(const TransitionRecord& record) {
    episode_buffer_.push_back(record);
}

void DatasetLogger::end_episode() {
    if (episode_buffer_.empty()) return;

    size_t n = episode_buffer_.size();
    int64_t obs_dim = static_cast<int64_t>(episode_buffer_[0].obs.size());
    int64_t act_dim = static_cast<int64_t>(episode_buffer_[0].action_ctrl.size());

    // Flatten data for cnpy
    std::vector<float> obs_flat(n * obs_dim);
    std::vector<float> act_ctrl_flat(n * act_dim);
    std::vector<float> act_dist_flat(n * act_dim);
    std::vector<float> act_total_flat(n * act_dim);
    std::vector<float> rewards(n);
    std::vector<uint8_t> terminated(n);
    std::vector<uint8_t> truncated(n);
    std::vector<float> alpha_targets(n);

    for (size_t i = 0; i < n; ++i) {
        auto& r = episode_buffer_[i];
        std::copy(r.obs.begin(), r.obs.end(), &obs_flat[i * obs_dim]);
        std::copy(r.action_ctrl.begin(), r.action_ctrl.end(), &act_ctrl_flat[i * act_dim]);
        std::copy(r.action_dist.begin(), r.action_dist.end(), &act_dist_flat[i * act_dim]);
        std::copy(r.action_total.begin(), r.action_total.end(), &act_total_flat[i * act_dim]);
        rewards[i] = r.reward;
        terminated[i] = r.terminated ? 1 : 0;
        truncated[i] = r.truncated ? 1 : 0;
        alpha_targets[i] = r.robust_weight_target;
    }

    // Save .npz
    std::string episode_dir = log_dir_ + "/episodes";
    fs::create_directories(episode_dir);

    std::ostringstream fname;
    fname << episode_dir << "/ep_" << std::setw(6) << std::setfill('0')
          << episode_id_ << ".npz";

    cnpy::npz_save(fname.str(), "obs", obs_flat.data(),
                    {n, static_cast<size_t>(obs_dim)}, "w");
    cnpy::npz_save(fname.str(), "action_ctrl", act_ctrl_flat.data(),
                    {n, static_cast<size_t>(act_dim)}, "a");
    cnpy::npz_save(fname.str(), "action_dist", act_dist_flat.data(),
                    {n, static_cast<size_t>(act_dim)}, "a");
    cnpy::npz_save(fname.str(), "action_total", act_total_flat.data(),
                    {n, static_cast<size_t>(act_dim)}, "a");
    cnpy::npz_save(fname.str(), "reward", rewards.data(), {n}, "a");
    cnpy::npz_save(fname.str(), "terminated", terminated.data(), {n}, "a");
    cnpy::npz_save(fname.str(), "truncated", truncated.data(), {n}, "a");
    cnpy::npz_save(fname.str(), "robust_weight_target", alpha_targets.data(),
                    {n}, "a");

    // Append CSV summary
    {
        std::ofstream csv(csv_path_, std::ios::app);
        if (!csv_header_written_) {
            csv << "episode_id,length,total_reward,has_disturbance,mean_alpha_target\n";
            csv_header_written_ = true;
        }
        float total_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0f);
        bool has_dist = false;
        for (auto& r : episode_buffer_) {
            if (r.disturbance_tag != "none") { has_dist = true; break; }
        }
        float mean_alpha = std::accumulate(alpha_targets.begin(),
                                            alpha_targets.end(), 0.0f) / n;
        csv << episode_id_ << "," << n << "," << total_reward << ","
            << (has_dist ? 1 : 0) << "," << mean_alpha << "\n";
    }

    episode_id_++;
    episode_buffer_.clear();
}

} // namespace rzsm
