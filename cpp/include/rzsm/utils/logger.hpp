#pragma once
#include <fstream>
#include <string>
#include <vector>

namespace rzsm {

struct TransitionRecord {
    int t;
    int episode_id;
    std::vector<float> obs;
    std::vector<float> action_ctrl;
    std::vector<float> action_dist;
    std::vector<float> action_total;
    float reward;
    bool terminated;
    bool truncated;
    std::vector<float> disturbance_params;
    std::string disturbance_tag = "none";
    float robust_weight_target = 0.0f;
};

class DatasetLogger {
public:
    explicit DatasetLogger(const std::string& log_dir);

    void log(const TransitionRecord& record);
    void end_episode();

    const std::string& log_dir() const { return log_dir_; }

private:
    std::string log_dir_;
    std::vector<TransitionRecord> episode_buffer_;
    int episode_id_ = 0;
    bool csv_header_written_ = false;
    std::string csv_path_;
};

} // namespace rzsm
