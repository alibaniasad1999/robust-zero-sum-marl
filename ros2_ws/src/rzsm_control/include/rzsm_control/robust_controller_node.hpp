#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>

// rzsm library (from cpp/)
#include <rzsm/common.hpp>
#include <rzsm/networks/mlp.hpp>
#include <rzsm/detector/transformer.hpp>
#include <rzsm/detector/history_buffer.hpp>
#include <rzsm/detector/blender.hpp>

// Generated message
#include "rzsm_control/msg/detector_status.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rzsm_control {

class RobustControllerNode : public rclcpp::Node {
public:
    explicit RobustControllerNode(
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~RobustControllerNode() override = default;

private:
    // ── Callbacks ────────────────────────────────────────────────
    void joint_state_callback(
        const sensor_msgs::msg::JointState::SharedPtr msg);
    void imu_callback(
        const sensor_msgs::msg::Imu::SharedPtr msg);
    void control_timer_callback();

    // ── Inference pipeline ───────────────────────────────────────
    void run_inference();
    std::vector<float> build_observation() const;
    void publish_command(const std::vector<float>& action);
    void publish_detector_status(
        float dist_prob, float alpha, float latency_ms,
        const std::vector<float>& a_opt,
        const std::vector<float>& a_rob,
        const std::vector<float>& a_blend);
    void publish_diagnostics(float latency_ms, float dist_prob);

    // ── Initialization ───────────────────────────────────────────
    void load_models();
    void preallocate_buffers();

    // ── Parameters ───────────────────────────────────────────────
    std::string pi_opt_path_;
    std::string pi_rob_path_;
    std::string detector_path_;

    int64_t obs_dim_;
    int64_t act_dim_;
    int64_t sequence_length_;

    std::vector<int64_t> hidden_sizes_;
    double act_limit_;

    int64_t d_model_;
    int64_t nhead_;
    int64_t num_layers_;
    int64_t dim_feedforward_;

    double control_frequency_;
    int smoothing_window_;
    bool use_smoothing_;

    // ── Torch components ─────────────────────────────────────────
    torch::Device device_{torch::kCPU};
    rzsm::MLPActor pi_opt_{nullptr};
    rzsm::MLPActor pi_rob_{nullptr};
    rzsm::TransformerDisturbanceDetector detector_{nullptr};
    std::unique_ptr<rzsm::HistoryBuffer> history_buffer_;
    std::unique_ptr<rzsm::AdaptiveControllerBlender> blender_;

    // Pre-allocated tensor (avoids per-cycle allocation)
    torch::Tensor obs_tensor_;

    // ── Sensor state (mutex-protected) ───────────────────────────
    mutable std::mutex sensor_mutex_;
    sensor_msgs::msg::JointState::SharedPtr latest_joint_state_;
    sensor_msgs::msg::Imu::SharedPtr latest_imu_;
    bool joint_state_received_{false};
    bool imu_received_{false};

    // ── ROS interfaces ───────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
        joint_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr
        imu_sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr
        cmd_vel_pub_;
    rclcpp::Publisher<rzsm_control::msg::DetectorStatus>::SharedPtr
        detector_status_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
        diagnostics_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
};

}  // namespace rzsm_control
