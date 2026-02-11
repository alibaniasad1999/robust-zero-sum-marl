#include "rzsm_control/robust_controller_node.hpp"

#include <cstring>
#include <numeric>

namespace rzsm_control {

RobustControllerNode::RobustControllerNode(const rclcpp::NodeOptions& options)
    : Node("robust_controller_node", options)
{
    // ── Declare parameters ───────────────────────────────────────
    declare_parameter("pi_opt_path", std::string(""));
    declare_parameter("pi_rob_path", std::string(""));
    declare_parameter("detector_path", std::string(""));
    declare_parameter("obs_dim", 27);
    declare_parameter("act_dim", 8);
    declare_parameter("sequence_length", 20);
    declare_parameter("hidden_sizes", std::vector<int64_t>{256, 256});
    declare_parameter("act_limit", 1.0);
    declare_parameter("d_model", 128);
    declare_parameter("nhead", 4);
    declare_parameter("num_layers", 3);
    declare_parameter("dim_feedforward", 256);
    declare_parameter("control_frequency", 200.0);
    declare_parameter("smoothing_window", 5);
    declare_parameter("use_smoothing", true);
    declare_parameter("device", std::string("auto"));

    // ── Read parameters ──────────────────────────────────────────
    pi_opt_path_     = get_parameter("pi_opt_path").as_string();
    pi_rob_path_     = get_parameter("pi_rob_path").as_string();
    detector_path_   = get_parameter("detector_path").as_string();
    obs_dim_         = get_parameter("obs_dim").as_int();
    act_dim_         = get_parameter("act_dim").as_int();
    sequence_length_ = get_parameter("sequence_length").as_int();
    hidden_sizes_    = get_parameter("hidden_sizes").as_integer_array();
    act_limit_       = get_parameter("act_limit").as_double();
    d_model_         = get_parameter("d_model").as_int();
    nhead_           = get_parameter("nhead").as_int();
    num_layers_      = get_parameter("num_layers").as_int();
    dim_feedforward_ = get_parameter("dim_feedforward").as_int();
    control_frequency_ = get_parameter("control_frequency").as_double();
    smoothing_window_  = static_cast<int>(get_parameter("smoothing_window").as_int());
    use_smoothing_     = get_parameter("use_smoothing").as_bool();

    auto device_str = get_parameter("device").as_string();
    device_ = rzsm::select_device(device_str);

    RCLCPP_INFO(get_logger(), "Device: %s",
                device_.is_cuda() ? "CUDA" : "CPU");

    // ── Load models and allocate buffers ─────────────────────────
    load_models();
    preallocate_buffers();

    // ── Subscriptions (SensorDataQoS = best_effort for low latency) ─
    auto sensor_qos = rclcpp::SensorDataQoS();

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", sensor_qos,
        std::bind(&RobustControllerNode::joint_state_callback, this,
                  std::placeholders::_1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "imu", sensor_qos,
        std::bind(&RobustControllerNode::imu_callback, this,
                  std::placeholders::_1));

    // ── Publishers ───────────────────────────────────────────────
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
        "cmd_vel", rclcpp::QoS(1).best_effort());

    detector_status_pub_ = create_publisher<rzsm_control::msg::DetectorStatus>(
        "detector_status", rclcpp::QoS(10));

    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/diagnostics", rclcpp::QoS(10));

    // ── Control timer ────────────────────────────────────────────
    auto period = std::chrono::duration<double>(1.0 / control_frequency_);
    control_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&RobustControllerNode::control_timer_callback, this));

    RCLCPP_INFO(get_logger(),
        "Robust controller started: obs_dim=%ld, act_dim=%ld, freq=%.0f Hz",
        obs_dim_, act_dim_, control_frequency_);
}

// ── Model loading ────────────────────────────────────────────────

void RobustControllerNode::load_models() {
    // Construct networks with matching architecture
    pi_opt_ = rzsm::MLPActor(obs_dim_, act_dim_, hidden_sizes_,
                              rzsm::ActivationType::Tanh, act_limit_);
    pi_rob_ = rzsm::MLPActor(obs_dim_, act_dim_, hidden_sizes_,
                              rzsm::ActivationType::Tanh, act_limit_);
    detector_ = rzsm::TransformerDisturbanceDetector(
        obs_dim_, sequence_length_, d_model_, nhead_, num_layers_,
        dim_feedforward_, /*dropout=*/0.0);

    // Load saved weights
    if (!pi_opt_path_.empty()) {
        torch::load(pi_opt_, pi_opt_path_);
        RCLCPP_INFO(get_logger(), "Loaded pi_opt from: %s (%ld params)",
                    pi_opt_path_.c_str(), rzsm::count_vars(*pi_opt_));
    } else {
        RCLCPP_WARN(get_logger(), "pi_opt_path is empty — using random weights");
    }

    if (!pi_rob_path_.empty()) {
        torch::load(pi_rob_, pi_rob_path_);
        RCLCPP_INFO(get_logger(), "Loaded pi_rob from: %s (%ld params)",
                    pi_rob_path_.c_str(), rzsm::count_vars(*pi_rob_));
    } else {
        RCLCPP_WARN(get_logger(), "pi_rob_path is empty — using random weights");
    }

    if (!detector_path_.empty()) {
        torch::load(detector_, detector_path_);
        RCLCPP_INFO(get_logger(), "Loaded detector from: %s (%ld params)",
                    detector_path_.c_str(), rzsm::count_vars(*detector_));
    } else {
        RCLCPP_WARN(get_logger(), "detector_path is empty — using random weights");
    }

    // Move to device, set eval mode, freeze
    pi_opt_->to(device_);
    pi_rob_->to(device_);
    detector_->to(device_);

    pi_opt_->eval();
    pi_rob_->eval();
    detector_->eval();

    rzsm::freeze(*pi_opt_);
    rzsm::freeze(*pi_rob_);
    rzsm::freeze(*detector_);
}

void RobustControllerNode::preallocate_buffers() {
    history_buffer_ = std::make_unique<rzsm::HistoryBuffer>(
        obs_dim_, sequence_length_, device_);

    blender_ = std::make_unique<rzsm::AdaptiveControllerBlender>(
        detector_, *history_buffer_, smoothing_window_);

    obs_tensor_ = torch::zeros(
        {1, obs_dim_},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
}

// ── Sensor callbacks ─────────────────────────────────────────────

void RobustControllerNode::joint_state_callback(
    const sensor_msgs::msg::JointState::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    latest_joint_state_ = msg;
    joint_state_received_ = true;
}

void RobustControllerNode::imu_callback(
    const sensor_msgs::msg::Imu::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    latest_imu_ = msg;
    imu_received_ = true;
}

// ── Control loop ─────────────────────────────────────────────────

void RobustControllerNode::control_timer_callback() {
    if (!joint_state_received_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "Waiting for /joint_states...");
        return;
    }
    run_inference();
}

void RobustControllerNode::run_inference() {
    auto t0 = std::chrono::steady_clock::now();

    // 1. Build observation from sensor data
    auto obs = build_observation();

    // 2. Run both policies + detector (all in no-grad mode)
    std::vector<float> a_opt(act_dim_);
    std::vector<float> a_rob(act_dim_);
    {
        torch::NoGradGuard no_grad;

        // Copy obs into pre-allocated tensor (zero-copy reuse)
        auto obs_ptr = obs_tensor_.data_ptr<float>();
        std::memcpy(obs_ptr, obs.data(), obs_dim_ * sizeof(float));

        // Forward through both policies
        auto a_opt_t = pi_opt_->forward(obs_tensor_).to(torch::kCPU).contiguous();
        auto a_rob_t = pi_rob_->forward(obs_tensor_).to(torch::kCPU).contiguous();

        std::memcpy(a_opt.data(), a_opt_t.data_ptr<float>(),
                     act_dim_ * sizeof(float));
        std::memcpy(a_rob.data(), a_rob_t.data_ptr<float>(),
                     act_dim_ * sizeof(float));
    }

    // 3. Blend via transformer detector
    auto [blended, alpha, dist_prob] = blender_->get_blended_action(
        obs.data(), a_opt.data(), a_rob.data(), act_dim_, use_smoothing_);

    auto t1 = std::chrono::steady_clock::now();
    float latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 4. Publish everything
    publish_command(blended);
    publish_detector_status(dist_prob, alpha, latency_ms, a_opt, a_rob, blended);
    publish_diagnostics(latency_ms, dist_prob);
}

std::vector<float> RobustControllerNode::build_observation() const {
    std::vector<float> obs(obs_dim_, 0.0f);

    std::lock_guard<std::mutex> lock(sensor_mutex_);

    int idx = 0;

    // Joint positions
    if (latest_joint_state_) {
        for (size_t i = 0; i < latest_joint_state_->position.size() &&
                           idx < obs_dim_; ++i) {
            obs[idx++] = static_cast<float>(latest_joint_state_->position[i]);
        }
        // Joint velocities
        for (size_t i = 0; i < latest_joint_state_->velocity.size() &&
                           idx < obs_dim_; ++i) {
            obs[idx++] = static_cast<float>(latest_joint_state_->velocity[i]);
        }
    }

    // IMU data (if available and we still have space)
    if (latest_imu_ && idx < obs_dim_) {
        obs[idx++] = static_cast<float>(latest_imu_->linear_acceleration.x);
        if (idx < obs_dim_)
            obs[idx++] = static_cast<float>(latest_imu_->linear_acceleration.y);
        if (idx < obs_dim_)
            obs[idx++] = static_cast<float>(latest_imu_->linear_acceleration.z);
        if (idx < obs_dim_)
            obs[idx++] = static_cast<float>(latest_imu_->angular_velocity.x);
        if (idx < obs_dim_)
            obs[idx++] = static_cast<float>(latest_imu_->angular_velocity.y);
        if (idx < obs_dim_)
            obs[idx++] = static_cast<float>(latest_imu_->angular_velocity.z);
    }

    return obs;
}

// ── Publishing ───────────────────────────────────────────────────

void RobustControllerNode::publish_command(const std::vector<float>& action) {
    auto msg = geometry_msgs::msg::TwistStamped();
    msg.header.stamp = now();
    msg.header.frame_id = "base_link";

    // Map action vector to Twist fields.
    // Adjust this mapping for your specific robot.
    if (act_dim_ >= 1) msg.twist.linear.x  = action[0];
    if (act_dim_ >= 2) msg.twist.linear.y  = action[1];
    if (act_dim_ >= 3) msg.twist.linear.z  = action[2];
    if (act_dim_ >= 4) msg.twist.angular.x = action[3];
    if (act_dim_ >= 5) msg.twist.angular.y = action[4];
    if (act_dim_ >= 6) msg.twist.angular.z = action[5];

    cmd_vel_pub_->publish(msg);
}

void RobustControllerNode::publish_detector_status(
    float dist_prob, float alpha, float latency_ms,
    const std::vector<float>& a_opt,
    const std::vector<float>& a_rob,
    const std::vector<float>& a_blend)
{
    auto msg = rzsm_control::msg::DetectorStatus();
    msg.stamp = now();
    msg.disturbance_probability = dist_prob;
    msg.blending_weight = alpha;
    msg.latency_ms = latency_ms;
    msg.action_optimal.assign(a_opt.begin(), a_opt.end());
    msg.action_robust.assign(a_rob.begin(), a_rob.end());
    msg.action_blended.assign(a_blend.begin(), a_blend.end());
    msg.obs_dim = static_cast<uint32_t>(obs_dim_);
    msg.act_dim = static_cast<uint32_t>(act_dim_);
    msg.sequence_length = static_cast<uint32_t>(sequence_length_);

    detector_status_pub_->publish(msg);
}

void RobustControllerNode::publish_diagnostics(
    float latency_ms, float dist_prob)
{
    auto msg = diagnostic_msgs::msg::DiagnosticArray();
    msg.header.stamp = now();

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "rzsm_controller";
    status.hardware_id = "jetson";

    float budget_ms = static_cast<float>(1000.0 / control_frequency_);
    if (latency_ms < budget_ms * 0.8f) {
        status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        status.message = "Running within time budget";
    } else if (latency_ms < budget_ms) {
        status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        status.message = "Approaching time budget limit";
    } else {
        status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
        status.message = "Exceeded time budget";
    }

    auto kv = [](const std::string& key, const std::string& value) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = key;
        kv.value = value;
        return kv;
    };

    status.values.push_back(kv("latency_ms", std::to_string(latency_ms)));
    status.values.push_back(kv("budget_ms", std::to_string(budget_ms)));
    status.values.push_back(kv("disturbance_prob",
                               std::to_string(dist_prob)));
    status.values.push_back(kv("device",
                               device_.is_cuda() ? "cuda" : "cpu"));
    status.values.push_back(kv("frequency",
                               std::to_string(control_frequency_)));

    msg.status.push_back(status);
    diagnostics_pub_->publish(msg);
}

}  // namespace rzsm_control
