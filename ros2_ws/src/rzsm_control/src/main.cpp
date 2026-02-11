#include <rclcpp/rclcpp.hpp>
#include "rzsm_control/robust_controller_node.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rzsm_control::RobustControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
