// ROS
#include <ros/ros.h>

// MoveIt!
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

// TF2
#include <eigen_conversions/eigen_msg.h>

#include <string>
#include "receive_direction.h"

std::string tf_prefix_ = "robot1/";

moveit::core::MoveItErrorCode moveToCartesianPose(moveit::planning_interface::MoveGroupInterface &group, geometry_msgs::Pose target_pose) {
    group.setStartStateToCurrentState();
    group.setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    auto error_code = group.plan(my_plan);
    bool success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);

    moveit_visual_tools::MoveItVisualTools visual_tools("world");
    visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

    ROS_INFO("Move planning (cartesian pose goal) %s", success ? "SUCCESS" : "FAILED");
    if (success) {
        error_code = group.execute(my_plan);
    }
    return error_code;
}

moveit::core::MoveItErrorCode moveToNamedPose(moveit::planning_interface::MoveGroupInterface &group, std::string named_pose) {
    robot_state::RobotState start_state(*group.getCurrentState());
    group.setStartState(start_state);
    group.setNamedTarget(named_pose);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    auto error_code = group.plan(my_plan);
    bool success = (error_code == moveit::core::MoveItErrorCode::SUCCESS);

    ROS_INFO("Move planning (named pose goal) %s", success ? "SUCCESS" : "FAILED");
    if (success) {
        error_code = group.execute(my_plan);
    }
    return error_code;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "dual_arm_test");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::NodeHandle nh;

    ROS_INFO_STREAM("Setting up MoveIt.");

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    geometry_msgs::Pose target_pose_left;
    target_pose_left.position.x = 0.92525;
    target_pose_left.position.y = 0.30384;
    target_pose_left.position.z = 1.2803;
    target_pose_left.orientation.x = 0.0229577;
    target_pose_left.orientation.y = 0.00179;
    target_pose_left.orientation.z = -0.00010;
    target_pose_left.orientation.w = 0.999735;
    geometry_msgs::Pose target_pose_right;
    target_pose_right.position.x = 0.82806;
    target_pose_right.position.y = -0.19653;
    target_pose_right.position.z = 1.2903;
    target_pose_right.orientation.x = -0.00817;
    target_pose_right.orientation.y = 0.00160;
    target_pose_right.orientation.z = 0.0;
    target_pose_right.orientation.w = 0.9999965;
    geometry_msgs::Pose target_pose_final;

    ROS_INFO_STREAM("Getting direction");
    int direction = receive_direction();

    std::string arm_group_name;
    std::string gripper_group_name;
    if (direction == 0) { // left
        arm_group_name = "arm1";
        gripper_group_name = "gripper1";
        target_pose_final = target_pose_left;
    } else {
        arm_group_name = "arm2";
        gripper_group_name = "gripper2";
        target_pose_final = target_pose_right;
    }

    moveit::planning_interface::MoveGroupInterface arm_group(arm_group_name);
    moveit::planning_interface::MoveGroupInterface gripper_group(gripper_group_name);
    arm_group.setPlanningTime(45.0);
    arm_group.setPlannerId("RRTConnect");
    arm_group.setMaxAccelerationScalingFactor(0.30);
    arm_group.setMaxVelocityScalingFactor(0.30);
    gripper_group.setMaxAccelerationScalingFactor(0.30);
    gripper_group.setMaxVelocityScalingFactor(0.30);
    moveToNamedPose(gripper_group, "open");
    moveToCartesianPose(arm_group, target_pose_final);

    ROS_INFO_STREAM("Finished.");
    return 0;
}