# Mecanumbot ROS 2 environment (sourced ON the robot by the `robot` wrapper).
# These match the robot's ~/.bashrc; a non-interactive SSH shell does not get them.
export ROS_DOMAIN_ID=19
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
# ROS distro + this project's overlay:
#   source /opt/ros/humble/setup.bash
#   source ~/mecanumbot_ws/install/setup.bash
