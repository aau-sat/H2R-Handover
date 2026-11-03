#!/usr/bin/env bash
set -euo pipefail

if [[ "${POST_STARTUP_DONE:-}" == "1" ]]; then
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y apt-transport-https curl lsb-release

mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.realsenseai.com/Debian/librealsense.pgp \
  | tee /etc/apt/keyrings/librealsense.pgp > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.realsenseai.com/Debian/apt-repo $(lsb_release -cs) main" \
  | tee /etc/apt/sources.list.d/librealsense.list > /dev/null

apt-get update
apt-get install -y \
  ros-foxy-desktop \
  python3-argcomplete \
  python3-colcon-common-extensions \
  librealsense2-dkms \
  librealsense2-utils \
  librealsense2-dev \
  librealsense2-dbg \
  ros-foxy-librealsense2*

apt-get clean
rm -rf /var/lib/apt/lists/*

cat <<'EOF' > /etc/profile.d/ros_env.sh
# ROS environment for all shells.
if [[ -f /opt/ros/foxy/setup.bash ]]; then
  source /opt/ros/foxy/setup.bash
fi
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
EOF

chmod +x /etc/profile.d/ros_env.sh

if ! grep -q '/etc/profile.d/ros_env.sh' /etc/bash.bashrc; then
  echo 'source /etc/profile.d/ros_env.sh' >> /etc/bash.bashrc
fi

export POST_STARTUP_DONE=1
