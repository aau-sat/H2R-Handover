# How to install libraries

## librealsense

To get a full control of the camera, set up the [librealsense](https://github.com/realsenseai/librealsense) libraries.
```bash
git clone https://github.com/realsenseai/librealsense.git
cd librealsense
mkdir build && cd build
cmake ..
cmake --build .
```

## python libraries
Optionally create a `venv` environment.
```bash
pip install pyrealsense2 # Stable
```
