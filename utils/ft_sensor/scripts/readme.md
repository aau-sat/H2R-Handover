# General
1. Build the ros2_ws and source it
    - if there is a build issue you need to add the cmake args that are specified on anytype in the useful commands document
2. (optional) calibrate the sensor, is only needed between changes of the sensor
3. run `ros2 run ft_sensor publisher`

# Mass Estimation
1. execute the mass.py (it will continuesly publish the estimated weight)
2.  in order to mittigate the drift you need to regularly call the zero_calibrate_ft_sensor action; best if done in the robots home position; between each handover (`ros2 action send_goal /zero_calibrate_ft_sensor ft_sensor_msgs/action/Empty "{}"`)
3. when the robot has gripped the object we need to start mass estimation (`ros2 action send_goal /start_weight_estimation ft_sensor_msgs/action/Empty "{}"`)
4. when the object is back on the table or close to the table the estimation should be ended (`ros2 action send_goal /end_weight_estimation ft_sensor_msgs/action/Empty "{}"`)
    - the resulting estimated weight is logged to the console

# Touch down Detection
1. Run the touch down detection so it can run in the background