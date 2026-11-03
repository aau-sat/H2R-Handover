symmetric_container_priors = ["cup", "glass", "bottle", "mug", "can", "cylinder", "bag", "food bag"]
asymmetric_container_priors = ["box", "food box", "food container", "food package"]
other_priors = ["bowl", "plate"]
object_priors = symmetric_container_priors + asymmetric_container_priors + other_priors

vlm_prompt = f"Color and name precisely the object at the center of white table from the fixed set: {', '.join(object_priors)} (e.g. red cup). If there is no object in the center, respond with 'NaN'. (Note: transparent cup is not 'glass')." #"You see a white table. In the center of the table is a object. Respond only with the object name. Be specific as you can. Make it generic."
left_color_camera_topic_suffix = "depth_clipped" # image_raw or depth_clipped; only the last part is specified in the config, the rest is added in the code to be more flexible
right_color_camera_topic_suffix = "depth_clipped" # image_raw or depth_clipped; only the last part is specified in the config, the rest is added in the code to be more flexible