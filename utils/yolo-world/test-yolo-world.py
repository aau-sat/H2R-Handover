from ultralytics import YOLOWorld

# 1. Load a pre-trained model (s, m, or l versions available)
model = YOLOWorld("yolov8s-world.pt") 
print(model.names)

# 2. Define your custom vocabulary
# You can detect anything here without retraining!
model.set_classes(["sunglasses", "backpack", "coffee mug", "smartphone"])

# 3. Run tracking on a video stream
# source=0 uses your webcam; you can also use "path/to/video.mp4"
results = model.track(source=0, show=True, tracker="bytetrack.yaml")