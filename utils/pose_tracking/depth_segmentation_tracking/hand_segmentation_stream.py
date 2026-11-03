import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def configure_pytorch_safety_globals() -> None:
    """Configures PyTorch to trust specific Lightning classes for secure loading."""
    torch.serialization.add_safe_globals([ModelCheckpoint])


class DualSegmentationVisualizer(Node):
    """ROS 2 Node for synchronizing, segmenting, and visualizing two video streams.

    This node subscribes to two image topics, runs a pre-trained PyTorch 
    hand segmentation model on both frames, and overlays the resulting 
    probability heatmaps onto the original images before displaying them.

    Attributes:
        bridge (CvBridge): Utility to convert ROS Image messages to OpenCV.
        device (torch.device): Compute device (CUDA or CPU).
        model (torch.nn.Module): The loaded hand segmentation model.
        preprocess (torchvision.transforms.Compose): Image preprocessing pipeline.
    """

    def __init__(self):
        """Initializes the node, loads the PyTorch model, and sets up subscribers."""
        super().__init__('dual_segmentation_visualizer')
        self.bridge = CvBridge()
        
        self.get_logger().info('Loading PyTorch segmentation model...')
        configure_pytorch_safety_globals()
        
        # Best Practice: Automatically utilize GPU acceleration on Ubuntu if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load(
            'guglielmocamporese/hands-segmentation-pytorch',
            'hand_segmentor',
            pretrained=True,
            force_reload=False
        ).to(self.device)
        self.model.eval()
        
        self.preprocess = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.sub_image_1 = message_filters.Subscriber(self, ROSImage, 'camera1/image_raw')
        self.sub_image_2 = message_filters.Subscriber(self, ROSImage, 'camera2/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_image_1, self.sub_image_2],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info('Node ready! Waiting for synchronized video streams...')

    def process_and_overlay(self, cv_image: np.ndarray) -> np.ndarray:
        """Runs hand segmentation on a single image and overlays the heatmap.

        Args:
            cv_image (np.ndarray): The original BGR image from the camera.

        Returns:
            np.ndarray: The original image with the segmentation heatmap blended on top.
        """
        # Convert OpenCV BGR to PIL RGB for the PyTorch transforms
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            # Move tensor back to CPU for OpenCV processing
            hand_prob = probs[0, 1].cpu().numpy()
            
        # Resize probability map back to the camera's original resolution
        orig_h, orig_w = cv_image.shape[:2]
        hand_prob_resized = cv2.resize(hand_prob, (orig_w, orig_h))
        
        # Convert probabilities (0.0 - 1.0) to an 8-bit heatmap image
        heatmap_8bit = np.uint8(255 * hand_prob_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        # Blend the original image and the heatmap together
        alpha = 0.5 # 50% transparency
        blended_image = cv2.addWeighted(cv_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return blended_image

    def sync_callback(self, msg1: ROSImage, msg2: ROSImage):
        """Processes synchronized messages, applies segmentation, and displays them."""
        try:
            cv_image1 = self.bridge.imgmsg_to_cv2(msg1, 'bgr8')
            cv_image2 = self.bridge.imgmsg_to_cv2(msg2, 'bgr8')

            # Segment and overlay both frames
            overlaid_1 = self.process_and_overlay(cv_image1)
            overlaid_2 = self.process_and_overlay(cv_image2)

            # Ensure matching heights for horizontal stacking
            if overlaid_1.shape[0] != overlaid_2.shape[0]:
                height = min(overlaid_1.shape[0], overlaid_2.shape[0])
                width1 = int(overlaid_1.shape[1] * (height / overlaid_1.shape[0]))
                width2 = int(overlaid_2.shape[1] * (height / overlaid_2.shape[0]))
                overlaid_1 = cv2.resize(overlaid_1, (width1, height))
                overlaid_2 = cv2.resize(overlaid_2, (width2, height))

            combined_image = np.hstack((overlaid_1, overlaid_2))

            cv2.imshow('Dual Hand Segmentation Overlay', combined_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Failed during callback: {e}')


def main(args=None):
    """Initializes and runs the DualSegmentationVisualizer node."""
    rclpy.init(args=args)
    node = DualSegmentationVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down gracefully...')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()