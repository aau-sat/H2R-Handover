#!/usr/bin/env python3
import os
import time
import cv2
import torch
import numpy as np
import torchvision
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage
from mesh_generator.action import GenerateMesh
import pathlib

# --- InstantMesh / Diffusers Imports (based on your snippet) ---
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.infer_util import remove_background, resize_foreground
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras

import rembg
import trimesh
import open3d as o3d
from einops import rearrange

# Local imports from the InstantMesh repository
# This script must be run from the root of the InstantMesh repo
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from utils.image_mesh_tools import diffusion_image_generation
# from src.utils.mesh_util import save_obj


def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):
    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3,
        faces=facenp_fx3,
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'obj')

def align_mesh_to_coordinate(mesh, scale=0.1):
    # mesh 스케일 조정
    mesh.vertices *= scale
    # Open3D mesh로 변환
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))

    obb = mesh_o3d.get_oriented_bounding_box()
    # obb.color = (1, 0, 0)
    # mesh의 중심점 계산
    center = mesh_o3d.get_center()

    # mesh의 중심점을 기준으로 회전
    mesh_o3d.rotate(np.linalg.inv(obb.R), center=center)

    # 회전된 mesh의 vertices를 원본 mesh에도 적용
    mesh.vertices = np.asarray(mesh_o3d.vertices)

    # 새로운 bounding box 계산
    # new_obb = mesh_o3d.get_oriented_bounding_box()
    # new_obb.color = (1, 0, 0)

    # return mesh, mesh_o3d, new_obb
    return mesh

def get_bounding_box(d_model, pad_rel=0.00, return_torch=False):
    """
    Get the bounding box of the non-zero elements in the input array with optional padding.

    Parameters:
    d_model (np.ndarray): 2D input array.
    pad_rel (float): Relative padding (default: 5%)

    Returns:
    torch.Tensor: Padded bounding box in xyxy format on CUDA.
    """
    # Find the indices of non-zero elements
    non_zero_indices = np.nonzero(d_model)

    if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
        x_min = y_min = x_max = y_max = 0
    else:
        # Get the bounding box coordinates
        y_min, x_min = np.min(non_zero_indices, axis=1)
        y_max, x_max = np.max(non_zero_indices, axis=1)

    # Calculate padding
    x_pad = pad_rel * (x_max - x_min)
    y_pad = pad_rel * (y_max - y_min)

    # Get image dimensions
    height, width = d_model.shape

    # Create padded bounding box
    x1 = max(0, x_min - x_pad)
    y1 = max(0, y_min - y_pad)
    x2 = min(width - 1, x_max + x_pad)
    y2 = min(height - 1, y_max + y_pad)

    # Create the bounding box in xyxy format
    bounding_box = np.array([x1, y1, x2, y2])
    if return_torch:
        return torch.from_numpy(bounding_box[None]).cuda()
    else:
        return bounding_box


def preprocess_image(color, mask, debug_dir=None, name=None, rem_bg=True, target_bright = 999, flip=True):
    """
    Preprocess input image by applying mask, removing background and resizing.

    Args:
        color: Input color image (numpy array)
        mask: Boolean mask array
        debug_dir: Directory to save debug images
        name: Optional name for the saved image

    Returns:
        PIL Image: Preprocessed image
    """
    # Create a copy and apply white background where mask is False
    input_image = color.copy()
    input_image[mask == False] = [255, 255, 255]

    # Remove background and resize
    if rem_bg:
        rembg_session = rembg.new_session()
        input_image = remove_background(Image.fromarray(input_image), rembg_session)
    else:
        x1, y1, x2, y2 = get_bounding_box(mask, pad_rel=0.1)
        input_image = Image.fromarray(input_image[int(y1):int(y2), int(x1):int(x2)])
        input_image = input_image.convert("RGBA")
    if flip:
        input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
    input_image = resize_foreground(input_image.convert("RGBA"), 0.80)

    # Save debug image if directory is provided
    if debug_dir:
        input_image.save(os.path.join(debug_dir, f'input_{name}.png'))

    return input_image

class InstantMeshActionServer(Node):
    def __init__(self):
        super().__init__('instant_mesh_server')

        # --- 1. ROS Parameters Configuration ---
        # We replace argparse with ROS parameters for runtime flexibility
        self.declare_parameter('config_path', 'configs/instant-mesh-base.yaml')
        self.declare_parameter('mesh_output_path', '/home/student-team/rgmc-2026/utils/mesh_generation/instantmesh/outputs/ros_generated')
        self.declare_parameter('diffusion_steps', 75)  # Reduced from 75 for speed in robotics
        self.declare_parameter('seed', 42)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('save_memory', False)  # Whether to free VRAM between stages

        # Retrieve values
        self.config_path = self.get_parameter('config_path').value
        self.output_dir = self.get_parameter('mesh_output_path').value
        self.steps = self.get_parameter('diffusion_steps').value
        self.seed = self.get_parameter('seed').value
        self.device_name = self.get_parameter('device').value
        self.save_memory = self.get_parameter('save_memory').value

        # Setup environment
        os.makedirs(self.output_dir, exist_ok=True)
        seed_everything(self.seed)
        self.device = torch.device(self.device_name)

        # --- 2. Action Server Setup ---
        # ReentrantCallbackGroup allows processing cancel requests/feedback while GPU is busy
        self.cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            GenerateMesh,
            'generate_mesh',
            execute_callback=self.execute_callback,
            callback_group=self.cb_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        self.bridge = CvBridge()
        
        # --- 3. Model Loading (Stage 0) ---
        self.get_logger().info('Initializing AI Models... This might take a moment.')
        # self._load_models()
        self._load_diffusion_model()
        if not self.save_memory:
            self._load_reconstruction_model()
        self.get_logger().info(f'InstantMesh Server Ready! (Device: {self.device_name})')

    def _load_models(self):
        """
        Loads the Diffusion and Reconstruction models into VRAM during node startup.
        This prevents loading delays during the actual action execution.
        """
        # Load Config
        if not os.path.exists(self.config_path):
            self.get_logger().error(f"Config file not found: {self.config_path}")
            return
            
        self.config = OmegaConf.load(self.config_path)
        self.config_name = os.path.basename(self.config_path).replace('.yaml', '')
        model_config = self.config.model_config
        self.infer_config = self.config.infer_config

        # A. Load Diffusion Model (Zero123++)
        self.get_logger().info("Loading Diffusion Model...")

        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", 
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # Load custom UNet (white background)
        if os.path.exists(self.infer_config.unet_path):
            unet_ckpt_path = self.infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
            
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        self.pipeline.unet.load_state_dict(state_dict, strict=True)
        self.pipeline = self.pipeline.to(self.device)

        # B. Load Reconstruction Model (LRM / InstantMesh)
        self.get_logger().info("Loading Reconstruction Model...")
        self.model = instantiate_from_config(model_config)
        
        if os.path.exists(self.infer_config.model_path):
            model_ckpt_path = self.infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{self.config_name.replace('-', '_')}.ckpt", repo_type="model")
            
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        # Filter state dict for the generator only
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)

        # Initialize Flexicubes if required by config
        # self.is_flexicubes = self.config_name.startswith('instant-mesh')
        # if self.is_flexicubes:
        self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        
        self.model.eval()
    
    # load models one after another in separate functions to save VRAM
    def _load_diffusion_model(self):
        if not os.path.exists(self.config_path):
            self.get_logger().error(f"Config file not found: {self.config_path}")
            return
            
        self.config = OmegaConf.load(self.config_path)
        self.config_name = os.path.basename(self.config_path).replace('.yaml', '')
        self.infer_config = self.config.infer_config

        # A. Load Diffusion Model (Zero123++)
        self.get_logger().info("Loading Diffusion Model...")

        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", 
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # Load custom UNet (white background)
        if os.path.exists(self.infer_config.unet_path):
            unet_ckpt_path = self.infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
            
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        self.pipeline.unet.load_state_dict(state_dict, strict=True)
        self.pipeline = self.pipeline.to(self.device)
    
    def _load_reconstruction_model(self):
        self.infer_config = self.config.infer_config

        # B. Load Reconstruction Model (LRM / InstantMesh)
        self.get_logger().info("Loading Reconstruction Model...")
        self.model = instantiate_from_config(self.config.model_config)
        
        if os.path.exists(self.infer_config.model_path):
            model_ckpt_path = self.infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{self.config_name.replace('-', '_')}.ckpt", repo_type="model")
            
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        # Filter state dict for the generator only
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)

        # Initialize Flexicubes if required by config
        # self.is_flexicubes = self.config_name.startswith('instant-mesh')
        # if self.is_flexicubes:
        self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        
        self.model.eval()

    # --- Model Utils ---
    def _instant_mesh_process(self, images, save_path):
        images = images.unsqueeze(0).to(self.device)
        images = torchvision.transforms.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * 1.0).to(self.device)
        chunk_size = 5
        with torch.no_grad():
            planes = self.model.forward_planes(images, input_cameras)
    
            mesh_out = self.model.extract_mesh(planes, use_texture_map=False, **self.infer_config, )
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, save_path)
        torch.cuda.empty_cache()  # Clear the GPU memory
        # return vertices, faces, vertex_colors

    def _diffusion_image_generation(self, input_image):
        output_image = self.pipeline(input_image, num_inference_steps=75, ).images[0]

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)
        return images

    # --- Action Server Callbacks ---

    async def execute_callback(self, goal_handle):
        """
        Main execution logic. Runs the inference pipeline.
        This is an async function to allow feedback publication.
        """
        self.get_logger().info('Starting execution...')
        
        feedback_msg = GenerateMesh.Feedback()
        result = GenerateMesh.Result()

        try:
            # --- Step 1: Input Processing ---
            feedback_msg.current_status = "Processing Input Images"
            goal_handle.publish_feedback(feedback_msg)

            # Convert ROS Messages to OpenCV
            # Mask should be mono8 (grayscale), Image rgb8
            cv_rgb = self.bridge.imgmsg_to_cv2(goal_handle.request.rgb_image, desired_encoding='rgb8')
            cv_mask = self.bridge.imgmsg_to_cv2(goal_handle.request.mask_image, desired_encoding='mono8')

            cv_mask = cv_mask.astype(np.bool_)

            input_image = preprocess_image(cv_rgb, cv_mask, rem_bg=True, flip=False)

            # --- Step 2: Multiview Generation (Diffusion) ---
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return GenerateMesh.Result(success=False, message="Canceled by user")

            # feedback_msg.current_status = "Running Diffusion (Stage 1)"
            # feedback_msg.progress = 10.0
            # goal_handle.publish_feedback(feedback_msg)

            start_time = time.time()

            # Run pipeline
            # Note: We use torch.no_grad() to save memory
            torch.cuda.empty_cache()
            images = self._diffusion_image_generation(input_image=input_image)

            if self.save_memory:
                del self.pipeline  # Free VRAM
                torch.cuda.empty_cache() # added to clear memory
                self._load_reconstruction_model()  # Load reconstruction model

            # feedback_msg.progress = 50.0
            # goal_handle.publish_feedback(feedback_msg)

            # --- Step 3: 3D Reconstruction ---
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return GenerateMesh.Result(success=False, message="Canceled by user")

            # feedback_msg.current_status = "Reconstructing Mesh (Stage 2)"
            # goal_handle.publish_feedback(feedback_msg)
                
            # Save the result
            # check that goal_handle.request.mesh_path's folder is valid
            # extract folder from a filename
            # if goal_handle.request.mesh_path == None or goal_handle.request.mesh_path == "":
            self.get_logger().warn("No output name provided, using default path.")
            save_path = os.path.join(self.output_dir, 'mesh.obj')
            # else:
            #     output_folder = pathlib.Path(goal_handle.request.mesh_path).parent
            #     if not output_folder.exists():
            #         output_folder.mkdir(parents=True, exist_ok=True)
            #     save_path = goal_handle.request.mesh_path
            self._instant_mesh_process(images, save_path)
            print(f"Total Inference Time: {time.time() - start_time:.2f} seconds")
            mesh = trimesh.load(save_path)
            mesh = align_mesh_to_coordinate(mesh)
            save_path = os.path.join(self.output_dir, 'mesh.obj')
            mesh.export(save_path) # overwrite with aligned mesh

            # --- Step 4: Finalize ---
            goal_handle.succeed()
            result.mesh_path = save_path
            result.status = True

            # feedback_msg.progress = 100.0
            # feedback_msg.current_status = "Finished"
            # goal_handle.publish_feedback(feedback_msg)
            
            self.get_logger().info(f"Task Complete. Saved to: {save_path}")
            
            if self.save_memory:
                del self.model  # Free VRAM
                torch.cuda.empty_cache()  # Clear the GPU memory

            return result

        except Exception as e:
            self.get_logger().error(f"Inference failed: {str(e)}")
            goal_handle.abort()
            result.status = False
            # result.message = f"Error: {str(e)}"
            return result

    def goal_callback(self, goal_request):
        """Accept new goals."""
        self.get_logger().info('Received new mesh generation request.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle cancellation requests."""
        self.get_logger().info('Received cancellation request.')
        return CancelResponse.ACCEPT


def main(args=None):
    rclpy.init(args=args)
    
    node = InstantMeshActionServer()
    
    # Use MultiThreadedExecutor to ensure the action server remains responsive
    # during long-running GPU operations
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()