import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def configure_pytorch_safety_globals() -> None:
    """Configures PyTorch to trust specific Lightning classes for secure loading.

    This is required for PyTorch 2.6+ to load legacy checkpoints securely
    while keeping the `weights_only=True` safeguard active.
    """
    torch.serialization.add_safe_globals([ModelCheckpoint])

def segment_real_image_heatmap(image_path: str, model: torch.nn.Module) -> None:
    """Loads a real image, runs inference, and visualizes the probability heatmap.

    Args:
        image_path (str): The absolute or relative path to the input image file.
        model (torch.nn.Module): The loaded PyTorch segmentation model.

    Raises:
        FileNotFoundError: If the specified image path does not exist.
    """
    try:
        original_img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find image at '{image_path}'. Please verify the path.")
        return

    # Preprocess the image for the ResNet50 encoder using ImageNet statistics
    preprocess = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    input_tensor = preprocess(original_img).unsqueeze(0)

    # Run inference using best practices (no_grad saves memory)
    model.eval()
    with torch.no_grad():
        # Get raw logits from the model
        logits = model(input_tensor)
        
        # Convert raw logits to probabilities (0.0 to 1.0 range)
        probabilities = F.softmax(logits, dim=1)
        
        # Extract only the probability map for class 1 (hand)
        # Shape changes from [1, 2, H, W] to [H, W]
        hand_prob_map = probabilities[0, 1].cpu().numpy()

    # Visualize original image vs. probability heatmap side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_img.resize((256, 256)))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display probabilities with a fixed scale from 0 to 1
    im = axes[1].imshow(hand_prob_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Hand Probability Heatmap")
    axes[1].axis('off')

    # Add a colorbar to interpret the confidence levels
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Allow the necessary PyTorch Lightning classes for secure loading
    configure_pytorch_safety_globals()

    # 2. Load the model from PyTorch Hub
    print("Loading model from PyTorch Hub...")
    model = torch.hub.load(
        'guglielmocamporese/hands-segmentation-pytorch',
        'hand_segmentor',
        pretrained=True,
        force_reload=False
    )
    print("Model loaded successfully!")

    # 3. Define path and run inference
    # Replace with an actual image path on your Ubuntu system
    IMAGE_PATH = "/home/student-team/Downloads/testset-hof/testset/8.jpg" 
    segment_real_image_heatmap(IMAGE_PATH, model)