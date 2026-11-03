import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
try:
    from ultralytics import YOLO
except Exception:
    # ultralytics is optional for dataset preparation; training will raise a clear error if missing
    YOLO = None


# Dataset: /home/student-team/Downloads/hand_over_face_corrected/hand_over_face
# Format: PASCAL VOC (XML annotations with polygon coordinates) + PNG masks
# 300 images total

def parse_voc_annotation(xml_path: str) -> dict:
    """
    Parse PASCAL VOC XML annotation file.
    
    Args:
        xml_path (str): Path to the XML file.
        
    Returns:
        dict: Annotation data including objects, image info, etc.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotation = {
        'filename': root.find('filename').text if root.find('filename') is not None else '',
        'width': int(root.find('imagesize/ncols').text) if root.find('imagesize/ncols') is not None else 0,
        'height': int(root.find('imagesize/nrows').text) if root.find('imagesize/nrows') is not None else 0,
        'objects': []
    }
    
    for obj in root.findall('object'):
        obj_data = {
            'name': obj.find('name').text if obj.find('name') is not None else 'hand',
            'attributes': obj.find('attributes').text if obj.find('attributes') is not None else 'unknown',
            'polygon': []
        }
        
        # Extract polygon points
        polygon = obj.find('polygon')
        if polygon is not None:
            for pt in polygon.findall('pt'):
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)
                obj_data['polygon'].append([x, y])
        
        annotation['objects'].append(obj_data)
    
    return annotation


def convert_voc_to_yolo_segmentation(xml_path: str, output_txt_path: str, class_id: int = 0) -> bool:
    """
    Convert PASCAL VOC XML annotation to YOLO segmentation format (normalized polygon coordinates).
    
    Args:
        xml_path (str): Path to the XML annotation file.
        output_txt_path (str): Path to save the YOLO format TXT file.
        class_id (int): Class ID for YOLO format (default 0 for 'hand').
        
    Returns:
        bool: True if conversion successful, False otherwise.
    """
    try:
        annotation = parse_voc_annotation(xml_path)
        
        if annotation['width'] == 0 or annotation['height'] == 0:
            print(f"Warning: Invalid image dimensions in {xml_path}")
            return False
        
        if not annotation['objects']:
            # No objects found, create empty label file
            open(output_txt_path, 'w').close()
            return True
        
        # Convert to YOLO format
        yolo_labels = []
        for obj in annotation['objects']:
            polygon = obj['polygon']
            
            if not polygon or len(polygon) < 3:
                continue
            
            # Normalize polygon coordinates to [0, 1]
            normalized_coords = []
            for x, y in polygon:
                norm_x = x / annotation['width']
                norm_y = y / annotation['height']
                # Clamp to [0, 1]
                norm_x = max(0, min(1, norm_x))
                norm_y = max(0, min(1, norm_y))
                normalized_coords.append(f"{norm_x:.6f} {norm_y:.6f}")
            
            # Format: class_id x1 y1 x2 y2 ... (all normalized)
            yolo_line = f"{class_id} " + " ".join(normalized_coords)
            yolo_labels.append(yolo_line)
        
        # Write to YOLO format TXT file
        if yolo_labels:
            with open(output_txt_path, 'w') as f:
                f.write('\n'.join(yolo_labels) + '\n')
        else:
            # No valid polygons, create empty file
            open(output_txt_path, 'w').close()
        
        return True
    
    except Exception as e:
        print(f"Error converting {xml_path}: {str(e)}")
        return False


def prepare_dataset(source_dir: str, output_dir: str, train_ratio: float = 0.8, seed: int = 42) -> None:
    """
    Prepare dataset by splitting into train and val sets.
    
    Args:
        source_dir (str): Path to the source dataset directory.
        output_dir (str): Path to the output directory for organized dataset.
        train_ratio (float): Ratio for train set (default 0.8 = 80/20 split).
        seed (int): Random seed for reproducible split.
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Get list of image IDs
    images_dir = source_path / 'images_resized'
    annotations_dir = source_path / 'annotations'
    masks_dir = source_path / 'masks'
    
    image_files = sorted([f.stem for f in images_dir.glob('*.jpg')])
    print(f"Total images found: {len(image_files)}")
    
    # Split into train and val
    split_idx = int(len(image_files) * train_ratio)
    random.shuffle(image_files)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train set: {len(train_files)} images")
    print(f"Val set: {len(val_files)} images")
    
    # Copy files to respective directories
    for split, files in [('train', train_files), ('val', val_files)]:
        split_path = output_path / split
        for img_id in files:
            # Copy image
            src_img = images_dir / f'{img_id}.jpg'
            dst_img = split_path / 'images' / f'{img_id}.jpg'
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Convert XML annotation to YOLO format
            src_ann = annotations_dir / f'{img_id}.xml'
            dst_label = split_path / 'labels' / f'{img_id}.txt'
            if src_ann.exists():
                convert_voc_to_yolo_segmentation(str(src_ann), str(dst_label), class_id=0)
            else:
                # Create empty label if no annotation
                open(dst_label, 'w').close()
            
            # Copy mask
            src_mask = masks_dir / f'{img_id}.png'
            dst_mask = split_path / 'masks' / f'{img_id}.png'
            if src_mask.exists():
                shutil.copy2(src_mask, dst_mask)
    
    print(f"\n✓ Dataset prepared at: {output_dir}")
    print(f"  - Train: {len(list((split_path.parent / 'train' / 'images').glob('*.jpg')))} images")
    print(f"  - Val: {len(list((split_path.parent / 'val' / 'images').glob('*.jpg')))} images")
    print(f"  - Train labels: {len(list((split_path.parent / 'train' / 'labels').glob('*.txt')))} YOLO format TXT files")
    print(f"  - Val labels: {len(list((split_path.parent / 'val' / 'labels').glob('*.txt')))} YOLO format TXT files")
    print(f"\nVOC annotations converted to YOLO segmentation format")


def create_dataset_config(output_dir: str, yaml_path: str) -> None:
    """
    Create a YAML configuration file for YOLO training.
    
    Args:
        output_dir (str): Path to the dataset directory.
        yaml_path (str): Path to save the YAML config file.
    """
    config_content = f"""path: {output_dir}
train: train/images
val: val/images

nc: 1
names: ['hand']
"""
    
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Configuration file created: {yaml_path}")
    
    # Verify labels are present
    train_label_list = list((Path(output_dir) / 'train' / 'labels').glob('*.txt'))
    val_label_list = list((Path(output_dir) / 'val' / 'labels').glob('*.txt'))

    print(f"Found {len(train_label_list)} train labels and {len(val_label_list)} val labels")

    # Show sample label
    if train_label_list:
        sample_label = train_label_list[0]
        try:
            with open(sample_label, 'r') as f:
                content = f.read().strip()
        except Exception:
            content = ''
        if content:
            print(f"  Sample label (/train/labels/{sample_label.name}):")
            print(f"    {content[:200]}...")


def train_segmentation_model(config_path: str, model_type: str = 'yolov8n-seg.pt', epochs: int = 100,
                            finetune: bool = False, freeze_backbone: bool = False, lr: float = 0.001) -> None:
    """
    Trains or fine-tunes a YOLO segmentation model using the specified configuration.

    Args:
        config_path (str): The absolute path to the dataset YAML configuration file.
        model_type (str, optional): The YOLO model weights to load. Defaults to 'yolov8n-seg.pt'.
        epochs (int, optional): The total number of training epochs. Defaults to 100.
        finetune (bool, optional): Whether to perform fine-tuning on pre-trained weights. Defaults to False.
        freeze_backbone (bool, optional): Whether to freeze backbone layers during fine-tuning. Defaults to False.
        lr (float, optional): Initial learning rate. Defaults to 0.001 (good for fine-tuning). Use 0.01 for training from scratch.

    Raises:
        FileNotFoundError: If the specified config_path does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Initialize the model
    model = YOLO(model_type)

    # Fine-tuning parameters
    if finetune:
        print("Fine-tuning mode enabled")
        if freeze_backbone:
            print("Freezing backbone layers")
            # Freeze all backbone layers (first 10 layers typically)
            for param in model.model.model[:10].parameters():
                param.requires_grad = False
    
    # Start training with optimal parameters
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=640,
        batch=32,           
        workers=8,         
        device='0',         # Target the primary NVIDIA GPU
        lr0=lr,             # Initial learning rate (0.001 for fine-tuning, 0.01 for training)
        lrf=0.01,           # Final learning rate ratio
        project='/home/student-team/rgmc-2026/utils/pose_tracking/depth_segmentation_tracking/models',
        name='hand_seg_finetuned' if finetune else 'hand_seg_v1',
        patience=20,        # Early stopping patience
        save=True,          # Save checkpoints
        verbose=True
    )
    
    return results


def train_from_scratch(config_path: str, model_type: str = 'yolov8n-seg.pt', epochs: int = 100) -> None:
    """
    Trains a YOLO segmentation model from scratch (no pre-training).

    Args:
        config_path (str): The absolute path to the dataset YAML configuration file.
        model_type (str, optional): The YOLO model weights to load. Defaults to 'yolov8n-seg.pt'.
        epochs (int, optional): The total number of training epochs. Defaults to 100.
    """
    print("Training from scratch with default YOLO weights as initialization")
    train_segmentation_model(config_path, model_type, epochs, finetune=False, lr=0.01)


def finetune_segmentation_model(config_path: str, pretrained_model: str, epochs: int = 50, 
                               freeze_backbone: bool = True) -> None:
    """
    Fine-tunes a pre-trained YOLO segmentation model on the hand dataset.

    Args:
        config_path (str): The absolute path to the dataset YAML configuration file.
        pretrained_model (str): Path to pre-trained model weights or model name.
        epochs (int, optional): The total number of training epochs. Defaults to 50.
        freeze_backbone (bool, optional): Whether to freeze backbone layers. Defaults to True.
    """
    print("Fine-tuning mode with lower learning rate")
    train_segmentation_model(config_path, pretrained_model, epochs, finetune=True, 
                            freeze_backbone=freeze_backbone, lr=0.001)

if __name__ == "__main__":
    # Dataset paths
    source_dataset = "/home/student-team/Downloads/hand_over_face_corrected/hand_over_face"
    output_dataset = "/home/student-team/rgmc-2026/utils/pose_tracking/depth_segmentation_tracking/data/hand_over_face_dataset"
    yaml_config = "/home/student-team/rgmc-2026/utils/pose_tracking/depth_segmentation_tracking/data/hand_dataset.yaml"
    
    # Step 1: Prepare and split the dataset
    print("=" * 60)
    print("STEP 1: Preparing dataset...")
    print("=" * 60)
    prepare_dataset(
        source_dir=source_dataset,
        output_dir=output_dataset,
        train_ratio=0.8,
        seed=42
    )
    
    # Step 2: Create YAML configuration
    print("\n" + "=" * 60)
    print("STEP 2: Creating configuration file...")
    print("=" * 60)
    create_dataset_config(output_dataset, yaml_config)
    
    # Step 3: Choose training mode
    print("\n" + "=" * 60)
    print("STEP 3: Training options")
    print("=" * 60)
    print("""
    Option A: Fine-tune pre-trained model (RECOMMENDED for small dataset)
    finetune_segmentation_model(yaml_config, pretrained_model='yolov8n-seg.pt', epochs=50, freeze_backbone=True)
    
    Option B: Train from scratch
    train_from_scratch(yaml_config, epochs=100)
    
    Option C: Custom training with parameters
    train_segmentation_model(yaml_config, model_type='yolov8m-seg.pt', epochs=100, 
                           finetune=True, freeze_backbone=False, lr=0.001)
    """)
    
    finetune_segmentation_model(yaml_config, pretrained_model='yolov8x-seg.pt', epochs=50, freeze_backbone=True)