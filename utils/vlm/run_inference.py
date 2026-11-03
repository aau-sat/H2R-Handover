"""
Run inference on an image using the cached Qwen3-VL model.
Requires download_and_setup_model.py to have been run first.
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os

# Set the cache directory to match where models were downloaded
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

def run_inference(image_path, prompt_text="You see a white table. In the center of the table is a object. Respond only with the object name.", model_name="Qwen/Qwen3-VL-8B-Instruct"):
    """
    Run inference on an image using the cached model.
    
    Args:
        image_path: Path to the image file
        prompt_text: Text prompt for the model
        model_name: Model identifier on Hugging Face Hub
    """
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    print(f"Loading model: {model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, 
        dtype="auto", 
        device_map="auto"
    )
    
    print("Loading processor")
    processor = AutoProcessor.from_pretrained(model_name)
    
    print(f"Running inference on: {image_path}")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text


if __name__ == "__main__":
    # Example usage
    image_path = "/home/student-team/rgmc-2026/utils/vlm/captured_data/left_00002_20260505_120120_536787.png"
    result = run_inference(image_path, "Describe this image.")
    print("\nResult:")
    print(result)
