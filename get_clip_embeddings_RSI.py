import os
import torch
import glob
import argparse
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from tqdm import tqdm

# Set HF Token
os.environ["HF_TOKEN"] = "...."

# ==========================================
# 1. Path Configuration
# ==========================================
SAT_DIR = r"./satellite_images"
SAT_PT_DIR = r"./satellite_images_clip_pt"

# Hardware Configuration
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f">>> Device: {device} | Dtype: {dtype}")


# ==========================================
# 2. Model Loading
# ==========================================
def load_clip_model():
    model_id = "flax-community/clip-rsicd-v2"
    print(f">>> [Satellite Mode] Loading: {model_id}")

    try:
        model = CLIPVisionModel.from_pretrained(model_id, torch_dtype=dtype)
        processor = CLIPImageProcessor.from_pretrained(model_id)
        model.to(device)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"!!! Model Load Failed: {e}")
        exit(1)


# ==========================================
# 3. Process Satellite Images
# ==========================================
def process_satellite(start_idx=0, end_idx=None):
    # Load model locally to free memory after use
    model, processor = load_clip_model()

    print(f"\n>>> [Processing Satellite] Dir: {SAT_DIR}")
    os.makedirs(SAT_PT_DIR, exist_ok=True)

    image_paths = glob.glob(os.path.join(SAT_DIR, "*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_paths.sort()

    total_files = len(image_paths)
    if end_idx is None or end_idx > total_files:
        end_idx = total_files

    target_paths = image_paths[start_idx:end_idx]
    print(f">>> Range: [{start_idx} : {end_idx}] (Total: {len(target_paths)})")

    if not target_paths:
        print("!!! No files in range.")
        return

    for img_path in tqdm(target_paths, desc=f"Sat [{start_idx}-{end_idx}]"):
        fname = os.path.basename(img_path)
        save_name = os.path.splitext(fname)[0] + ".pt"
        save_path = os.path.join(SAT_PT_DIR, save_name)

        # Skip if already processed
        if os.path.exists(save_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype)

            with torch.no_grad():
                out = model(**inputs)
                feat = out.last_hidden_state.squeeze(0)

            torch.save(feat.cpu().bfloat16(), save_path)

        except Exception as e:
            print(f"!!! Failed: {fname} - {e}")

    # Clean up GPU memory
    del model
    del processor
    torch.cuda.empty_cache()


# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Feature Extraction for Satellite Images")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")

    args = parser.parse_args()

    process_satellite(start_idx=args.start, end_idx=args.end)

    print("\n>>> Feature extraction complete!")