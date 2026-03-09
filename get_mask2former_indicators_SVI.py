import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from mmseg.apis import MMSegInferencer
from typing import Union
import time

# ---------------------- Config ----------------------
image_folder = r'C:\Files\GLAN_DATA\first_wave\EMA30min\street_view_images_merged_addgansvi'
output_csv = r'C:\Files\GLAN_DATA\first_wave\EMA30min\street_view_images_merged_segmen.csv'

# 19 Cityscapes classes
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
           'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
           'motorcycle', 'bicycle']

# ---------------------- Init ----------------------
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
print(f"Found {len(image_paths):,} images")

inferencer = MMSegInferencer(
    model='mask2former_swin-t_8xb2-90k_cityscapes-512x1024',
    device='cuda:0'
)

# CSV header - added 'filename' column
columns = ['filename', 'id', 'lat', 'lon'] + [f'{c}_percent' for c in classes]

# Write header only if file doesn't exist
if not os.path.exists(output_csv):
    pd.DataFrame(columns=columns).to_csv(output_csv, index=False, encoding='utf-8-sig')


# ---------------------- Helper ----------------------
def get_pred(result: Union):
    if hasattr(result, 'predictions'):  # MMSeg 1.2+
        return result.predictions[0].pred_sem_seg.data[0].cpu().numpy()
    else:
        return result.pred_sem_seg.data[0].cpu().numpy()


# ---------------------- Main Loop ----------------------
for img_path in tqdm(image_paths, desc="Predicting", unit="img"):
    filename = os.path.basename(img_path)
    name_part = filename.replace('.jpg', '')
    parts = name_part.split('_')
    img_id, lat, lon = parts[0], parts[1], parts[2]

    # inference
    result = inferencer(img_path, return_datasamples=True)
    pred = get_pred(result)  # (H, W)

    # calculate percentages
    total = pred.size
    counts = np.bincount(pred.flatten(), minlength=19)
    percentages = (counts / total * 100.0).round(6)

    # Create row with filename as first column
    row = [filename, img_id, lat, lon] + percentages.tolist()

    # safe append
    for attempt in range(5):
        try:
            pd.DataFrame([row], columns=columns).to_csv(
                output_csv, mode='a', header=False, index=False, encoding='utf-8-sig'
            )
            break
        except PermissionError:
            if attempt == 0:
                print("CSV is open – close it and waiting 3s...")
            time.sleep(3)

print(f"\nAll done! Results saved to:\n{output_csv}")
