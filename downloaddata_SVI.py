# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 17:07:23 2025
@author: 15641
"""

import pandas as pd
import requests
from PIL import Image
import io
import os

# Replace with your Google API key (obtain from https://console.cloud.google.com/)
GOOGLE_API_KEY = '...'

CSV_PATH = 'C:/Files/GLAN_DATA/first_wave/EMA30min/unique_coordinates.csv'
OUTPUT_ROOT = 'C:/Files/GLAN_DATA/first_wave/EMA30min/street_view_images'
MAX_REQUESTS = 9900             # Max requests per run (adjust based on your free quota)
headings = [0, 90, 180, 270]   # North, East, South, West

# ================================================
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Read CSV with emaid, lon_gtt, lat_gtt as string types
df = pd.read_csv(CSV_PATH, dtype={'emaid': str, 'lon_gtt': str, 'lat_gtt': str})

request_count = 0

for index, row in df.iterrows():
    emaid = row['emaid']
    lon = row['lon_gtt']
    lat = row['lat_gtt']

    # Check if all headings for this coordinate have already been downloaded
    all_exist = all(os.path.exists(os.path.join(OUTPUT_ROOT, f"{emaid}_{lon}_{lat}_{h}.jpg")) for h in headings)
    if all_exist:
        print(f"[SKIP] All images for emaid {emaid} at ({lon}, {lat}) already downloaded")
        continue

    print(f"[START] Processing emaid={emaid} at ({lon}, {lat})")

    for heading in headings:
        if request_count >= MAX_REQUESTS:
            print(f"Reached maximum request limit of {MAX_REQUESTS}, stopping")
            break

        filename = f"{emaid}_{lon}_{lat}_{heading}.jpg"
        output_path = os.path.join(OUTPUT_ROOT, filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"  → Already exists: {filename}")
            continue

        url = f"https://maps.googleapis.com/maps/api/streetview" \
              f"?size=640x640&location={lat},{lon}&heading={heading}&pitch=0&fov=90&key={GOOGLE_API_KEY}"

        try:
            response = requests.get(url, timeout=30)
            request_count += 1
        except Exception as e:
            print(f"  → Request error at {heading}°: {e}")
            continue

        if response.status_code == 200 and len(response.content) > 5000:  # Valid image >5KB, avoids empty images
            img = Image.open(io.BytesIO(response.content))
            img.save(output_path)
            print(f"  → Download successful {heading}° → {filename} ({request_count}/{MAX_REQUESTS})")
        else:
            print(f"  → Download failed {heading}° (status={response.status_code}), possibly no street view coverage at this location")

    # Re-check if all headings are now complete (in case this run filled the gaps)
    all_exist = all(os.path.exists(os.path.join(OUTPUT_ROOT, f"{emaid}_{lon}_{lat}_{h}.jpg")) for h in headings)
    if all_exist:
        print(f"[DONE] All images for emaid {emaid} at this point downloaded successfully\n")
    else:
        print(f"[INCOMPLETE] emaid {emaid} at this point still has missing images, will resume on next run\n")

    if request_count >= MAX_REQUESTS:
        break

print("Run completed")
print(f"Total API requests made: {request_count}")