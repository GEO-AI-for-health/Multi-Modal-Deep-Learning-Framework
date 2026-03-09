import pandas as pd
import requests
from PIL import Image
import io
import os
from tqdm import tqdm

# Replace with your Mapbox API key
MAPBOX_API_KEY = 'sk'

# Output directory
OUTPUT_DIR = 'C:/Files/GLAN_DATA/first_wave/EMA30min/satellite_images_1024_zoom18'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read unique_coordinates.csv
df = pd.read_csv('C:/Files/GLAN_DATA/first_wave/EMA30min/unique_coordinates.csv')

for index, row in tqdm(df.iterrows(), total=len(df), desc="Downloading", unit="img"):
    emaid = int(row['emaid'])
    lon = row['lon_gtt']
    lat = row['lat_gtt']

    # Corrected URL format: set bearing=0 and pitch=0 to ensure orthographic imagery (nadir view),
    # size 512x512@2x (higher resolution), and append &logo=false&attribution=false to remove logo and attribution text
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},17,0,0/512x512@2x?access_token={MAPBOX_API_KEY}&logo=false&attribution=false"
    # z=18
    # @2x: downloads 1024x1024 pixel image
    #url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},18,0,0/512x512@2x?access_token={MAPBOX_API_KEY}&logo=false&attribution=false"

    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        # Filename format: emaid_lon_lat.jpg (lon and lat kept as strings to preserve original precision)
        filename = f"{emaid}_{lon}_{lat}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        img.save(output_path)
        print(f"Saved satellite image for {emaid} to {output_path}")
    else:
        print(f"Failed to download image for {emaid}: {response.status_code} - {response.reason}")