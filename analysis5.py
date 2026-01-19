import os
import cv2
import numpy as np
from skimage import io, feature, color, measure
from skimage.transform import resize
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\sculpture_analysis\sculptures'          # Your images folder
IMG_SIZE = (256, 256)                                  
OUTPUT_DIR = r'D:\sculpture_analysis\analysis_results'  
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELING_CSV = os.path.join(OUTPUT_DIR, 'manual_labeling.csv')  # CSV for labeling

NUM_MANUAL_DAMAGE_LABELS = 50
NUM_MANUAL_MATERIAL_LABELS = 60
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# =================================================================

# Image generator (memory-safe)
def image_generator(image_dir, size=IMG_SIZE):
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            img = io.imread(img_path, as_gray=False)
            if len(img.shape) == 2:
                img = np.stack((img,) * 3, axis=-1)
            img_resized = resize(img.astype(np.float32), size, anti_aliasing=True)
            yield img_resized, filename

# Feature functions
def compute_damage_features(img):
    gray = color.rgb2gray(img)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    return edge_density, contour_count

def compute_texture_features(img):
    gray = color.rgb2gray(img)
    gray_quant = np.clip(gray * 31, 0, 31).astype(np.uint8)
    glcm = feature.graycomatrix(gray_quant, [1], [0], levels=32, symmetric=True, normed=True)
    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    energy = feature.graycoprops(glcm, 'energy')[0, 0]
    return contrast, homogeneity, energy

def compute_entropy(img):
    gray = color.rgb2gray(img)
    return measure.shannon_entropy(gray)

def compute_color_features(img):
    hsv = color.rgb2hsv(img)
    rgb_mean = np.mean(img, axis=(0, 1))
    brightness = np.mean(hsv[:, :, 2])
    return list(rgb_mean) + [brightness]

# ========================= MAIN FEATURE EXTRACTION =========================
print("Extracting features from all images (this may take a while)...")
all_features = []

file_list = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_images = len(file_list)
print(f"Found {total_images} images.")

for i, (img, filename) in enumerate(image_generator(DATA_DIR, IMG_SIZE)):
    edge_density, contour_count = compute_damage_features(img)
    contrast, homogeneity, energy = compute_texture_features(img)
    entropy = compute_entropy(img)
    mean_r, mean_g, mean_b, brightness = compute_color_features(img)

    all_features.append({
        'filename': filename,
        'edge_density': edge_density,
        'contour_count': contour_count,
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'entropy': entropy,
        'mean_R': mean_r,
        'mean_G': mean_g,
        'mean_B': mean_b,
        'brightness': brightness,
        'true_damage': '',      # Empty → to be filled in CSV
        'true_material': ''     # Empty → to be filled in CSV
    })

    if (i + 1) % 100 == 0 or (i + 1) == total_images:
        print(f"Processed {i + 1}/{total_images} images...")

df_master = pd.DataFrame(all_features)
df_master.set_index('filename', inplace=True)

# ========================= PREPARE CSV FOR MANUAL LABELING =========================
# Randomly select images for damage and material labeling
damage_sample = random.sample(list(df_master.index), min(NUM_MANUAL_DAMAGE_LABELS, total_images))
material_sample = random.sample(list(df_master.index), min(NUM_MANUAL_MATERIAL_LABELS, total_images))

# Mark which rows need labeling
df_master['needs_damage_label'] = df_master.index.isin(damage_sample)
df_master['needs_material_label'] = df_master.index.isin(material_sample)

# Sort so that images needing labels appear at the top (easier in Excel)
df_labeling = df_master.sort_values(by=['needs_damage_label', 'needs_material_label'], ascending=False)


# Save the CSV for manual labeling
df_labeling.to_csv(LABELING_CSV)
print(f"\nManual labeling CSV created: {LABELING_CSV}")
print("\nNEXT STEPS:")
print("1. Open the CSV file in Excel or Google Sheets.")
print("2. Images at the top need labeling:")
print("   - For damage: fill 'true_damage' column with 'Damaged' or 'Undamaged'")
print("   - For material: fill 'true_material' column with one of: Terracotta, Bronze, Sandstone, Marble, Limestone, Metal, Other")
print("3. Save the file when done.")
print("4. Run the SECOND script (provided below) to train classifiers and predict on all images.")