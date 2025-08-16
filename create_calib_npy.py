import numpy as np
import cv2
import glob
from tqdm import tqdm

IMAGE_DIR_PATH = '/workspace/datasets/train/images'
IMG_SIZE = 640
NUM_IMAGES = 100
OUTPUT_NPY_PATH = '/workspace/calibration_set.npy'

def main():
    image_paths = sorted(glob.glob(f'{IMAGE_DIR_PATH}/*.jpg'))[:NUM_IMAGES]
    
    if not image_paths:
        print(f"Error: No .jpg files found in '{IMAGE_DIR_PATH}' folder.")
        print("Check if the <actual_folder_name> in IMAGE_DIR_PATH is correctly set.")
        return

    print(f"Processing {len(image_paths)} images to generate '{OUTPUT_NPY_PATH}'.")
    
    calibration_data = np.zeros((len(image_paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    for i, path in enumerate(tqdm(image_paths, desc="Preprocessing Images")):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        calibration_data[i] = img_resized

    np.save(OUTPUT_NPY_PATH, calibration_data)
    print(f"\nSuccess! '{OUTPUT_NPY_PATH}' has been created.")
    print(f"Data shape: {calibration_data.shape}")

if __name__ == '__main__':
    main()
