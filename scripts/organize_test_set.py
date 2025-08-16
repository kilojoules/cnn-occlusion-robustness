import os
import pandas as pd
import shutil
from tqdm import tqdm

def organize_gtsrb_test_set():
    """
    Reads the GTSRB test set CSV and organizes the flat image files
    into class-specific subfolders.
    """
    # --- Configuration ---
    # This script assumes it's being run from the project's root directory.
    base_path = '../GTSRB_dataset/GTSRB_test/Final_Test/'
    images_dir = os.path.join(base_path, 'Images')

    # CORRECTED path to the CSV file, now pointing inside the Images directory
    # CORRECTED filename to 'GT-final_test.test.csv'
    csv_path = os.path.join(images_dir, 'GT-final_test.test.csv')

    print(f"Reading image labels from: {csv_path}")

    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: Ground truth file not found at '{csv_path}'")
        print("Please ensure the path is correct and you've unzipped the dataset.")
        return

    # Read the CSV file. GTSRB uses semicolons as separators.
    try:
        df = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print(f"Found {len(df)} image entries.")
    print("Starting to organize images into class folders...")

    # Iterate over each row in the dataframe
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row['Filename']
        class_id = row['ClassId']

        # Create the five-digit folder name (e.g., 5 -> '00005')
        class_folder_name = f"{class_id:05d}"

        # Create the destination directory if it doesn't exist
        dest_dir = os.path.join(images_dir, class_folder_name)
        os.makedirs(dest_dir, exist_ok=True)

        # Construct the full source and destination paths
        src_path = os.path.join(images_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        # Move the file, checking if it exists first
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)

    print("\n✅ Test set reorganization complete!")
    print(f"Images have been moved into class-specific subfolders inside '{images_dir}'")


if __name__ == '__main__':
    organize_gtsrb_test_set()
