'''
   Use this in case you want to delete all contents of a folder. Dataset folders are big and this is a fast way to delete them.
'''


import os
import shutil
from tqdm import tqdm

# Define the path to the dataset folder
dataset_path = 'train'

# Check if the folder exists
if not os.path.exists(dataset_path):
    print(f"The folder '{dataset_path}' does not exist.")
    exit(1)

# List all files and directories in the dataset folder
items = [os.path.join(dataset_path, item) for item in os.listdir(dataset_path)]

# Iterate and delete each item with progress bar
for item in tqdm(items, desc="Deleting items", unit="item"):
    try:
        if os.path.isfile(item) or os.path.islink(item):
            os.remove(item)
        elif os.path.isdir(item):
            shutil.rmtree(item)
    except Exception as e:
        print(f"Error deleting {item}: {e}")

print("All files and folders deleted.")