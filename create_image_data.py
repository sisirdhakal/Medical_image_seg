import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

anemic_dataset_india_dir = "/kaggle/input/eyes-defy-anemia/dataset anemia/India"
new_dataset_dir = "/kaggle/output/new_dataset"

# creating directories for images and masks
os.makedirs(os.path.join(new_dataset_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_dir, "masks"), exist_ok=True)

# listing all entries in the specified directory
all_entries = os.listdir(anemic_dataset_india_dir)
all_entries = [entry for entry in all_entries if entry != "India.xlsx"]
all_entries = sorted(all_entries, key=lambda x: int(x))

# preparing lists to hold the image and mask paths
images = []
masks = []

failed_entries = []

for entry in all_entries:
    # constructing the path for the entry's directory
    entry_dir = os.path.join(anemic_dataset_india_dir, entry)

    # checking if the entry directory exists
    if not os.path.isdir(entry_dir):
        print(f"{entry_dir} is not a directory.")
        continue

    # initializing file paths
    image_path = None
    mask_path = None

    # listing files in the entry directory
    for filename in os.listdir(entry_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(entry_dir, filename)
        elif filename.endswith("_forniceal_palpebral.png"):
            mask_path = os.path.join(entry_dir, filename)

    # checking if both files were found
    if image_path is None or mask_path is None:
        print(f"missing files for entry: {entry}")
        continue  # skipping to the next entry if files are missing

    # loading the segmented image for mask creation
    try:
        segmented_image = Image.open(mask_path)  # ensuring it's RGBA
        print(segmented_image.mode)
        image_array = np.array(segmented_image)

        # checking the shape of the image array
        if image_array.shape[2] < 4:
            print(f"not an RGBA image for entry: {entry}")
            continue  # skipping if not an RGBA image

        # separating the RGBA channels
        a = image_array[..., 3]  # getting the alpha channel

        # creating a binary mask where the alpha channel is greater than 0
        binary_mask = (a == 255).astype(np.uint8)  # converting to binary mask

        # saving the original image and binary mask to the new dataset
        new_image_path = os.path.join(new_dataset_dir, "images", f"{entry}.jpg")
        new_mask_path = os.path.join(new_dataset_dir, "masks", f"{entry}_mask.png")

        Image.fromarray(image_array).save(new_image_path)
        Image.fromarray(binary_mask * 255).save(
            new_mask_path
        )  # scaling mask for visibility

        # appending valid image and mask paths to the lists
        images.append(new_image_path)
        masks.append(new_mask_path)

    except Exception as e:
        failed_entries.append(entry)
        # print(f"error processing entry {entry}: {e}")
        # continue  # skipping to the next entry if there's an error


# for the failed entries trying with cv2 

for entry in failed_entries:
    # constructing the path for the entry's directory
    entry_dir = os.path.join(anemic_dataset_india_dir, entry)

    # checking if the entry directory exists
    if not os.path.isdir(entry_dir):
        print(f"{entry_dir} is not a directory.")
        continue

    # initializing file paths
    image_path = None
    mask_path = None

    # listing files in the entry directory
    for filename in os.listdir(entry_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(entry_dir, filename)
        elif filename.endswith("_forniceal_palpebral.png"):
            mask_path = os.path.join(entry_dir, filename)

    # checking if both files were found
    if image_path is None or mask_path is None:
        print(f"missing files for entry: {entry}")
        continue  # skipping to the next entry if files are missing

    # loading the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"failed to read image: {image_path}")
        continue

    # loading the segmented image for mask creation
    try:
        segmented_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        image_array = np.array(segmented_image)
        rgba_image = image_array[..., [2, 1, 0, 3]]

        # normalizing 16-bit image to 8-bit
        if np.max(rgba_image) > 255:
            rgba_image = (rgba_image / 256).astype(
                np.uint8
            )  # scaling down to 0-255 range

        # checking the shape of the image array
        if rgba_image.shape[2] < 4:
            print(f"not an RGBA image for entry: {entry}")
            continue  # skipping if not an RGBA image

        # separating the RGBA channels
        a = rgba_image[..., 3]  # getting the alpha channel

        print(np.unique(rgba_image))

        # creating a binary mask where the alpha channel is greater than 0
        binary_mask = (a == 255).astype(np.uint8)  # converting to binary mask

        # plotting original image and binary mask side by side
        plt.figure(figsize=(10, 5))

        # original image
        plt.subplot(1, 3, 1)
        plt.imshow(
            cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        )  # converting BGR to RGB
        plt.axis("off")
        plt.title(f"original image {entry}")

        # binary mask
        plt.subplot(1, 3, 2)
        plt.imshow(binary_mask, cmap="gray")
        plt.axis("off")
        plt.title(f"binary mask {entry}")

        # segmented img
        plt.subplot(1, 3, 3)
        plt.imshow(rgba_image, cmap="gray")
        plt.axis("off")
        plt.title(f"segmented img {entry}")

        plt.show()

    except Exception as e:
        print(f"error processing entry {entry}: {e}")
