from PIL import Image
import os
import glob

# Parent folder containing image files
parent_folder = "/home/umaru/praktikum/changed_version/ginr-ipc/results.tmp/ply/overfit_occ_sdf/occ_80_640"

# Output GIF file name
output_gif = "/home/umaru/praktikum/changed_version/ginr-ipc/results.tmp/ply/overfit_occ_sdf/occ_80_640/output.gif"

# List all image files in the parent folder and its subdirectories
image_paths = glob.glob(os.path.join(parent_folder, "*.png"), recursive=True)
sorted_image_paths = sorted(image_paths)
#sorted_image_paths= sorted(image_paths, key=lambda x: int(x.split('/')[9][12:-4]))


# Open and process each image
images = []
for image_path in sorted_image_paths:
    try:
        with Image.open(image_path) as img:
            # Determine the dimensions of the final cropped image
            target_width = min(img.size)
            target_height = min(img.size)

            # Calculate cropping box to center-crop the image
            left = (img.width - target_width) / 2
            top = (img.height - target_height) / 2
            right = (img.width + target_width) / 2
            bottom = (img.height + target_height) / 2

            # Crop the PNG image
            cropped_img = img.crop((left, top, right, bottom))
            images.append(cropped_img)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
# Save the center-cropped images as an animated GIF
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=500,  # Adjust the duration (in milliseconds) between frames
    loop=0,  # 0 means loop indefinitely, change to a different number for finite loops
)
