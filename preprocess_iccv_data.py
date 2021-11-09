import os 
import glob
import shutil

base_dir = "/home/guowei/Research/View-Synthesis-Current-Works/nonrigid_nerf/data"
scene_name = "scene006"

output_name= "iccv-01-full"

# create target directory
output_dir = os.path.join(base_dir, output_name)
output_image_dir = os.path.join(base_dir, output_name, "images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# copy pose file
shutil.copy2(
	os.path.join(base_dir, scene_name, "poses_bounds.npy"), 
	os.path.join(base_dir, output_dir, "poses_bounds.npy")
)

# copy and organize image files 
num_cams = 10
frame_range = [0, 120]

for i in range(num_cams):
	cam_img_path = os.path.join(base_dir, scene_name, "cam0{}".format(i), "*.jpg")
	images = sorted(glob.glob(cam_img_path))[frame_range[0]:frame_range[1]]
	for frame_idx, img_path in enumerate(images):
		shutil.copy2(img_path, os.path.join(output_image_dir, "{}{:04d}.jpg".format(i, frame_idx)))





