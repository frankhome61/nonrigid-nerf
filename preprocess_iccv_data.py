import os 
import glob
import shutil
import numpy as np
from PIL import Image

base_dir = "/home/guowei/Research/View-Synthesis-Current-Works/nonrigid_nerf/data"
scene_name = "synthetic_scene_01"

output_name= "synthetic-01-two-view"

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

poses = np.load(os.path.join(base_dir, scene_name, "poses_bounds.npy"))


# copy and organize image files 
cams = [4,5]
num_cams = len(cams)
frame_range = [0, 120]
width = 640
height = 360
factor = 1

for i in cams:
	cam_img_path = os.path.join(base_dir, scene_name, "cam0{}".format(i), "*.jpg")
	images = sorted(glob.glob(cam_img_path))[frame_range[0]:frame_range[1]]
	for frame_idx, img_path in enumerate(images):

		dst = os.path.join(output_image_dir, "{}{:04d}.jpg".format(i, frame_idx))
		im = Image.open(img_path)
		original_width, original_height = im.size
		im = im.resize((width, height), Image.LANCZOS)
		im.save(dst)
		factor = width / original_width

for p in poses:
	p[[4,9,14]] *= factor
np.save(os.path.join(base_dir, output_dir, "poses_bounds.npy"), poses)

		#shutil.copy2(img_path, os.path.join(output_image_dir, "{}{:04d}.jpg".format(i, frame_idx)))





