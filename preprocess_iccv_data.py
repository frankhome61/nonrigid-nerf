import os 
import glob
import shutil

base_dir = "/home/guowei/Research/View-Synthesis-Current-Works/nonrigid_nerf/data"
scene_name = "scene026"

output_name= "iccv-03-two-view"

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
cams = [4,5]
num_cams = len(cams)
frame_range = [360, 420]

for i in cams:
	cam_img_path = os.path.join(base_dir, scene_name, "cam0{}".format(i), "*.jpg")
	images = sorted(glob.glob(cam_img_path))[frame_range[0]:frame_range[1]]
	for frame_idx, img_path in enumerate(images):
		shutil.copy2(img_path, os.path.join(output_image_dir, "{}{:04d}.jpg".format(i, frame_idx)))





