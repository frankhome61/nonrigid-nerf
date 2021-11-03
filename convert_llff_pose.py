import numpy as np 
import json
import os 

base_path = "/home/guowei/Research/View-Synthesis-Current-Works/nonrigid_nerf/data"
scene_name = "iccv-01"

poses = np.load(os.path.join(base_path, scene_name, "poses_bounds.npy"))
max_depth, min_depth = poses[0][-1], poses[0][-2]

calibration_dict = {}
calibration_dict['min_bound'] = min_depth
calibration_dict['max_bound'] = max_depth

for idx, poses in enumerate(poses):
	per_cam_calibration = {}
	pose_mat = poses[:-2].reshape(3,5)
	rotation = pose_mat[:,:3]
	translation = pose_mat[:,3]
	h, w, f = pose_mat[:,-1]
	per_cam_calibration['translation'] = translation.tolist()
	per_cam_calibration['rotation'] = rotation.tolist()
	per_cam_calibration['center_x'] =  w // 2
	per_cam_calibration['center_y'] = h // 2
	per_cam_calibration['focal_x'] = f 
	per_cam_calibration['focal_y'] = f
	per_cam_calibration['height'] = h
	per_cam_calibration['width'] = w 

	calibration_dict[str(idx)] = per_cam_calibration

with open(os.path.join(base_path, scene_name, "calibration.json"), "w") as json_file:
		json.dump(calibration_dict, json_file, indent=4)


num_frames = 5
num_cams = 10
image_to_camera_id_and_timestep = {}

for i in range(num_frames):
	for idx in range(num_cams):
		image_path = os.path.join(base_path, scene_name, "image", "{}0{}.jpg".format(i, idx))
		image_name = "{}0{}.jpg".format(i, idx)
		image_to_camera_id_and_timestep[image_name] = [str(idx), i]

with open(os.path.join(base_path, scene_name, "image_to_camera_id_and_timestep.json"), "w") as json_file:
    json.dump(image_to_camera_id_and_timestep, json_file, indent=4)

