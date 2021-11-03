import numpy as np 
import os 

base_path = ""
scene_name = ""

poses = np.load(os.path.join(base_path, scene_name, "poses_bounds.npy"))
max_depth, min_depth = poses[0][:-2]

calibration_dict = {}
calibration['min_bound'] = min_depth
calibration['max_bound'] = max_depth

for idx, poses in enumerate(poses):
	per_cam_calibration = {}
	pose_mat = poses[idx][:-2].reshape(3,5)
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

	calibration[str(idx)] = per_cam_calibration
