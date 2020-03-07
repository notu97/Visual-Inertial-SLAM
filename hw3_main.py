#In[]
import numpy as np
from utils import *

def projection(v):
	v_proj=np.divide(v,v[2])
	return v_proj


# Stereo camera calibration matrix M


#In[]
if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	M=np.block([[K[0:2,:], np.zeros((2,1))],[K[0:2,:], np.zeros((2,1))]])
	M[2,3] = -K[0,0]*b

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)


# %%


# %%
