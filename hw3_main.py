#In[]
import numpy as np
from utils import *
from scipy.linalg import expm

def projection(v):
	v_proj=np.divide(v,v[2])
	return v_proj

def hat_operator(v):
	return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

# Stereo camera calibration matrix M


#In[]
if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	trajectory = np.zeros((4,4,np.size(t)))
	imu_mu_t_t = np.identity(4)					# mean
	imu_sigma_t_t = np.identity(6)				# covariance
	trajectory[:,:,0] = imu_mu_t_t

	M=np.block([[K[0:2,:], np.zeros((2,1))],[K[0:2,:], np.zeros((2,1))]])
	M[2,3] = -K[0,0]*b
	
	V = 100
	landmark_mu_t = -1*np.ones((4,np.shape(features)[1]))	# mean	4*M
	landmark_sigma_t = np.identity(3*np.shape(features)[1])*V	# covariance	3M*3M
	D = np.vstack((np.identity(3),np.zeros((1,3))))
	D = np.kron(np.eye(np.shape(features)[1]),D)
#In[]	
	time=1 # TImestamp
	# # Get valid indices and get feature coords in Left camera frame
	# ind=np.array(np.where(np.sum(features[:,:,t],axis=0)!=-4 ))[0]

	# x_by_z=(features[0,ind,t]-M[0,2])/M[0,0]
	# y_by_z=(features[1,ind,t]-M[1,2])/M[1,1]
	# one_by_z=(-features[2,ind,t]+M[2,2]+(M[2,0]*x_by_z))/M[2,3]
	# x=x_by_z/one_by_z
	# y=y_by_z/one_by_z
	# z=1/one_by_z
	# P=np.vstack((x,y,z,np.ones(len(ind))))

	# # Convert to imu coord
	# P_imu_coord= np.matmul(np.linalg.inv(cam_T_imu),P)
#In[]
	# (a) IMU Localization via EKF Prediction
	tau = t[0,time] - t[0,time-1]
	ut = np.vstack((linear_velocity[:,time].reshape(3,1),rotational_velocity[:,time].reshape(3,1)))
	ut_hat=np.hstack((np.vstack((hat_operator(rotational_velocity[:,time]),np.zeros(3))), np.hstack((linear_velocity[:,time],0)).reshape(4,1) ))
	ut_cap=np.vstack((np.hstack((hat_operator(rotational_velocity[:,time]),hat_operator(linear_velocity[:,time]))),np.hstack((np.zeros((3,3)), hat_operator(rotational_velocity[:,time]))) ))
	imu_mu_t_t=np.matmul(expm(-tau*ut_hat),imu_mu_t_t)
	imu_sigma_t_t=expm(-tau*ut_cap)@imu_sigma_t_t@np.transpose(expm(-tau*ut_cap))+np.random.multivariate_normal(np.zeros(6),0.5*np.identity(6),6)
	
	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)


# %%


# %%
