#In[]
import numpy as np
from utils import *
from scipy.linalg import expm

def pi_func(v):
	'''
	projection function
	'''
	v_proj=np.divide(v,v[2])
	return v_proj

def J_pi_func(v):
	'''
	Jacobian of projection function
	'''
	a=np.array([[1,0,-v[0]/v[2],0],
			  [0,1,-v[1]/v[2],0],
			  [0,0,0,0],
			  [0,0,-v[3]/v[2],1]])
	a=a/v[2]
	# J=a/v[2]
	return a


def hat_operator(v):
	'''
	Hat operator
	'''
	return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

# Stereo camera calibration matrix M


#In[]
if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	o_T_r=np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
	trajectory = np.zeros((4,4,np.size(t)))
	imu_mu_t_t = np.identity(4)					# mean
	imu_sigma_t_t = np.identity(6)				# covariance
	trajectory[:,:,0] = imu_mu_t_t
	Proj=np.block([np.identity(3),np.zeros((3,1))])

	M=np.block([[K[0:2,:], np.zeros((2,1))],[K[0:2,:], np.zeros((2,1))]])
	M[2,3] = -K[0,0]*b
	V=10
	landmark_mu_t = -1*np.ones((4,np.shape(features)[1]))	# mean	4*M
	landmark_sigma_t = np.identity(3*np.shape(features)[1])*10	# covariance	3M*3M
	I = np.vstack((np.identity(3),np.zeros((1,3))))
	I = np.kron(np.eye(np.shape(features)[1]),I)
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
	for time in np.arange(1,len(t.T),1):
		tau = t[0,time] - t[0,time-1]
		ut = np.vstack((linear_velocity[:,time].reshape(3,1),rotational_velocity[:,time].reshape(3,1)))
		ut_hat=np.hstack((np.vstack((hat_operator(rotational_velocity[:,time]),np.zeros(3))), np.hstack((linear_velocity[:,time],0)).reshape(4,1) ))
		ut_cap=np.vstack((np.hstack((hat_operator(rotational_velocity[:,time]),hat_operator(linear_velocity[:,time]))),np.hstack((np.zeros((3,3)), hat_operator(rotational_velocity[:,time]))) ))
		imu_mu_t_t=np.matmul(expm(-tau*ut_hat),imu_mu_t_t)
		imu_sigma_t_t=expm(-tau*ut_cap)@imu_sigma_t_t@np.transpose(expm(-tau*ut_cap))+np.random.multivariate_normal(np.zeros(6),0.5*np.identity(6),6)
		trajectory[:,:,time] = np.linalg.inv(imu_mu_t_t)
		# (b) Landmark Mapping via EKF Update

		Cam_T_W=np.matmul(cam_T_imu,imu_mu_t_t)
		W_T_Cam = np.linalg.inv(Cam_T_W)
		ind=np.array(np.where(np.sum(features[:,:,time],axis=0)!=-4 ))[0]
		# x_by_z=(features[0,ind,time]-M[0,2])/M[0,0]
		# y_by_z=(features[1,ind,time]-M[1,2])/M[1,1]
		# one_by_z=(-features[2,ind,time]+M[2,2]+(M[2,0]*x_by_z))/M[2,3]
		# x=x_by_z/one_by_z
		# y=y_by_z/one_by_z
		# z=1/one_by_z
		# # P_optical=np.linalg.pinv(M)@features[:,:,time]
		# P_optical=np.vstack((x,y,z,np.ones(len(ind))))
		# P_cam=np.matmul(np.linalg.inv(o_T_r),P_optical)
		# # Convert to world frame
		# P_world_coord= np.matmul(W_T_Cam,P_cam)
		#In[]
		if (len(ind.T)!=0):
			valid_features=features[:,ind,time]
			x_by_z=(valid_features[0,:]-M[0,2])/M[0,0]
			y_by_z=(valid_features[1,:]-M[1,2])/M[1,1]
			one_by_z=(-valid_features[2,:]+M[2,2]+(M[2,0]*x_by_z))/M[2,3]
			x=x_by_z/one_by_z
			y=y_by_z/one_by_z
			z=1/one_by_z
			# P_optical=np.linalg.pinv(M)@features[:,:,time]
			P_optical=np.vstack((x,y,z,np.ones(len(ind))))
			P_cam=np.matmul(np.linalg.inv(o_T_r),P_optical)
			# Convert to world frame
			P_world_coord= np.matmul(W_T_Cam,P_cam)
			update_feature_ind=np.empty(0,np.int16)
			update_feature=np.zeros((4,1))

		# In[]	
			for j in range(len(ind)):
				curr_ind=ind[j]
				if (np.array_equal(landmark_mu_t[:,ind[j]], np.array([-1,-1,-1,-1]) )):
					landmark_mu_t[:,ind[j]]=P_world_coord[:,j]
				else:
					# get the index of landmarks whose mu and sigma is to be updated
					update_feature_ind=np.append(update_feature_ind,curr_ind) 
					update_feature=np.hstack((update_feature,P_world_coord[:,j].reshape(4,1)))

			if(np.size(update_feature_ind)!=0):
				mu_t_bar=landmark_mu_t[:,update_feature_ind]
				z_t_bar=M@pi_func(o_T_r@Cam_T_W@mu_t_bar)
				z_t=valid_features[:,update_feature_ind]
				temp=Cam_T_W@mu_t_bar
				H_t=np.zeros((4*(len(temp.T)),3*(np.shape(features)[1]) ))
				for i in range(len(temp)):
					q=update_feature_ind[i]
					H_t[4*i:4*(i+1),3*q:3*(q+1)]=M@J_pi_func(temp[:,i])@(Cam_T_W@Proj.T)
				
				K_t=landmark_sigma_t@H_t.T@ np.linalg.inv((H_t@landmark_sigma_t@H_t.T)+np.identity(4*np.size(update_feature_ind))*V)
				landmark_mu_t=((landmark_mu_t.reshape(-1,1))+ ((I@K_t)@( (z_t-z_t_bar).reshape(-1,1)) )).reshape(4,-1)
				landmark_sigma_t=(np.eye(3*np.shape(features)[1])-(K_t@H_t))@landmark_sigma_t



	# (c) Visual-Inertial SLAM
#In[]
	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
	visualize_trajectory_2d(trajectory,landmark_mu_t,path_name="Unknown",show_ori=True)


# %%


# %%
