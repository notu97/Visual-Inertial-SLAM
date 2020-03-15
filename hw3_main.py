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
	return a


def hat_operator(v):
	'''
	Hat operator
	'''
	return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

def big_dot_opp(v):
	'''
	Dot operator
	'''
	return np.block([[np.eye(3),-1*hat_operator(v)],[np.zeros((1,6))]])


#In[]
if __name__ == '__main__':
	filename = "./data/0027.npz" # Change dataset here
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
	features=features[:,::5,:] # Downsample Data. Set it to 1 to get best result
	
	world_T_imu = np.zeros((4,4,np.size(t)))
	imu_mu_t_t = np.identity(4)					# IMU inverse pose mean
	imu_sigma_t_t = np.identity(6)				# IMU inverse pose covariance
	world_T_imu[:,:,0] = imu_mu_t_t
	Proj=np.block([np.identity(3),np.zeros((3,1))])
	M=np.block([[K[0:2,:], np.zeros((2,1))],[K[0:2,:], np.zeros((2,1))]])
	M[2,3] = -K[0,0]*b
	V=15
	motion_noise=0.001
	landmark_mu_t = -1*np.ones((4,np.shape(features)[1]))	# landmark pose mean	4*M
	landmark_sigma_t = np.identity(3*np.shape(features)[1])*V	# landmark pose covariance	3M*3M
	I = np.vstack((np.identity(3),np.zeros((1,3))))
	I = np.kron(np.eye(np.shape(features)[1]),I)
	E=(np.eye(3*np.shape(features)[1]))

#In[]
	# (a) IMU Localization via EKF Prediction
	for time in np.arange(1,len(t.T),1):
		tau = t[0,time] - t[0,time-1]
		ut = np.vstack((linear_velocity[:,time].reshape(3,1),rotational_velocity[:,time].reshape(3,1)))
		ut_hat=np.hstack((np.vstack((hat_operator(rotational_velocity[:,time]),np.zeros(3))), np.hstack((linear_velocity[:,time],0)).reshape(4,1) ))
		ut_cap=np.vstack((np.hstack((hat_operator(rotational_velocity[:,time]),hat_operator(linear_velocity[:,time]))),np.hstack((np.zeros((3,3)), hat_operator(rotational_velocity[:,time]))) ))
		imu_mu_t_t=np.matmul(expm(-tau*ut_hat),imu_mu_t_t) # Predict IMU inverse Pose
		imu_sigma_t_t= np.linalg.multi_dot([expm(-tau*ut_cap), imu_sigma_t_t ,np.transpose(expm(-tau*ut_cap))])+(motion_noise*np.eye(6))
		
		world_T_imu[:,:,time] = np.linalg.inv(imu_mu_t_t)  # Save inverse pose to plot the trajectory

		Cam_T_W=np.matmul(cam_T_imu,imu_mu_t_t)
		W_T_Cam = np.linalg.inv(Cam_T_W)
		ind=np.array(np.where(np.sum(features[:,:,time],axis=0)!=-4 ))[0]
		update_feature_ind=np.empty(0,np.int16)
		H_t_next=np.zeros((4*(len(ind)),6 ))
		print(time-1)
	    #(b) Landmark pose via EKF Update
		if (len(ind.T)!=0):
			valid_features=features[:,ind,time]
			features_time=features[:,:,time]
			# Convert features from camera to world frame
			x=((valid_features[0,:]-M[0,2])*b)/ (valid_features[0,:]-valid_features[2,:])
			y=(valid_features[1,:]-M[1,2])*(-M[2,3])/(M[1,1]*(valid_features[0,:]-valid_features[2,:]))
			z= -M[2,3]/(valid_features[0,:]-valid_features[2,:])
			P_optical=np.vstack((x,y,z,np.ones(len(ind))))
			P_world_coord= np.matmul(W_T_Cam,P_optical)
				
			for j in range(len(ind)):
				curr_ind=ind[j]
				if (np.array_equal(landmark_mu_t[:,ind[j]], np.array([-1,-1,-1,-1]) )):
					# If landmark not seen before then initilize it with the world coordinate pose calculated earlier
					landmark_mu_t[:,ind[j]]=P_world_coord[:,j]
				else:
					# get the index of landmarks whose mu and sigma is to be updated
					update_feature_ind=np.append(update_feature_ind,curr_ind) 
					
			if(np.size(update_feature_ind)!=0):
				mu_t_bar=landmark_mu_t[:,update_feature_ind]
				z_t_bar=np.matmul(M,pi_func(np.dot (Cam_T_W,mu_t_bar))) 
				z_t=features_time[:,update_feature_ind]
				temp=np.matmul(Cam_T_W,mu_t_bar)
				H_t=np.zeros((4*(len(update_feature_ind)),3*(np.shape(features)[1]) ))
				
				for i in range(len(update_feature_ind)): # Build the Block diagonal matrix, H_t for landmark pose update (4*(features to update) x 3M)
					q=update_feature_ind[i]
					H_t[4*i:4*(i+1),3*q:3*(q+1)]=np.matmul(np.matmul(M,J_pi_func(temp[:,i])), np.matmul(Cam_T_W,np.transpose(Proj)))
							
				# Compute kalman Gain for landmark pose update			
				K_t=np.linalg.multi_dot([landmark_sigma_t,
										np.transpose(H_t), 
										np.linalg.inv(np.linalg.multi_dot([H_t,landmark_sigma_t,np.transpose(H_t)])+V*np.identity(4*np.size(update_feature_ind)))])	
				landmark_mu_t= ((landmark_mu_t.reshape(-1,1,order='F'))+ ( np.linalg.multi_dot([I,K_t,((z_t-z_t_bar).reshape(-1,1,order='F'))]) )).reshape(4,-1,order='F')
				landmark_sigma_t=np.dot((E-np.dot(K_t,H_t)),landmark_sigma_t)

				# (c) Visual-Inertial SLAM		
			for j in range(len(ind)): # Build 4N_t x 6 matrix for imu pose update
				H_t_next[4*j:4*(j+1),0:6]=M@J_pi_func(Cam_T_W@landmark_mu_t[:,ind[j]])@cam_T_imu@big_dot_opp((imu_mu_t_t@landmark_mu_t[:,ind[j]]))

			# calculate z bar
			Z_bar=np.matmul(M,pi_func(np.dot (Cam_T_W,landmark_mu_t[:,ind]))) 

			# Compute Kalman Gain to update IMU pose
			K_t_next=(imu_sigma_t_t@(H_t_next.T))@np.linalg.inv( (H_t_next@imu_sigma_t_t@H_t_next.T)+ (V*np.identity(4*len(ind)) ) )
			e=(K_t_next@((valid_features-Z_bar).reshape(-1,1,order='F')))

			e_cap=np.hstack((np.vstack((hat_operator(e[3:6,0]),np.zeros(3))), np.hstack((e[0:3,0],0)).reshape(4,1) ))
			
			imu_mu_t_t=expm(e_cap)@imu_mu_t_t # Update IMU Pose mean
			imu_sigma_t_t= ((np.eye(len(imu_sigma_t_t))-(K_t_next@H_t_next))@imu_sigma_t_t) # Update IMU Pose Covariance
			
			# Update landmark Pose and Covaraince with updated IMU Pose
			Cam_T_W_new=np.matmul(cam_T_imu,imu_mu_t_t)
			if(np.size(update_feature_ind)!=0):
				mu_t_bar=landmark_mu_t[:,update_feature_ind]
				z_t_bar=np.matmul(M,pi_func(np.dot (Cam_T_W_new,mu_t_bar))) 
				z_t=features_time[:,update_feature_ind]
				temp=np.matmul(Cam_T_W_new,mu_t_bar)
				H_t=np.zeros((4*(len(update_feature_ind)),3*(np.shape(features)[1]) ))
				
				for i in range(len(update_feature_ind)):
					q=update_feature_ind[i]
					H_t[4*i:4*(i+1),3*q:3*(q+1)]=np.matmul(np.matmul(M,J_pi_func(temp[:,i])), np.matmul(Cam_T_W_new,np.transpose(Proj)))

				K_t=np.linalg.multi_dot([landmark_sigma_t,
										np.transpose(H_t), 
										np.linalg.inv(np.linalg.multi_dot([H_t,landmark_sigma_t,np.transpose(H_t)])+(V*np.identity(4*np.size(update_feature_ind))))])	
				landmark_mu_t= ((landmark_mu_t.reshape(-1,1,order='F'))+ ( np.linalg.multi_dot([I,K_t,((z_t-z_t_bar).reshape(-1,1,order='F'))]) )).reshape(4,-1,order='F')
				landmark_sigma_t=np.dot((E-np.dot(K_t,H_t)),landmark_sigma_t)
	

	
	#In[]
	# You can use the function below to visualize the robot pose over time
	visualize_trajectory_2d(world_T_imu,show_ori=True)

