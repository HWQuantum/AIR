import numpy as np
import matplotlib.pyplot as plt
from scipy import io as sio
import os
 
# Input parameters
 
save_path = ## TO FILL


for i in range(10):
    print(i)
    training_file_name = 'Image_Inflation'+str(i)


    res = 80 #27 # resolution of data cube
    N = 1000 # no. of cubes simulated
    K = 1 # max. no. of allowed rings in each cube
    
    R_max = 0.9 # max radius
    R_min = 0.4 # min radius
    
    W_max = 0.15 # max width
    W_min = 0.025 # min width
    
    # Create space for output
    
    I_3D = np.zeros((res,res,res,N))
    
    I_2D_proj = np.zeros((res,res,N))
    
    # Create axes, 3D voxel grid and polar coordinate system
    
    x = np.linspace(0, 1, num=res)
    y = x
    z = x
    
    X,Y,Z = np.meshgrid(x,y,z)
    
    R = np.sqrt([np.square(X) + np.square(Y) + np.square(Z)])
    R = np.squeeze(R)
    
    COST = Z/R
    
    COST[np.isnan(COST)] = 1
    
    P2 = 0.5*(3*np.square(COST) - 1)
    
    PHI = np.arctan2(X,Y)
    
    P22 = 3*(1 - np.square(COST))
    
    # Generate random training
    
    for n in range(N):
    
        I = np.zeros((res,res,res))
    
        for j in range(K):
    
            check = 10
    
            while check > 1:
    
                R0 = np.random.rand()*(R_max - R_min) + R_min
                W0 = np.random.rand()*(W_max - W_min) + W_min
                check = (R0 + 2*W0)
                check = np.amin(check)
                
                check = -1;
    
            while check < 0:
    
                B2 = -1 + (2-(-1))*np.random.rand()
                B22 = -1 + 2*np.random.rand()
                
                I_ = np.exp(-1*np.square((R-R0)/W0))*(1 + B2*P2 + B22*P22*np.cos(2*PHI))
                
                check = np.min(I_)
                
                I = I + I_
    
                I_3D[:,:,:,n] = I/np.max(I)
                
                I_pro = np.transpose(np.nansum(I,axis=1))
                
                I_2D_proj[:,:,n] = I_pro/np.max(I_pro)
    
    sio.savemat(os.path.join(save_path, training_file_name + '_I_3D.mat'),{'I_3D':I_3D})
    sio.savemat(os.path.join(save_path, training_file_name + '_I_2Dproj.mat'),{'I_2D_proj':I_2D_proj})
#print('Done!')
#print(B22)
 
# # Show examples (comment out if you don't care!)
 
# data = I[:,2,:]
 
# data = np.transpose(data)
 
# data = np.squeeze(data)
 
# plt.imshow(data, interpolation='bicubic')
# plt.gca().invert_yaxis()
# plt.title('Slice')
# plt.show()
 
# data = np.nansum(I,axis=1)
 
# data = np.transpose(data)
 
# data = np.squeeze(data)
 
# plt.imshow(data, interpolation='bicubic')
# plt.gca().invert_yaxis()
# plt.title('Projection')
# plt.show()

