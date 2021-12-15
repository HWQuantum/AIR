

# Arbitrary Image Reinflation: A deep learning technique for recovering 3D photoproduct distributions from a single 2D projection

Chris Sparling<sup> 1,‡ </sup>, Alice Ruget<sup> 1,‡ </sup>, Jonathan Leach<sup> 1</sup> and Dave Townsend<sup> 1,2 </sup>

<sup> 1 </sup> Institute of Photonics & Quantum Sciences, Heriot-Watt University, Edinburgh, EH14 4AS, UK

<sup> 2 </sup> Institute of Chemical Sciences, Heriot-Watt University, Edinburgh, EH14 4AS, UK

<sup> ‡ </sup> These authors contributed equally to this work


## I. Dependencies 
Python 3.8.11
Tensorflow 2.4.1
Keras 2.4.0

## II. Training Dataset
Create_dataset.py is used to simulate different 3D distributions I_3D and their corresponding 2D projections I_2D_proj.  
1. Fill the saving path save_path in create_dataset.py
2. Adjust the different parameters 
3. Run create_dataset.py

## III. Network 
AIR.py is used to train and test the network. 

### 1. Train your own network
After creating the dataset you can train the network by pick case = 'train' in AIR.py and specifying the path of the training dataset in save_path. 

### 2. Reproduce the examples of the paper
We provide the checkpoint and the data for three different scenarios of the paper at the DOI address: *DOIwaitingforapproval*. 

1. In AIR.py, pick case = 'A' for the result of IV. B. Simulated Data: (1 + 1) Parallel Polarization Geometry. (Figure 6)
2. In AIR.py, pick case = 'B' for the result of IV. C. Experimental Data: (2 + 1) REMPI of α-Pinene. (Figure 8)
3. In AIR.py, pick case = 'B' for the result of IV. D. Simulated Data: (1 + 1) Orthogonal Polarization Geometry (Figure 12)	
The results are saved respectively in Figure_6_prediction.mat, Figure_8_prediction.mat, Figure_12_prediction.mat. 

### 3. Plot the results
WMIisosurf.m is used to plot the results. For the figures of the paper, we used a contrast cont of 1 and the shape 'half'.
