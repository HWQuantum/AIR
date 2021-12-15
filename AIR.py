import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import tensorflow as tf
import scipy.io as sio
from tensorflow import keras
tf.keras.backend.clear_session()
import numpy as np
import time

from tensorflow.python.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, concatenate,Flatten,Reshape,Cropping2D, Conv3D,Conv3DTranspose,MaxPool3D,Cropping3D, Reshape,Lambda
from tensorflow.python.keras.models import Model



 
Directory = './'
case = 'B' # Pick between 'train', 'A' (IV. B Parallel polarization geometry), 'B' (IV. C REMPI alpha pinene) and 'C' (IV. D Orthogonal polarization geometry). 

#------------------------------------------------------------------------------------------------------------------------
# Load the MODELS
#------------------------------------------------------------------------------------------------------------------------

if case == 'train':
    save_path = '' # TO FILL (path of create_dataset.py)
    is_train = 1 
    name = 'training'
    checkpoint_path = os.path.join(Directory, 'Checkpoint_'+name+'/cp.ckpt')
    
elif case == 'A':
    is_train = 0
    name = 'parallel_polarization_geometry'
    checkpoint_path = os.path.join(Directory, 'Checkpoint_'+name+'/cp.ckpt')

elif case == 'B':
    is_train = 0
    name = 'REMPI_alpha_pinene'
    checkpoint_path = os.path.join(Directory, 'Checkpoint_'+name+'/cp.ckpt')

elif case == 'C':
    is_train = 0
    name = 'orthogonal_polarization_geometry'
    checkpoint_path = os.path.join(Directory, 'Checkpoint_'+name+'/cp.ckpt')


#------------------------------------------------------------------------------------------------------------------------
# Load the training dataset - For the Training 
#------------------------------------------------------------------------------------------------------------------------
if is_train:
    for i in range(7):
        print(i)
        if i == 0:
            X = sio.loadmat(os.path.join(save_path, 'Image_Inflation_dataset'+str(i)+'_I_2Dproj.mat'))['I_2D_proj']
            X_inv = sio.loadmat(os.path.join(save_path, 'Image_Inflation_dataset'+str(i)+'_I_3D.mat'))['I_3D']
        else:
            mat_2D = sio.loadmat(os.path.join(save_path, 'Image_Inflation_dataset'+str(i)+'_I_2Dproj.mat'))['I_2D_proj']
            mat_3D = sio.loadmat(os.path.join(save_path, 'Image_Inflation_dataset'+str(i)+'_I_3D.mat'))['I_3D']
            X = np.concatenate((X,mat_2D), axis=-1)
            X_inv = np.concatenate((X_inv,mat_3D), axis=-1)

    X = np.swapaxes(X, 0, -1)
    X = np.swapaxes(X, 1, 2)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

    X_inv = np.swapaxes(X_inv, 0, -1)
    X_inv = np.swapaxes(X_inv, 1, 2)
    X_inv = np.reshape(X_inv, (X_inv.shape[0], X_inv.shape[1], X_inv.shape[2], X_inv.shape[3]))


    X = X[:10000]
    X_inv = X_inv[:10000]
    X_training = X[100:]
    X_training_inv = X_inv[100:]
    X_validation = X[:100]
    X_validation_inv = X_inv[:100]
    print(X_training.shape)
    print(X_validation.shape)
    print(X_training_inv.shape)
    print(X_validation_inv.shape)
#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------


def create_model():
    UB = True
    w = 7
    inputs = Input((80,80,1))
    conv1 = Conv2D(64, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(inputs)
    conv12 = Conv2D(64, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(conv1)
    pool1 = MaxPool2D((2,2), padding='same')(conv12) 

    conv2 = Conv2D(128, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool1)
    conv21 = Conv2D(128, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(conv2)
    pool2 = MaxPool2D((2,2), padding='same')(conv21) 

    deconv4 = Conv2DTranspose(128, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(pool2)
    conv41 = Conv2D(128, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(deconv4)
    conv42 = Conv2D(128, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(conv41)
    merge4 = concatenate([conv42,conv21], axis = 3) # 128 13 13 

    deconv3 = Conv2DTranspose(64, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge4)
    conv31 = Conv2D(64, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(deconv3)
    conv32 = Conv2D(64, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(conv31)
    
    merge3 = concatenate([conv32,conv12], axis = 3)

    conv_last1 = Conv2D(64, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(merge3)
    conv_last = Conv2D(80, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(conv_last1)

    model = Model(inputs,conv_last)

    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00010), loss = "mean_squared_error", metrics = ["accuracy"])
   
    return model

model = create_model()

model.summary()
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
if is_train:
    Nepochs = 3000
    start_time = time.time()
    history = model.fit(X_training, X_training_inv, validation_data = (X_validation, X_validation_inv), epochs = Nepochs, batch_size = 32, callbacks=[cp_callback,WandbCallback()])
    training_time = time.time()-start_time
    sio.savemat(os.path.join(Directory, 'Simulated_data', 'history_.mat'), history.history)


#------------------------------------------------------------------------------------------------------------------------
# Test & Save on validation data
#--------------------------------------------------------------------------------------------------------------------

model.load_weights(checkpoint_path)

if is_train:
    # load 5 images of the validation data
    X_val = X_validation[0:5]
    X_val_inv = X_validation_inv[0:5]
    predict_data = np.zeros((X_val.shape[1],X_val.shape[2],80,5))
    Nimages = X_val.shape[0]

    # predict for each image
    for index in range(Nimages):
        val_image = np.squeeze(X_val[index, :,:,:])
        val_image = np.reshape(val_image, (1, val_image.shape[0],val_image.shape[1],1))
        predict_image = model.predict(val_image)
        predict_data[:,:, :, index] = np.squeeze(predict_image)

    # save
    dictionary = {}
    dictionary['prediction'] = predict_data
    sio.savemat(os.path.join(Directory, 'res_validation_inflation'+name+'.mat'), dictionary)

if case == 'A':
    # load data
    path = os.path.join(Directory, 'Figure_6.mat')
    input_data = sio.loadmat(path)['I_2D']
    input_data = input_data[80:, 80:]
    input_data = np.reshape(input_data, (1, 80,80,1))

    # predict only 1/4 of the distribution
    predict_image = np.squeeze(model.predict(input_data)) 

    # recover full distribution
    flip_1 = np.flip(predict_image, axis=0)
    res_flip1 = np.concatenate((flip_1, predict_image), axis = 0)
    flip_2 = np.flip(res_flip1, axis=1)
    res_flip2 = np.concatenate((flip_2, res_flip1), axis = 1)
    flip_3 = np.flip(res_flip2, axis=2)
    predictions = np.concatenate((flip_3, res_flip2), axis = 2)

    # save
    dictionary = {}
    dictionary['predictions_AIR'] = predictions
    sio.savemat(os.path.join(Directory, 'Figure_6_prediction.mat'), dictionary)

if case == 'B':
    # load data
    path = os.path.join(Directory, 'Figure_8.mat')
    input_data = sio.loadmat(path)['I_2D']
    input_data = input_data[80:, 80:]
    input_data = np.reshape(input_data, (1, 80,80,1))

    # predict only 1/4 of the distribution
    predict_image = np.squeeze(model.predict(input_data))
    
    # recover full distribution
    flip_1 = np.flip(predict_image, axis=0)
    res_flip1 = np.concatenate((flip_1, predict_image), axis = 0)
    flip_2 = np.flip(res_flip1, axis=1)
    res_flip2 = np.concatenate((flip_2, res_flip1), axis = 1)
    flip_3 = np.flip(res_flip2, axis=2)
    predictions = np.concatenate((flip_3, res_flip2), axis = 2)

    # save
    dictionary = {}
    dictionary['input_data'] = input_data
    dictionary['predictions_AIR'] = predictions
    sio.savemat(os.path.join(Directory, 'Figure_8_prediction.mat'), dictionary)

    
if case == 'C':
    # load data
    path = os.path.join(Directory, 'Figure_12.mat')
    input_data = sio.loadmat(path)['I_2D']
    input_data = input_data[80:, 80:]
    input_data = np.reshape(input_data, (1, 80,80,1))

    # predict only 1/4 of the distribution
    predict_image = np.squeeze(model.predict(input_data))

    # recover full distribution
    flip_1 = np.flip(predict_image, axis=0)
    res_flip1 = np.concatenate((flip_1, predict_image), axis = 0)
    flip_2 = np.flip(res_flip1, axis=1)
    res_flip2 = np.concatenate((flip_2, res_flip1), axis = 1)
    flip_3 = np.flip(res_flip2, axis=2)
    predictions = np.concatenate((flip_3, res_flip2), axis = 2)

    # save
    dictionary = {}
    dictionary['input_data'] = input_data
    dictionary['predictions_AIR'] = predictions
    sio.savemat(os.path.join(Directory, 'Figure_12_prediction.mat'), dictionary)