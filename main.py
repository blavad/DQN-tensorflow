##%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

import numpy as np
import cv2 
import random
import math

from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.activations import ExtractActivations
from tf_explain.core.smoothgrad import SmoothGrad
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

from tensorflow import keras
from tensorflow.keras import layers, optimizers, models

from flappyBird_cnn import FlappyBirdCnnEnv
from flappyBird_env import FlappyBirdEnv

import tf_tools
import data
import data_visualisation as dv

tf.keras.backend.clear_session()

print('Version TensorFlow:',tf.__version__)
print('Version OpenCV:', cv2.__version__)

def get_model(name):
    inputs = keras.Input(shape=(84, 84, 1), name='{}_inputs'.format(name))
    conv1 = layers.Conv2D(filters=32, kernel_size=[8, 8], strides=(
        4, 4), padding="same", name='{}_conv2d_1'.format(name), activation=tf.nn.relu)(inputs)
    conv2 = layers.Conv2D(filters=64, kernel_size=[4, 4], strides=(
        2, 2), padding="same", name='{}_conv2d_2'.format(name), activation=tf.nn.relu)(conv1)
    conv3 =layers. Conv2D(filters=64, kernel_size=[3, 3], strides=(
        1, 1), padding="same", name='{}_conv2d_3'.format(name), activation=tf.nn.relu)(conv2)
    flat = layers.Flatten()(conv3)
    fc1 = layers.Dense(512, name='{}_dense'.format(name), activation=tf.nn.relu)(flat)
    fc_out = layers.Dense(2, name='{}_dense_out'.format(name))(fc1)
    model = models.Model(inputs=inputs, outputs=fc_out)
    model.compile(optimizer="sgd", loss="categorical_crossentropy")
    return model     

##%%
def main():
    ckpt_name = "./models/FlappyFullImage/full-image3-6000000"
    save_name = 't2-6000'  
    dir_save_im = './img_plot/tf_explain/Flappy_Test_2'  
    model_args = {'name':'q-net'}
            
    tf.executing_eagerly()
    
    model = tf_tools.init_keras_model_from_ckpt(ckpt_name, get_model, model_args)
    print(model.summary())
    
    x, y, _ = data.getFlappyData(n_data=1)    
    
    X, y, _ = data.getDataEpisode(model, FlappyBirdCnnEnv())
    inputs = dv.filter_each(X, dv.group_inputs, {})
    
    """ 
    filter_args = {"model":model, "layer_name": 'q-net_conv2d_1',"class_index": 0 }    
    outGrad = dv.filter_each(X, dv.gradCAM, filter_args)
    dv.make_video(dir_save_im,"gradCam_{}".format(save_name), outGrad)

    filter_args = {"model":model, "layer_name": 'q-net_conv2d_1'}    
    outExtract = dv.filter_each(X, dv.extractActivations, filter_args)
    dv.make_video(dir_save_im,"extractActiv_{}".format(save_name), outExtract)

    filter_args = {"model":model, "patch_size": 5,"class_index": 0 }    
    occl = dv.filter_each(X, dv.occlSensitivity, filter_args)
    dv.make_video(dir_save_im,"occlSens0_{}".format(save_name), occl)
    
    filter_args = {"model":model, "patch_size": 5,"class_index": 1 }    
    occl = dv.filter_each(X, dv.occlSensitivity, filter_args)
    dv.make_video(dir_save_im,"occlSens1_{}".format(save_name), occl)

    filter_args = {"model":model, "class_index": 0 }    
    outSmooth = dv.filter_each(X, dv.smoothGrad, filter_args)
    dv.make_video(dir_save_im,"smooth_{}".format(save_name), outSmooth)

    outStd = dv.filter_each(X, dv.group_inputs, {})
    dv.make_video(dir_save_im,"inputs_{}".format(save_name), outStd) """
    
    
    #inputs_grp = dv.group_inputs(X)
    for step in {90,250}:
        dv.save(inputs[step], dir_save_im, "{}-inputs-{}.jpg".format(save_name, step))
        outGradCAM = dv.occlSensitivity(np.array([X[step]]), model, 0, 5)
        dv.save(outGradCAM, dir_save_im, "{}-occl0-step{}.jpg".format(save_name, step))
        outGradCAM = dv.occlSensitivity(np.array([X[step]]), model, 1, 5)
        dv.save(outGradCAM, dir_save_im, "{}-occl1-step{}.jpg".format(save_name, step))
        for num_conv in range(1,4):
            outGradCAM = dv.gradCAM(np.array([X[step]]), model, 'q-net_conv2d_{}'.format(num_conv), 0)
            dv.save(outGradCAM, dir_save_im, "{}-gradCAM0-{}-conv{}.jpg".format(save_name, step, num_conv))
            outGradCAM = dv.gradCAM(np.array([X[step]]), model, 'q-net_conv2d_{}'.format(num_conv), 1)
            dv.save(outGradCAM, dir_save_im, "{}-gradCAM1-{}-conv{}.jpg".format(save_name, step, num_conv))
            
            outGradCAM = dv.extractActivations(np.array([X[step]]), model, 'q-net_conv2d_{}'.format(num_conv))
            dv.save(outGradCAM, dir_save_im, "{}-activ-{}-conv{}.jpg".format(save_name, step, num_conv))
                
    # dv.show(inputs_grp)
    # dv.show(outGradCAM)
    
main()
