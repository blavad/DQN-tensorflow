from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.smoothgrad import SmoothGrad
from tf_explain.core.activations import ExtractActivations
from tf_explain.core.grad_cam import GradCAM

import cv2
import math
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def gradCAM(X, model, layer_name, class_index):
    explainerGradCam = GradCAM()
    outputs =  explainerGradCam.explain((X,None), model, layer_name, class_index)
    return cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)

def extractActivations(X, model, layer_name):
    explainerActiv = ExtractActivations()
    outputs =  explainerActiv.explain((X,None), model, layer_name)
    return outputs

def smoothGrad(X, model, class_index):
    explainerSmoothGrad = SmoothGrad()
    outputs = explainerSmoothGrad.explain((X, None), model, class_index)
    return outputs


def occlSensitivity(X, model, class_index, patch_size):
    explainerOccl = OcclusionSensitivity()
    outputs = explainerOccl.explain((X,None), model, class_index, patch_size)
    return cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)

def group_inputs(X):
    num_rows = math.ceil(math.sqrt(len(X)))
    num_columns = math.ceil(math.sqrt(len(X)))
    number_of_missing_elements = num_columns * num_rows - len(X)
    X = np.append(
        X,
        np.zeros((number_of_missing_elements, *X[0].shape)).astype(X.dtype),
        axis=0,
    )
    grid = np.concatenate(
        [
            np.concatenate(
                X[index * num_columns : (index + 1) * num_columns], axis=1
            )
            for index in range(num_rows)
        ],
        axis=0,
    )
    grid = 255*(grid+1)/2
    grid_3c = cv2.merge([grid, grid, grid])
    
    return grid_3c.astype(int)

def filter_each(X, filter, filter_kargs):
    res = []
    for img in X:
        im_filtered = filter(np.array([img]), **filter_kargs)
        res.append(im_filtered)
    return np.array(res)

def show(grid):
    if grid.shape[-1]==1:
        grid.shape = (grid.shape[0],grid.shape[1])
        plt.imshow(grid, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(grid)        
    plt.show()

def save(grid, output_dir, output_name):
    Path.mkdir(Path(output_dir), parents=True, exist_ok=True)
    cv2.imwrite(
        str(Path(output_dir) / output_name), grid
    )
    
def make_video(dir_save, save_name, X, fps=30):
    abs_path = os.path.join(dir_save,save_name)
    print("Create video {}.{}".format(save_name,"avi"))
    height = X[0].shape[0]
    width = X[0].shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('{}.avi'.format(abs_path), fourcc, float(fps), (width, height))
    for img in X:
        frame = np.random.randint(0, 256, 
                                (height, width, 3), 
                                dtype=np.uint8)
        if len(img.shape)<3 or img.shape[-1]==1:
            img = cv2.merge([img, img, img])
        frame[:,:,:] = img
        video.write(frame)
    video.release()
    print("Video Created")

def save_all_explainer(validation_data, model, name_conv, n_conv=1, dir_save_im='./', save_name='outputs'):
    explainerGradCam = GradCAM()
    explainerActiv = ExtractActivations()
    explainerOccl = OcclusionSensitivity()
    explainerSmoothGrad = SmoothGrad()

    for i in range(1, n_conv+1):
        output = explainerActiv.explain(
            validation_data, model, '{}_{}'.format(name_conv, i))
        explainerActiv.save(output, dir_save_im,
                            '{}-activ-conv{}.jpg'.format(save_name, i))

        output = explainerGradCam.explain(
            validation_data, model, '{}_{}'.format(name_conv, i), 0)
        explainerGradCam.save(output, dir_save_im,
                              '{}-gradCam0-conv{}.jpg'.format(save_name, i))

    output = explainerSmoothGrad.explain(validation_data, model, 0)
    explainerSmoothGrad.save(
        output, dir_save_im, '{}-smooth0.jpg'.format(save_name))
    
    output = explainerSmoothGrad.explain(validation_data, model, 1)
    explainerSmoothGrad.save(
        output, dir_save_im, '{}-smooth1.jpg'.format(save_name))

    output = explainerOccl.explain(validation_data, model, 0, 5)
    explainerOccl.save(output, dir_save_im,
                       '{}-occlSens0.jpg'.format(save_name))
    output = explainerOccl.explain(validation_data, model, 1, 5)
    explainerOccl.save(output, dir_save_im,
                       '{}-occlSens1.jpg'.format(save_name))