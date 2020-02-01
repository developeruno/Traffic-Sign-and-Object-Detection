# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 20:14:37 2019

@author: mayan
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from frcnn_test_vgg import *
import pickle,os
from keras.layers import Input
import numpy as np
import cv2, os
from matplotlib import pyplot as plt
from threading import Thread
from queue import Queue
import tensorflow as tf
from keras.utils import plot_model
## In[]:
base_path = "./"

output_weight_path = os.path.join(base_path, 'model/model_frcnn_vgg.hdf5')

record_path = os.path.join(base_path, 'model/record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)

base_weight_path = os.path.join(base_path, './vgg16_weights_tf_dim_ordering_tf_kernels.h5')

config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')


with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)
    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    
def getModel():
    num_features = 512
    
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)
    
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)
    
    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = vgg_16_model(img_input, trainable=True)
    
    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)
    
    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))
    
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    
    model_classifier = Model([feature_map_input, roi_input], classifier)
    C.model_path = "./model/model_frcnn_vgg-pre1.hdf5"
    if os.path.exists(C.model_path):
        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)
    
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    
    #plot_model(model_rpn,to_file='./model/model_rpn.png')
    #plot_model(model_classifier,to_file='./model/model_classifier.png')
    return model_rpn,model_classifier_only

model_rpn,model_classifier = getModel()   
# In[]:
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

## In[]:
graph = tf.get_default_graph()
bbox_threshold = 0.88

def get_rpn_Predictions(R,F_input,start=0,end=None):
        ROIes = np.zeros(shape=(end-start-1,4,4),dtype='float64')
        F_inputs = np.zeros(shape=(end-start-1,18,27,512),dtype='float64')
        for jk in range(start,end):
                ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break
                if jk == R.shape[0]//C.num_rois:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded
                F_inputs[jk-start,:,:,:] = F_input
                ROIes[jk,:,:] = ROIs
                
        bboxes,probs = {},{}
        with graph.as_default():
            [predictions,regressions] = model_classifier.predict_on_batch([F_inputs, ROIes])
        # Calculate bboxes coordinates on resized image
        for jk,(P_cls, P_regr) in enumerate(zip(predictions,regressions)):  
            for ii in range(P_cls.shape[0]):
                # Ignore 'bg' class
                if np.max(P_cls[ii, :]) < bbox_threshold or np.argmax(P_cls[ii, :]) == (P_cls.shape[1] - 1):
                    continue
                
                cls_name = class_mapping[np.argmax(P_cls[ii, :])]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
    
                (x, y, w, h) = ROIes[jk, ii, :]
    
                cls_num = np.argmax(P_cls[ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[ii, :]))
                
        return bboxes,probs
    
test_array = None
def predict(img):
    X, ratio = format_img(img, C)
    X = np.transpose(X, (0, 2, 3, 1))
    [Y1, Y2, F] = model_rpn.predict(X)

    R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.8)
    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    #st = time.time()

    global bboxes,probs
    bboxes,probs = {},{}
    all_dets = []
    total = R.shape[0]//C.num_rois + 1
    st = time.time()
    
    bboxes,probs = get_rpn_Predictions(R,F.copy(),end= total)
    
    for key in bboxes:
        bbox = np.array(bboxes[key])    
        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]
    
            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            
            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)
    
            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))
    
            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)
    
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    end = time.time() - st
    print("Total time taken:",end)

    return img

def predict_thread(stop):
    while not stop():
        if len(inp) == 0:
            continue
        img = inp[-1].copy()
        img = predict(img)
        out.clear()
        out.append(img)
        inp.clear()
## In[]    


img_path = os.path.join(base_path,'test/00a1a0eddc2f0200.jpg')
img = cv2.imread(img_path)
img = predict(img)

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
plt.show()
# In[]:

capture = cv2.VideoCapture("rtsp://admin:legomindstromSEV@192.168.0.100:554/Streaming/Channels/201/")
#capture = cv2.VideoCapture("http://192.168.1.2:8080/video")
inp = []
out = []
stop_thread = False
thread = Thread(target=predict_thread,name="Model_Predict_Thread",args=(lambda: stop_thread,))
thread.start()
while True:
    ret, frame = capture.read()
    if frame is not None and thread.is_alive():
        frame = cv2.resize(frame,(620,480))
        #frame = predict(frame)
        inp.append(frame.copy())
        if len(out) > 0:
            frame = out[-1]
            cv2.imshow('Window title', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_thread = True
                break

    else:
        print("Frame Not found")

capture.release()
cv2.destroyAllWindows()
thread._stop()