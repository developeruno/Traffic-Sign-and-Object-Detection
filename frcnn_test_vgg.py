
# coding: utf-8

# In[3]:
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from frcnn_train_vgg import *
from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

from rcnn_utils import *
import cv2
# #### Config setting
# In[ ]:


def get_data(input_path):
	found_bg = False
	all_imgs = {}

	classes_count = {}

	class_mapping = {}

	visualise = True

	i = 1
	
	with open(input_path,'r') as f:

		print('Parsing annotation files')

		for line in f:

			# Print process
			sys.stdout.write('\r'+'idx=' + str(i))
			i += 1

			line_split = line.strip().split(',')

			# Make sure the info saved in annotation file matching the format (path_filename, x1, y1, x2, y2, class_name)
			# Note:
			#	One path_filename might has several classes (class_name)
			#	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
			#	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates
			#   x1,y1-------------------
			#	|						|
			#	|						|
			#	|						|
			#	|						|
			#	---------------------x2,y2

			(filename,x1,y1,x2,y2,class_name) = line_split

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}
				
				img = cv2.imread(filename)
				(rows,cols) = img.shape[:2]
				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['bboxes'] = []
				# if np.random.randint(0,6) > 0:
				# 	all_imgs[filename]['imageset'] = 'trainval'
				# else:
				# 	all_imgs[filename]['imageset'] = 'test'

			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])
		
		# make sure the bg class is last in the list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
		
		return all_data, classes_count, class_mapping

# #### Get new image size and augment the image
# #### Generate the ground_truth anchors
# In[ ]:

if __name__ == "main":
    """
    base_path = 'drive/My Drive/AI/Faster_RCNN'
    
    test_path = 'drive/My Drive/AI/Dataset/Open Images Dataset v4 (Bounding Boxes)/person_car_phone_test_annotation.txt' # Test data (annotation file)
    
    test_base_path = 'drive/My Drive/AI/Dataset/Open Images Dataset v4 (Bounding Boxes)/test' # Directory to save the test images
    """ 
    
    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)
        C.record_path = "./model/record.csv"
    
    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
        
    # Load the records
    record_df = pd.read_csv(C.record_path)
    
    r_epochs = len(record_df)
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
    plt.title('class_acc')
    
    plt.show()
    
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    plt.show()
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    plt.show()
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
    plt.title('elapsed_time')
    
    plt.show()


# In[20]:

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)
    C.record_path = "./mdoel/record.csv"
C.model_path = "./model/model_frcnn_vgg-keras.hdf5"
C.class_mapping = {'Traffic light': 0, 'Car': 1, 'Traffic sign': 2, 'Person': 3, 'bg': 4}


class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

def load_models():
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
    
    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)
    
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    
    return model_rpn,model_classifier,model_classifier_only       
    
    # In[ ]:
    

test_imgs = os.listdir('./test/')

imgs_path = []
for i in range(12):
	idx = np.random.randint(len(test_imgs))
	imgs_path.append(test_imgs[idx])

all_imgs = []

classes = {}


# In[]:


def predict(img,model_rpn,model_classifier,model_classifier_only,bbox_threshold = 0.1):
        X, ratio = format_img(img, C)
        
        X = np.transpose(X, (0, 2, 3, 1))
    
        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = model_rpn.predict(X)
    
        # Get bboxes by applying NMS 
        # R.shape = (300, 4)
        R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)
    
        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
    
        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
    
        for jk in range(R.shape[0]//C.num_rois + 1):
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
    
            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
    
            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue
    
                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
    
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
    
                (x, y, w, h) = ROIs[0, ii, :]
    
                cls_num = np.argmax(P_cls[0, ii, :])
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
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
    
        all_dets = []
    
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
                textOrg = (real_x1 + 10, real_y1 + 10)
    
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                
                im = img[real_x1:real_x2,real_y1:real_y2]
                #cv2.imshow(textLabel,im)
                
        cv2.imshow("image",img)
        cv2.waitKey(0)
# In[]:
if __name__ == "__main__":
    model_rpn,model_classifier,model_classifier_only = load_models()
                
    # In[37]:
from skimage import io

# In[]:

img = cv2.imread('./train/0d04b6af5aa3d210.jpg')
predict(img,model_rpn,model_classifier,model_classifier_only,0.81)
# In[]:

img = io.imread('test/000bd0b4fa27644c.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
predict(img,model_rpn,model_classifier,model_classifier_only,0.7)

# In[]:

img = io.imread('train/3c59b139b80475f0.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
predict(img,model_rpn,model_classifier,model_classifier_only,0.7)

# In[]:

img = io.imread('train/4b3559ff61663719.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
predict(img,model_rpn,model_classifier,model_classifier_only,0.7)

# In[]:

img = io.imread('https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRBFcfEteZ0Fijni-03mOO9bSdTA9dJ7ojjy49XWr3_S4L_Lt_x')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
predict(img,model_rpn,model_classifier,model_classifier_only,0.8)

# In[]:

img = io.imread('https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTF5ylK1TWWRkOouGpvMnk6H0HGrCMrJ-PRX7S_tHyHWQfTep0m')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
predict(img,model_rpn,model_classifier,model_classifier_only,0.7)


# In[]:
    # If the box classification value is less than this, we ignore this box
    bbox_threshold = 0.7
    
    for idx, img_name in enumerate(imgs_path):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        st = time.time()
        filepath = os.path.join('./test', img_name)
    
        predict(img_name)
    
        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        plt.figure(figsize=(10,10))
        plt.grid()
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()


# #### Measure mAP

# In[ ]:


def get_map(pred, gt, f):
	T = {}
	P = {}
	fx, fy = f

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:
		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_y1 = pred_box['y1']
		pred_y2 = pred_box['y2']
		pred_prob = pred_box['prob']
		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []
		P[pred_class].append(pred_prob)
		found_match = False

		for gt_box in gt:
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']/fx
			gt_x2 = gt_box['x2']/fx
			gt_y1 = gt_box['y1']/fy
			gt_y2 = gt_box['y2']/fy
			gt_seen = gt_box['bbox_matched']
			if gt_class != pred_class:
				continue
			if gt_seen:
				continue
			iou_map = iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
			if iou_map >= 0.5:
				found_match = True
				gt_box['bbox_matched'] = True
				break
			else:
				continue

		T[pred_class].append(int(found_match))

	for gt_box in gt:
		if not gt_box['bbox_matched']:# and not gt_box['difficult']:
			if gt_box['class'] not in P:
				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	#import pdb
	#pdb.set_trace()
	return T, P


# In[ ]:


def format_img_map(img, C):
	"""Format image for mAP. Resize original image to C.im_size (300 in here)

	Args:
		img: cv2 image
		C: config

	Returns:
		img: Scaled and normalized image with expanding dimension
		fx: ratio for width scaling
		fy: ratio for height scaling
	"""

	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	fx = width/float(new_width)
	fy = height/float(new_height)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	# Change image channel from BGR to RGB
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	# Change img shape from (height, width, channel) to (channel, height, width)
	img = np.transpose(img, (2, 0, 1))
	# Expand one dimension at axis 0
	# img shape becames (1, channel, height, width)
	img = np.expand_dims(img, axis=0)
	return img, fx, fy


# In[ ]:

if __name__ == "main":
    print(class_mapping)
    
    # This might takes a while to parser the data
    test_imgs, _, _ = get_data(test_path)
    
    T = {}
    P = {}
    mAPs = []
    for idx, img_data in enumerate(test_imgs):
        print('{}/{}'.format(idx,len(test_imgs)))
        st = time.time()
        filepath = img_data['filepath']
    
        img = cv2.imread(filepath)
    
        X, fx, fy = format_img_map(img, C)
    
        # Change X (img) shape from (1, channel, height, width) to (1, height, width, channel)
        X = np.transpose(X, (0, 2, 3, 1))
    
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)
    
    
        R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    
        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
    
        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
    
        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
    
            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
    
            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
    
            # Calculate all classes' bboxes coordinates on resized image (300, 400)
            # Drop 'bg' classes bboxes
            for ii in range(P_cls.shape[1]):
    
                # If class name is 'bg', continue
                if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue
    
                # Get class name
                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
    
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
    
                (x, y, w, h) = ROIs[0, ii, :]
    
                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
    
        all_dets = []
    
        for key in bboxes:
            bbox = np.array(bboxes[key])
    
            # Apply non-max-suppression on final bboxes to get the output bounding boxe
            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                all_dets.append(det)
    
    
        print('Elapsed time = {}'.format(time.time() - st))
        t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key])
            P[key].extend(p[key])
        all_aps = []
        for key in T.keys():
            ap = average_precision_score(T[key], P[key])
            print('{} AP: {}'.format(key, ap))
            all_aps.append(ap)
        print('mAP = {}'.format(np.mean(np.array(all_aps))))
        mAPs.append(np.mean(np.array(all_aps)))
        #print(T)
        #print(P)
        
    print()
    print('mean average precision:', np.mean(np.array(mAPs)))
        
    mAP = [mAP for mAP in mAPs if str(mAP)!='nan']
    mean_average_prec = round(np.mean(np.array(mAP)), 3)
    print('After training %dk batches, the mean average precision is %0.3f'%(len(record_df), mean_average_prec))
    
    # record_df.loc[len(record_df)-1, 'mAP'] = mean_average_prec
    # record_df.to_csv(C.record_path, index=0)
    # print('Save mAP to {}'.format(C.record_path))
    
