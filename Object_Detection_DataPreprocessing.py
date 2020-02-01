
# coding: utf-8

# In[1]:


import numpy as np
import time
import sys
import os
import random
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt
from shutil import copyfile

import cv2
import tensorflow as tf


# ### Load data from .csv file
# 
# * `train-images-boxable.csv` file contains the image name and image url
# * `train-annotations-bbox.csv` file contains the bounding box info with the image id (name) and the image label name
# * `class-descriptions-boxable.csv` file contains the image label name corresponding to its class name
# 
# Download link:
# 
# https://storage.googleapis.com/openimages/web/download.html
# 
# https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/

# In[2]:


base_path = './'
images_boxable_fname = 'train-images-boxable.csv'
annotations_bbox_fname = 'train-annotations-bbox.csv'
class_descriptions_fname = 'class-descriptions-boxable.csv'


# In[3]:


images_boxable = pd.read_csv(os.path.join(base_path, images_boxable_fname))
images_boxable.head()


# In[4]:


annotations_bbox = pd.read_csv(os.path.join(base_path, annotations_bbox_fname))
annotations_bbox.head()


# In[13]:


class_descriptions = pd.read_csv(os.path.join(base_path, class_descriptions_fname),names=["name","class"])
class_descriptions.head()


# ### Show one image by using these three tables

# In[14]:


print('length of the images_boxable: %d' %(len(images_boxable)) )
print('First image in images_boxable👇')
img_name = images_boxable['image_name'][0]
img_url = images_boxable['image_url'][0]
print('\t image_name: %s' % (img_name))
print('\t img_url: %s' % (img_url))
print('')
print('length of the annotations_bbox: %d' %(len(annotations_bbox)))
print('The number of bounding boxes are larger than number of images.')
print('')
print('length of the class_descriptions: %d' % (len(class_descriptions)-1))
img = io.imread(img_url)


# In[15]:


height, width, _ = img.shape
print(img.shape)
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(img)
img_id = img_name[:16]
bboxs = annotations_bbox[annotations_bbox['ImageID']==img_id]
img_bbox = img.copy()
for index, row in bboxs.iterrows():
    xmin = row['XMin']
    xmax = row['XMax']
    ymin = row['YMin']
    ymax = row['YMax']
    xmin = int(xmin*width)
    xmax = int(xmax*width)
    ymin = int(ymin*height)
    ymax = int(ymax*height)
    label_name = row['LabelName']
    class_series = class_descriptions[class_descriptions['name']==label_name]
    class_name = class_series['class'].values[0]
    cv2.rectangle(img_bbox,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bbox,class_name,(xmin,ymin-10), font, 1,(0,255,0),2)
plt.subplot(1,2,2)
plt.title('Image with Bounding Box')
plt.imshow(img_bbox)
plt.show()


# In[ ]:


# io.imsave(os.path.join(base_path,'Person')+'/1.jpg', img)


# As we can see, by using these three tables, the image with bounding box could be drawn

# ### Get subset of the whole dataset
# 
# For here, I just want to detect three classes, which include person, mobile phone and car.
# 
# The dataset from [Open Images Dataset V4](https://storage.googleapis.com/openimages/web/download.html) is too large for me. So I just extract 1000 images for each class from the whole dataset.

# In[16]:
classes = ["Person","Car","Traffic light", "Traffic sign"]
pds = dict()
labels = dict()
bboxes = dict()
imgs_ids = dict()


for class_ in classes:
    pds[class_] = class_descriptions[class_descriptions['class'] == class_]
    labels[class_] = pds[class_]['name'].values[0]
    bboxes[class_] = annotations_bbox[annotations_bbox['LabelName']==labels[class_]]
    imgs_ids[class_] = np.unique(bboxes[class_]['ImageID'])
    print("There are %d %s in the dataset and unique are %d" % (len(bboxes[class_]),class_,len(imgs_ids[class_])))



# We just randomly pick 1000 images in here.

# In[21]:


# Shuffle the ids and pick the first 1000 ids
copy_ids = dict()
sub_imgs = dict()
sub_imgs_url = dict()

n = 1000

for name,ids in imgs_ids.items():
    copy_ids[name] = ids.copy()
    random.seed(1)
    random.shuffle(copy_ids[name])
    
    sub_imgs[name] = copy_ids[name][:n]
    sub_imgs_url[name] = [images_boxable[images_boxable['image_name']==img_id+'.jpg'] for img_id in sub_imgs[name]]
    
    print(sub_imgs[name][:10])

# In[24]:
sub_imgs_pd = {}

sub_imgs_pd[classes[0]] = pd.read_csv("sub-Person_img_url.csv");
sub_imgs_pd[classes[1]] = pd.read_csv("sub-Car_img_url.csv")
sub_imgs_pd[classes[2]] = pd.read_csv("sub-Traffic light_img_url.csv")

for name in classes:
    if name not in sub_imgs_pd:
        sub_imgs_pd[name] = pd.DataFrame()
    
    for i in range(len(sub_imgs_url[name])):
        sub_imgs_pd[name] = sub_imgs_pd[name].append(sub_imgs_url[name][i], ignore_index = True)
        
    sub_imgs_pd[name].to_csv(os.path.join(base_path, 'sub-%s_img_url.csv' % name))


# In[25]:
urls = dict()
for key,value in sub_imgs_url.items():
    urls[key] = [url['image_url'].values[0] for url in value]

# In[31]:


saved_dirs = {key:os.path.join(base_path,'data', key) for key in classes}

# ### Download images

# In[36]:
import os

# Download images
for class_name in classes:
    saved_dir = saved_dirs[class_name]
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    for url in urls[class_name]:       
        saved_path = os.path.join(saved_dir, url[-20:])
        if not os.path.exists(saved_path):
            img = io.imread(url)
            io.imsave(saved_path, img)

# ### Prepare dataset format for faster rcnn code
# 
# (fname_path, xmin, xmax, ymin, ymax, class_name)
# 
# train: 0.8
# validation: 0.2

# In[40]:


# Save images to train and test directory
train_path = os.path.join(base_path, 'train')
#os.mkdir(train_path)
test_path = os.path.join(base_path, 'test')
#os.mkdir(test_path)

for i in range(len(classes)):
    
    all_imgs = os.listdir(os.path.join(base_path + "data/", classes[i]))
    all_imgs = [f for f in all_imgs if not f.startswith('.')]
    random.seed(1)
    random.shuffle(all_imgs)
    
    train_imgs = all_imgs[:800]
    test_imgs = all_imgs[800:]
    
    # Copy each classes' images to train directory
    for j in range(len(train_imgs)):
        original_path = os.path.join(os.path.join(base_path + "data/", classes[i]), train_imgs[j])
        new_path = os.path.join(train_path, train_imgs[j])
        copyfile(original_path, new_path)
    
    # Copy each classes' images to test directory
    for j in range(len(test_imgs)):
        original_path = os.path.join(os.path.join(base_path+ "data/", classes[i]), test_imgs[j])
        new_path = os.path.join(test_path, test_imgs[j])
        copyfile(original_path, new_path)


# In[41]:


print('number of training images: ', len(os.listdir(train_path))) # subtract one because there is one hidden file named '.DS_Store'
print('number of test images: ', len(os.listdir(test_path)))


# The expected number of training images and validation images should be 3x800 -> 2400 and 3x200 -> 600.
# 
# However, there might be some overlap images which appear in two or three classes simultaneously. For instance, an image might be a person walking on the street and there are several cars in the street

# In[42]:


train_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

# Find boxes in each image and put them in a dataframe
train_imgs = os.listdir(train_path)
train_imgs = [name for name in train_imgs if not name.startswith('.')]

for i in range(len(train_imgs)):
    sys.stdout.write('Parse train_imgs ' + str(i) + '; Number of boxes: ' + str(len(train_df)) + '\r\n')
    sys.stdout.flush()
    img_name = train_imgs[i]
    img_id = img_name[0:16]
    tmp_df = annotations_bbox[annotations_bbox['ImageID']==img_id]
    for index, row in tmp_df.iterrows():
        labelName = row['LabelName']
        for class_name,label in labels.items():
            if label == labelName:
                train_df = train_df.append({'FileName': img_name, 
                                            'XMin': row['XMin'], 
                                            'XMax': row['XMax'], 
                                            'YMin': row['YMin'], 
                                            'YMax': row['YMax'], 
                                            'ClassName': class_name}, 
                                           ignore_index=True)


# In[44]:


test_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

# Find boxes in each image and put them in a dataframe
test_imgs = os.listdir(test_path)
test_imgs = [name for name in test_imgs if not name.startswith('.')]

for i in range(len(test_imgs)):
    sys.stdout.write('Parse test_imgs ' + str(i) + '; Number of boxes: ' + str(len(test_df)) + '\r')
    sys.stdout.flush()
    img_name = test_imgs[i]
    img_id = img_name[0:16]
    tmp_df = annotations_bbox[annotations_bbox['ImageID']==img_id]
    for index, row in tmp_df.iterrows():
        labelName = row['LabelName']
        for class_name,label in labels.items():
            if label == labelName:
                test_df = test_df.append({'FileName': img_name, 
                                            'XMin': row['XMin'], 
                                            'XMax': row['XMax'], 
                                            'YMin': row['YMin'], 
                                            'YMax': row['YMax'], 
                                            'ClassName': class_name}, 
                                           ignore_index=True)


# In[45]:


train_df.to_csv(os.path.join(base_path, 'train.csv'))
test_df.to_csv(os.path.join(base_path, 'test.csv'))


# ### Write train.csv to annotation.txt

# print(train_df.head())
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))

# For training
f= open(base_path + "/annotation.txt","w+")
for idx, row in train_df.iterrows():
#     sys.stdout.write(str(idx) + '\r')
#     sys.stdout.flush()
    img = cv2.imread((base_path + '/train/' + row['FileName']))
    height, width = img.shape[:2]
    x1 = int(row['XMin'] * width)
    x2 = int(row['XMax'] * width)
    y1 = int(row['YMin'] * height)
    y2 = int(row['YMax'] * height)

    fileName = os.path.join('train', row['FileName'])
    className = row['ClassName']
    f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
f.close()


# In[ ]:


print(test_df.head())
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

# For test
f= open(base_path + "/test_annotation.txt","w+")
for idx, row in test_df.iterrows():
    sys.stdout.write(str(idx) + '\r')
    sys.stdout.flush()
    img = cv2.imread((base_path + '/test/' + row['FileName']))
    if img is not None:
        height, width = img.shape[:2]
        x1 = int(row['XMin'] * width)
        x2 = int(row['XMax'] * width)
        y1 = int(row['YMin'] * height)
        y2 = int(row['YMax'] * height)
        
        google_colab_file_path = 'drive/My Drive/AI/Dataset/Open Images Dataset v4 (Bounding Boxes)/test'
        fileName = os.path.join(google_colab_file_path, row['FileName'])
        className = row['ClassName']
        f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
f.close()

