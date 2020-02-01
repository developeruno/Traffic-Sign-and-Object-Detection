# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:37:46 2020

@author: bhavesh
"""


from frcnn_test_vgg import predict, load_models
import cv2

_,_,model_classifier_only = load_models()
img = cv2.imread('./test/000bd0b4fa27644c.jpg')
predict(img,model_rpn,model_classifier,model_classifier_only)
