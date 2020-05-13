#!/usr/bin/env python
# coding: utf-8

# # Assessment 1: I can train and deploy a neural network

# At this point, you've worked through a full deep learning workflow. You've loaded a dataset, trained a model, and deployed your model into a simple application. Validate your learning by attempting to replicate that workflow with a new problem.
# 
# We've included a dataset which consists of two classes:  
# 
# 1) Face: Contains images which include the face of a whale  
# 2) Not Face: Contains images which do not include the face of a whale.  
# 
# The dataset is located at ```/dli/data/whale/data/train```.
# 
# Your challenge is:
# 
# 1) Use [DIGITS](/digits) to train a model to identify *new* whale faces with an accuracy of more than 80%.   
# 
# 2) Deploy your model by modifying and saving the python application [submission.py](../../../../edit/tasks/task-assessment/task/submission.py) to return the word "whale" if the image contains a whale's face and "not whale" if the image does not.  
# 
# Resources:
# 
# 1) [Train a model](../../task1/task/Train%20a%20Model.ipynb)  
# 2) [New Data as a goal](../../task2/task/New%20Data%20as%20a%20Goal.ipynb)  
# 3) [Deployment](../../task3/task/Deployment.ipynb)  
# 
# Suggestions: 
# 
# - Use empty code blocks to find out any informantion necessary to solve this problem: eg: ```!ls [directorypath] prints the files in a given directory``` 
# - Executing the first two cells below will run your python script with test images, the first should return "whale" and the second should return "not whale" 

# Start in [DIGITS](/digits/). 

# In[16]:


#Load modules & set objects for dataset and model
import caffe
import cv2
import sys

MODEL_JOB_DIR = '/dli/data/digits/20190926-221500-abfc'  ## Set this to be the job number for your model
DATASET_JOB_DIR = '/dli/data/digits/20190926-220654-3c91'  ## Set this to be the job number for your dataset

get_ipython().system(u'ls $MODEL_JOB_DIR')


# In[17]:


get_ipython().system(u'ls $DATASET_JOB_DIR')


# In[18]:


ARCHITECTURE = MODEL_JOB_DIR + '/deploy.prototxt'                 # Do not change
WEIGHTS = MODEL_JOB_DIR + '/snapshot_iter_270.caffemodel'    # Do not change


# In[19]:


#Build model
def deploy(img_path):

    caffe.set_mode_gpu()
    
    # Initialize the Caffe model using the model trained in DIGITS. Which two files constitute your trained model?
    net = caffe.Classifier(ARCHITECTURE, WEIGHTS,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256)) 
    
    # Create an input that the network expects. 
    input_image = caffe.io.load_image(DATASET_JOB_DIR+'/mean.jpg')
    input_image = cv2.resize(input_image, (256,256))
    mean_image = caffe.io.load_image('/dli/data/digits/20190926-220654-3c91/mean.jpg')
    ready_image = input_image-mean_image
    
#spot for viz

    # Make prediction
    prediction = net.predict([ready_image])

    # Create an output that is useful to a user. What is the condition that should return "whale" vs. "not whale"?
    if prediction.argmax() == 0:
        return "whale"
    else:
        return "not whale"
    
#Ignore this part    
if __name__ == "__main__":
    print(deploy(sys.argv[1]))


# In[20]:


get_ipython().system(u'python submission.py \'/dli/data/whale/data/train/face/w_1.jpg\'  #This should return "whale" at the very bottom')
get_ipython().system(u'python submission.py \'/dli/data/whale/data/train/not_face/w_1.jpg\'  #This should return "not whale" at the very bottom')

