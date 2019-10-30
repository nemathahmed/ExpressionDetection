import numpy as np
import cv2
import time
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 

def expmodel():
    IMG_SIZE=48
    tf.reset_default_graph() 
    convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
  
    convnet = conv_2d(convnet, 32, 5, activation ='relu') 
    convnet = max_pool_2d(convnet, 5) 
    
    convnet = conv_2d(convnet, 64, 5, activation ='relu') 
    convnet = max_pool_2d(convnet, 5) 
  
    convnet = conv_2d(convnet, 128, 5, activation ='relu') 
    convnet = max_pool_2d(convnet, 5) 
  
    convnet = conv_2d(convnet, 64, 5, activation ='relu') 
    convnet = max_pool_2d(convnet, 5) 
  
    convnet = conv_2d(convnet, 32, 5, activation ='relu') 
    convnet = max_pool_2d(convnet, 5) 
  
    convnet = fully_connected(convnet, 1024, activation ='relu') 
    convnet = dropout(convnet, 0.8) 
  
    convnet = fully_connected(convnet, 7, activation ='softmax') 
    convnet = regression(convnet, optimizer ='adam', learning_rate = 1e-3, 
      loss ='categorical_crossentropy', name ='targets') 
  
    model = tflearn.DNN(convnet, tensorboard_dir ='log') 
    return model

def show_webcam(model,mirror=False):
    cam = cv2.VideoCapture(0)
    count=0
    firstface=0
    while True:
        now=time.time()

        counte=0
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
        ret_val, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=gray
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            if firstface==0:
                ix=x
                iy=y
                iw=w
                ih=h
                firstface=1
                count=1
                future=now+10
                print(now,future)
        print(now)
        if count==1 and int(time.time())==int(future):
            count=0
            firstface=0
        if count==1:
            
            img = cv2.rectangle(img,(ix,iy),(ix+iw,iy+ih),(255,0,0),2)
            roi_gray = gray[iy:iy+ih, ix:ix+iw]
            roi_color = img[iy:iy+ih, ix:ix+iw]
            #eyes = eye_cascade.detectMultiScale(roi_gray)
        '''
                       for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                counte=1

            count=1
        '''

        #if count==0:
           #img = np.zeros([512,512,3])
           
        '''    
        if count==1:
            if counte==1:
                for i in range(ex,ex+ew):
                    for j in range(ey,ey+eh):
                        img[i,j,0]=0
                        img[i,j,1]=0
                        img[i,j,2]=0
            img=img[y:y+h, x:x+w]
            img=cv2.resize(img,(512,512))
        ''' 
        if count==1:  
            img=img[iy:iy+ih,ix:ix+iw]
            #img=img[:,:,2]
            img=cv2.resize(img,(512,512))
           # print(ix,iy,iw,ih)
        print(np.array(img).shape)
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=np.array(img)
        img1=np.array(img)
        img1=cv2.resize(img1,(48,48))
        img2=img1
        
        print(img1.shape)
        img1=img1.reshape(img1,(48,48,1))
        model=expmodel()
        model_out=model.predict(img1)
        
        model_out=list(model_out[0])
        ind1=model_out.index(max(model_out))
        head='no-face'
        if ind1==0:
            head='anger'
        elif ind1==1:
            head='disgust'
        elif ind1==2:
            head='fear'
        elif ind1==3:
            head='happy'
        elif ind1==4:
            head='sad'
        elif ind1==5:
            head='surprise'
        elif ind1==6:
            head='neutral'
        
        cv2.putText(img,head, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        
        cv2.imshow('yes', img)
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
MODEL_NAME = 'emotion-{}-{}.model'.format(1e-3, '6conv-basic') 
model=expmodel()
model.load(MODEL_NAME)
show_webcam(model)