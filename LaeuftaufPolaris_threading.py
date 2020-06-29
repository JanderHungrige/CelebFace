#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:35:23 2020

@author: base
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 21:44:09 2020

@author: base
"""
import numpy as np
import cv2
import os
import pandas as pd
from pathlib import Path
from numpy import expand_dims
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from time import time

import concurrent.futures

cap= cv2.VideoCapture(0)

if not cap.isOpened():
    print('VideoCapture not opened')
    #exit(0)

#face_cascade = cv2.CascadeClassifier(str(Path.cwd() / 'haarcascade_frontalface_alt.xml'))
#face_cascade = cv2.CascadeClassifier(str(Path.cwd() / 'lbpcascade_frontalface.xml'))
face_cascade = cv2.CascadeClassifier(str(Path.cwd() / 'lbpcascade_frontalface_improved.xml'))

Gesichter= True  # either True for only croped celebrity faces or False for original celbrity image. DEciding which images to show. Cropped or total resized image
brt = 90  # value could be + or - for brightness or darkness
gray=False
p=35# frame size around detected face
width=height=224 # size of the cropped image. Same as required for network
mitte=np.empty(shape=[0, 0])
mittleres_Gesicht_X=()

resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max

EMBEDDINGS_Celebs=pd.read_json(Path.cwd() / 'EMBEDDINGS_8k.json')

ret, frame = cap.read() 
framemitte=np.shape(frame)[1]/2

def splitDataFrameIntoSmaller(df, chunkSize):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf
      
def faceembedding(YourFace,CelebDaten):
    Dist=[]
    for i in range(len(CelebDaten.File)):
        Celebs=np.array(CelebDaten.Embedding[i]) 
        Dist.append(np.linalg.norm(YourFace-Celebs))    
    return Dist
 
def faceembeddingNP(YourFace,CelebDaten):
    Dist=[]
    for i in range(len(CelebDaten)):
        Celebs=np.array(CelebDaten[i]) 
        Dist.append(np.linalg.norm(YourFace-Celebs))    
    return Dist

celeb_embeddings=splitDataFrameIntoSmaller(EMBEDDINGS_Celebs, int(np.ceil(len(EMBEDDINGS_Celebs)/4)))   

while(True):
# CAPTURE FRAME BY FRAME    
    ret, frame = cap.read() 
    if gray==True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame=cv2.flip(frame,1)  
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame) 
    
#DECTECT FACE IN VIDEO CONTINUOUSLY       
    faces_detected = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)#, Size(50,50))
    for (x,y,w,h) in faces_detected:
        rechteck=cv2.rectangle(frame, (x-p, y-p+2), (x+w+p, y+h+p+2), (0, 255, 0), 2)  
        #rechteck=cv2.rectangle(frame, (x-p, y-p+2), (x+int(np.ceil(height))+p, y+int(np.ceil(height))+p+2), (0, 0, 100), 2)  
        cv2.imshow('frame', rechteck)     

# DETECT KEY INPUT  - ESC OR FIND MOST CENTERED FACE  
    key = cv2.waitKey(1)
    if key == 27: #Esc key
        cap.release()
        cv2.destroyAllWindows()
        break
    if key ==32: 
        mittleres_Gesicht_X=()
        mitte=()
        if faces_detected != (): # only if the cascader detected a face, otherwise error
            start1 = time()
#FIND MOST MIDDLE FACE            
            for (x,y,w,h) in faces_detected:
                mitte=np.append(mitte,(x+w/2))               
            mittleres_Gesicht_X = (np.abs(mitte - framemitte)).argmin()
            end1 = time()
            print('detect middel face ', end1-start1)
# FRAME THE DETECTED FACE
            start2=time()
            print(faces_detected[mittleres_Gesicht_X])
            (x, y, w, h) = faces_detected[mittleres_Gesicht_X]
            img=frame[y-p+2:y+h+p-2, x-p+2:x+w+p-2] #use only the detected face; crop it +2 to remove frame # CHECK IF IMAGE EMPTY (OUT OF IMAGE = EMPTY)     

            if len(img) != 0: # Check if face is out of the frame, then img=[], throwing error
                end2=time()
                print('detect face ',end2-start2)

# CROP IMAGE 
                start3=time()
                if img.shape > (width,height): #downsampling
                    img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) #resize the image to desired dimensions e.g., 256x256  
                elif img.shape < (width,height): #upsampling
                    img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC) #resize the image to desired dimensions e.g., 256x256                      
                cv2.imshow('frame',img_small)
                cv2.waitKey(1) #hit any key
                end3=time()
                print('face crop', end3-start3)
#CREATE FACE EMBEDDINGS
                start4=time()
                pixels = img_small.astype('float32')
                samples = expand_dims(pixels, axis=0)
                samples = preprocess_input(samples, version=2)
                EMBEDDINGS = resnet50_features.predict(samples)
                #print('.')
                end4=time()
                print('create face embeddings' , end4-start4)
# READ CELEB EMBEDDINGS AND COMPARE  
                start_EU=time()
                EuDist=[]
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    ergebniss_1=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[0].Embedding))
                    ergebniss_2=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[1].Embedding))
                    ergebniss_3=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[2].Embedding))
                    ergebniss_4=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[3].Embedding))                    

                if ergebniss_1.done() & ergebniss_2.done() & ergebniss_3.done() & ergebniss_4.done():
                    EuDist.extend(ergebniss_1.result())
                    EuDist.extend(ergebniss_2.result())
                    EuDist.extend(ergebniss_3.result())
                    EuDist.extend(ergebniss_4.result())
                end_EU=time()
                print('Create_EuDist', end_EU-start_EU)

                start_Min=time()     
                folder_idx= EMBEDDINGS_Celebs.Name[np.argmin(EuDist)]
                image_idx = EMBEDDINGS_Celebs.File[np.argmin(EuDist)] 
                end_Min=time()
                print('find minimum for facematch', end_Min-start_Min)
                
# PLOT IMAGES       
                start6=time()
                path=Path.cwd()

                if Gesichter == False:
                    pfad=str(Path.cwd() / 'sizeceleb_224_224' / str(folder_idx) / str(image_idx))
                elif Gesichter == True:
                    pfad=str(Path.cwd() / 'Celebs_faces' / str(folder_idx) / str(image_idx))    
                    
                Beleb=cv2.imread(pfad)                  
                if np.shape(Beleb) != (width,height): 
                    Beleb=cv2.resize(Beleb, (np.shape(img_small)[0] ,np.shape(img_small)[1]), interpolation=cv2.INTER_AREA)
                numpy_horizontal = np.hstack((img_small, Beleb))
                cv2.namedWindow('ItsYou',cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('ItsYou', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                numpy_horizontal= cv2.putText(numpy_horizontal, EMBEDDINGS_Celebs.Name[np.argmin(EuDist)], (5, 17), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.9, (116, 161, 142), 1)
                cv2.imshow('ItsYou', numpy_horizontal)   
                end6=time()
                print('print found image', end6-start6)
                total=time()
                print('totaltime ', total-start1)
                print(' Distance value: ', np.argmin(EuDist), ' | ' , 'Name: ', EMBEDDINGS_Celebs.Name[np.argmin(EuDist)],' | ' ,' Filename: ', EMBEDDINGS_Celebs.File[np.argmin(EuDist)])
                
# CLEARING ALL VARIANLES AND CONTINUE WITH THE PROGRAM
                cv2.waitKey(0) #hit any key
                faces_detected=None
                mittleres_Gesicht_X=None        
                img=None
                img_small=None
                pixels=None
                samples=None
                EMBEDDINGS=None          
                cv2.destroyWindow('ItsYou')
                if key == 27: #Esc key
                    break


            else: 
                rame= cv2.putText(frame, 'FACE MUST BE IN FRAME', (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (129, 173, 181), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(900)
                
        else:
            print('noface detected')
            
        

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

# Now detect face and crop

