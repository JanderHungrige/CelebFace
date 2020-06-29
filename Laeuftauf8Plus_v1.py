#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:11:31 2020

@author: base

Hier nutzen wir die Keras face_api (https://github.com/rcmalli/keras-vggface) um ein eigenes Gesicht mit celebrity Gesichtern verglichen.
Das eigene Gesicht wird mit opencv aufgenommen. Erkannt wird das Gesicht mittels Cascade classifier und dann ausgeschnitten.

In diesem Codeblock wird das model vorher transformiert. In dem jupyter notebook "Save TFlight model" Geladen (über import VGGFace)
Das model wird dann zuerst in h5 Format gespeichert, wieder geladen und mittels dem tlight converter convertiert. Es wird darüber diskutiert, aber das model muss scheinbar nicht compiled werde (trotz wWarnung)
Compilen bedeuted wohl nur das vorbereiten auf training. Was wir ja nicht vorhaben. Wir haben beide Varianten getestet und beide Varianten haben funktioniert.
Das tflight Model wird hier geladen, tensoren allokiert, und dann inferred. Das Ergebniss ist das Gleiche wie vorher bei der ersten version ohne tflight, bloß schneller.
"""
import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
from time import time
import concurrent.futures
import sys

print('Tensorflowversion: ' + tf.__version__)

# Define Variables
#-----------------------------------------------------------------------------

video_input_device=0
Gesichter= False  # either True for only croped celebrity faces or False for original celbrity image. DEciding which images to show. Cropped or total resized image
brt = 90  # value could be + or - for brightness or darkness
gray=False
p=35# frame size around detected face
width=height=224 # size of the cropped image. Same as required for network
mitte=np.empty(shape=[0, 0])
mittleres_Gesicht_X=()
Runningsystem='PC' # 'PC' or 'ARM' as string

if Runningsystem =='PC':
    cascaderpath=''
    modelpath='compare the models delete/tflite'
    embeddingpath='Embeddings_working/'
    
if Runningsystem =='ARM':
    cascaderpath='cascader/'
    modelpath='models/tflite'
    embeddingpath='Embeddings/'
    
#Load face cascader 
#-----------------------------------------------------------------------------

#face_cascade = cv2.CascadeClassifier(str(Path.cwd() / cascaderpath / 'haarcascade_frontalface_alt.xml'))
#face_cascade = cv2.CascadeClassifier(str(Path.cwd() / cascaderpath / 'lbpcascade_frontalface.xml'))
face_cascade = cv2.CascadeClassifier(str(Path.cwd() / cascaderpath / 'lbpcascade_frontalface_improved.xml'))
print('cascader loaded  ...')


# Load Model
#-----------------------------------------------------------------------------

# Load TFLite model and allocate tensors.Beide modelle funktionieren

#model_path=str(Path.cwd() / modelpath / "quanmtized_modelh5-15.tflite") # 9.694556474685669
model_path=str(Path.cwd() / modelpath / "converted_model.tflite") #modelpath
#model_path=str(Path.cwd() / modelpath /  "quantized_model.tflite") #
#model_path=str(Path.cwd() / modelpath /  "quantized_modelpb13-13.tflite")#
#model_path=str(Path.cwd() / modelpath /  "quantized_modelh5-13.tflite")#
#model_path=str(Path.cwd() / modelpath / "model2020.tflite") # 
try:  
    interpreter = tf.lite.Interpreter(model_path)   # input()    # To let the user see the error message
except ValueError as e:
    print("Error: Modelfile could not be found. Check if you are in the correct workdirectory. Errormessage:  " + str(e))
    #Depending on the version of TF running, check where lite is set :
    if tf.__version__.startswith ('1.'):
        print('lite in dir(tf.contrib)' + str('lite' in dir(tf.contrib)))

    elif tf.__version__.startswith ('2.'):
        print('lite in dir(tf)? ' + str('lite' in dir(tf)))

    #os.chdir('/home/root')
    print('workdir: ' + os.getcwd())
    sys.exit()

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 

print('model loaded        ...')

# Load Embeddings
#-----------------------------------------------------------------------------
##LOADING CSV (easier with pandas, but 8M Plus does not yet support pandas)
import json 
    
f = open((Path.cwd() / embeddingpath  / 'EMBEDDINGS.json'),'r') 
ImportedData =json.load(f)
dataE=[np.array(ImportedData['Embedding'][str(i)]) for i in range(len(ImportedData['Name']))]
dataN=[np.array(ImportedData['Name'][str(i)]) for i in range(len(ImportedData['Name']))]
dataF=[np.array(ImportedData['File'][str(i)]) for i in range(len(ImportedData['Name']))]

# In two steps for explanation
# dataN=ImportedData['Name']
# dataF=ImportedData['File']
# dataE=ImportedData['Embedding']
# aberE=[np.array(dataE[str(i)]) for i in range(len(dataE))]
# aberF=[np.array(dataF[str(i)]) for i in range(len(dataF))]
# aberN=[np.array(dataN[str(i)]) for i in range(len(dataN))]
print('Embeddings loaded      ...')

#Define functions
#-----------------------------------------------------------------------------
def preprocess_input(x, data_format, version): #Choose version same as in " 2-Create embeddings database.py or jupyter"
    x_temp = np.copy(x)
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp

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

print('functions defined         ...')
# Split data for threadding
#-----------------------------------------------------------------------------
celeb_embeddings=splitDataFrameIntoSmaller(dataE, int(np.ceil(len(dataE)/4)))   
# celeb_Names=splitDataFrameIntoSmaller(dataN, int(np.ceil(len(dataN)/4)))   
# celeb_File=splitDataFrameIntoSmaller(dataF, int(np.ceil(len(dataF)/4)))   
print('Embeddings split             ...')


#open Camera, Get frame middel for frame optimization
#-----------------------------------------------------------------------------
cap= cv2.VideoCapture(video_input_device)

if not cap.isOpened():
    print('Error: VideoCapture not opened')
    sys.exit(0)


ret, frame = cap.read() 
framemitte=np.shape(frame)[1]/2
print('camera loaded                   ...')
print('pre-processing done                !!!')

#Start
#-----------------------------------------------------------------------------


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
                samples = np.expand_dims(pixels, axis=0)
                samples = preprocess_input(samples, data_format=None, version=2)#data_format= None, 'channels_last', 'channels_first' . If None, it is determined automatically from the backend
                #now using the tflight model
                input_shape = input_details[0]['shape']
                input_data = samples
                interpreter.set_tensor(input_details[0]['index'], input_data)

                interpreter.invoke()
                EMBEDDINGS = interpreter.get_tensor(output_details[0]['index'])

                #print('.')
                end4=time()
                print('create face embeddings' , end4-start4)
# READ CELEB EMBEDDINGS AND COMPARE  
                start_EU=time()
                EuDist=[]
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    ergebniss_1=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[0]))
                    ergebniss_2=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[1]))
                    ergebniss_3=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[2]))
                    ergebniss_4=executor.submit(faceembeddingNP,EMBEDDINGS,np.array(celeb_embeddings[3]))

                if ergebniss_1.done() & ergebniss_2.done() & ergebniss_3.done() & ergebniss_4.done():
                    EuDist.extend(ergebniss_1.result())
                    EuDist.extend(ergebniss_2.result())
                    EuDist.extend(ergebniss_3.result())
                    EuDist.extend(ergebniss_4.result())
                end_EU=time()
                print('Create_EuDist', end_EU-start_EU)

                start_Min=time()     
                folder_idx= dataN[np.argmin(EuDist)]
                image_idx = dataF[np.argmin(EuDist)] 
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
                numpy_horizontal= cv2.putText(numpy_horizontal, str(dataN[np.argmin(EuDist)]), (5, 17), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.9, (116, 161, 142), 1)
                cv2.imshow('ItsYou', numpy_horizontal)   
                end6=time()
                print('print found image', end6-start6)
                total=time()
                print('totaltime ', total-start1)
                print(' Distance value: ', np.argmin(EuDist), ' | ' , 'Name: ', str(dataN[np.argmin(EuDist)]),' | ' ,' Filename: ', str(dataF[np.argmin(EuDist)]))
                
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