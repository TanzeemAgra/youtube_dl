from django.shortcuts import render
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from flask import Flask,render_template,url_for,request
from werkzeug.utils import secure_filename
from keras.models import load_model
#from keras.preprocessing.image import image
import numpy as np
#from keras.compat.v1 import ConfigProto
#from compat.v1 import InteractiveSession
#from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import keras
from django.core.files.storage import FileSystemStorage

media='media'
model=keras.models.load_model('../saved_models/trained.h5')

# Create your views here.

def makepredictions(path):
    #we open the image

    img=Image.open(path)

    #we resize the image for model

    img_d = img.resize((244,244))

    # we check if image is RGB or not

    if len(np.array(img_d).shape)<4:
        rgb_img =Image.new("RGB",img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img=img_d


    # here we convert the image into numpy array and reshape
    rgb_img=np.array(rgb_img,dtype=np.float64)
    rgb_img=rgb_img.reshape(1,244,244,3)

    #we make predictions here

    predictions =model.predict(rgb_img)
    a=int(np.argmax(predictions))
    if a==1:
        a="Result : Glioma Tumor"
    elif a==2:
        a="Result : Meningioma Tumor"
        
    elif a==3:
        a="Result : No Tumor"
    else:
        a="Result: Pictiuary Tumor"
    return a            

def index(request):
    if request.method == "POST" and request.FILES['upload']:

        if 'upload' not in request.FILES:
            err='No images Selected'
            return render(request,'index.html',{'err':err})
        f = request.FILES['upload']
        if f == '':
            err='No files selected'
            return render(request,'index.html',{'err':err})
        upload =request.FILES['upload']
        fss = FileSystemStorage() 
        file =fss.save(upload.name,upload)
        file_url=fss.url(file)
        predictions=makepredictions(os.path.join(media,file))
        return render(request,'index.html',{'pred':predictions,'file_url':file_url})


    

    else:
        return render(request,'index.html')