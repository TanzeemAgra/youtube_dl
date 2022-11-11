
from django.shortcuts import render
import keras
from PIL import Image
import numpy as np
import os
from django.core.files.storage import FileSystemStorage

# Create your views here.
media='media'
model = keras.models.load_model('../saved_models/trained.h5')

def makepredictions(path):
    #we open the image

    img=Image.open(path)

    #we resize the image for model

    img_d = img.resize((255,255))

    # we check if image is RGB or not

    if len(np.array(img_d).shape)<4:
        rgb_img =Image.new("RGB",img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img=img_d


    # here we convert the image into numpy array and reshape
    rgb_img=np.array(rgb_img,dtype=np.float64)
    rgb_img=rgb_img.reshape(-1,255,255,3)

    #we make predictions here

    predictions =model.predict(rgb_img)
    a=int(np.argmax(predictions))
    if a==1:
        a="Result : Bulbasaur"
    elif a==2:
        a="Result : Charmander"
        
    elif a==3:
        a="Result : Squirtle"
    else:
        a="Result: Tauros"
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
