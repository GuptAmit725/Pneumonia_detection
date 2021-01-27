import os
import tensorflow as tf
from tensorflow import keras
from flask import Flask
from flask import render_template
from flask import request
from keras.models import Model
from keras.layers import *

from prediction import prediction
tf.config.experimental_run_functions_eagerly(True)
# =====================================================================================================================
app=Flask(__name__)
UPLOAD_FOLDER = "C:/Users/amiti/Desktop/academics/MasterProject/Project_vis/static"

model = keras.models.load_model('C:/Users/amiti/Desktop/academics/MasterProject/Project_vis/vgg16_5.h5')
for i in model.layers:
    i.trainable=False
    #print(i.name, i.trainable)

@app.route("/",methods=['GET', 'POST'])

def upload_predict():
    if os.listdir(UPLOAD_FOLDER):
        for dir in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER+'/', dir))
    if request.method=='POST':
        image_file = request.files['image']
        if image_file:
            image_loc = os.path.join(UPLOAD_FOLDER+'/',image_file.filename)
            image_file.save(image_loc)
            img = keras.preprocessing.image.load_img(image_loc, target_size=(128,128))
            if request.form.get('info'):
                consent = 'y'
            else:
                consent = 'n'
            img_name = 'mask_'+image_file.filename
            pred, p = prediction(model,img,img_name)
            print(img_name, image_file.filename)
            return render_template("index.html", prediction=pred.max(),p=p, image_loc= image_file.filename, masked_im_loc=img_name, consent=consent)


    return render_template("index.html", prediction='null. Please upload image!', p=None, image_loc=None, masked_im_loc=None, consent='n')

if __name__ == '__main__':
    app.run(port=8888, debug=True)