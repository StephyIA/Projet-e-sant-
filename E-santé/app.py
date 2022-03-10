import os
from flask import Flask, request,redirect, render_template, url_for
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, Subgroup, View, Link, Separator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

app=Flask(__name__)
model=load_model("malaria_detector.h5")
modelpoumon=load_model("modelpoumon.h5")
modeltumeur=load_model("VGG16_model.h5")
APP_ROOT=os.path.dirname(os.path.abspath(__file__))

image_shape = (130, 130, 3)
Bootstrap(app)
@app.route('/predictpalu', methods=['GET', 'POST'])
@app.route('/palu.html', methods=['GET', 'POST'])
@app.route('/', methods=['GET','POST'])
def predictpalu():

    target=os.path.join(APP_ROOT,'imagespalu/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    if request.method == "POST":
        print("okok")
        imagefile = request.files['file']
        print(imagefile)
        #if imagefile.filename !='':
            #print("le fichier n'est pas vide")
        filename = imagefile.filename
            #print("voici le nom de image", filename)
        destination = "/".join([target, filename])
        print(destination)
        imagefile.save(destination)
    #else:
        #print("pas image")
        my_image = image.load_img(destination, target_size=image_shape)
        print(type(my_image))
        # image = image_gen.fit(file)
        # my_image =image.load_img(file, target_size=image_shape[:2])
        my_image = image.img_to_array(my_image)
        print(my_image.shape)
        my_image = np.expand_dims(my_image, axis=0)
        prediction = model.predict(my_image) > 0.5
        prediction=prediction.astype('int32')
        print(prediction)
        prediction=prediction[0][0]
        return render_template('palu.html',prediction_text='{}'.format(prediction))

    return render_template("palu.html")

@app.route('/poumon', methods=['GET', 'POST'])
@app.route('/poumons.html', methods=['GET', 'POST'])
def poumons():
    targetpoumon = os.path.join(APP_ROOT, 'imagespoumon/')
    print(targetpoumon)
    if not os.path.isdir(targetpoumon):
        os.mkdir(targetpoumon)
    if request.method == "POST":
        print("okok")
        imagefilepoumon = request.files['file']
        print(imagefilepoumon)
        # if imagefile.filename !='':
        # print("le fichier n'est pas vide")
        filename = imagefilepoumon.filename
        # print("voici le nom de image", filename)
        destinationpoumon = "/".join([targetpoumon, filename])
        print(destinationpoumon)
        imagefilepoumon.save(destinationpoumon)
        # else:
        # print("pas image")
        my_image = image.load_img(destinationpoumon, target_size=(224,224,3))
        print(type(my_image))
        # image = image_gen.fit(file)
        # my_image =image.load_img(file, target_size=image_shape[:2])
        my_image = image.img_to_array(my_image)
        print(my_image.shape)
        my_image = np.expand_dims(my_image, axis=0)
        prediction = modelpoumon.predict(my_image)

        print(prediction)
        pred_class = decode_predictions(prediction, top=1)
        print(pred_class)
        result = str(pred_class[0][0][1])
        #prediction = prediction[0][0]
        return render_template('poumons.html', prediction_poumon='{}'.format(prediction))
    return render_template("poumons.html")


@app.route('/tumeur', methods=['GET', 'POST'])
@app.route('/tumeur.html')
def tumeur():

    target=os.path.join(APP_ROOT,'imagestumeurs/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    if request.method == "POST":
        print("okok")
        image_tumeur_file = request.files['file']
        print(image_tumeur_file)
        #if imagefile.filename !='':
            #print("le fichier n'est pas vide")
        filename = image_tumeur_file.filename
            #print("voici le nom de image", filename)
        destination = "/".join([target, filename])
        print(destination)
        image_tumeur_file.save(destination)
    #else:
        #print("pas image")
        my_image = image.load_img(destination, target_size=(224,224))
        print(type(my_image))
        # image = image_gen.fit(file)
        # my_image =image.load_img(file, target_size=image_shape[:2])
        my_image = image.img_to_array(my_image)
        print(my_image.shape)
        my_image = np.expand_dims(my_image, axis=0)
        prediction_tumeur = modeltumeur.predict(my_image) > 0.5
        prediction_tumeur=prediction_tumeur.astype('int32')
        print(prediction_tumeur)
        prediction_tumeur=prediction_tumeur[0][0]
        return render_template('tumeur.html', prediction_text='{}'.format(prediction_tumeur))
    return render_template("tumeur.html")

if __name__ == "__main__":
    app.run(port=9000, debug=True)

