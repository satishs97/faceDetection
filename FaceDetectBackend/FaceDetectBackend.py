#Import library
import os.path
import numpy as np
import cv2
import json
from flask import Flask, request, Response
import uuid

#Function detect


def faceDetect(img):
    #base_dir = "FaceData/"faces
    #print(id, fullname)
    #Rotation img
    img_rotate_90_counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_rotate_90_counterclockwise, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rotated270 = cv2.rectangle(img_rotate_90_counterclockwise, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #save file
    #base_dir = base_dir + id
    #if not os.path.exists(base_dir):faces
    #    os.makedirs(base_dir)faces
    file_name = '%s.jpg'%uuid.uuid4().hex
    path_file = 'static/'+file_name
    cv2.imwrite(path_file, img_rotate_90_counterclockwise)
    return json.dumps(path_file) #return image file namefaces

#API
app = Flask(__name__)

#route http post to this method
@app.route('/api/upload', methods=['POST'])
def upload():
    #retrieve image from client
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #print(img)
    #retrieve id, fullname from client
    #id = request.form['id']
    #fullname = request.form['fullname']
    #process image
    img_processed = faceDetect(img).replace("\"","")
    #response
    data_response = {
        #"id": id,
        #"fullname": fullname,
        "image_processed": img_processed
    }
    data_json = json.dumps(data_response)
    return Response(response=data_json, status=200, mimetype="application/json") #return json string

#start server
app.run(host='0.0.0.0',port=5000)
