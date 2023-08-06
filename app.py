import os
import cv2
import time
import numpy as np
import tensorflow as tf
import webbrowser

from flask import jsonify
from FaceDetectionModule import faceDetector
from flask import Flask, render_template, Response

app = Flask(__name__)

# loading the model
gender_model = tf.keras.models.load_model("Gender-Recognition Model")
skin_model = tf.keras.models.load_model("Skin-Type-Recognition-Degraded")
skin_disease_model = tf.keras.models.load_model("Skin_Disease_Model")

url = 123
cap = cv2.VideoCapture(0)
# cap.set(3, 924)
# cap.set(4, 800)
cap.set(10, 150)

def FPS(img, fps, latency):
	cv2.putText(img, f"FPS: {str(int(fps))}", org=(7, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
				fontScale=1, color=(0, 0, 0), thickness=1)

	cv2.putText(img, f"Latency: {str(latency)}s", org=(97, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
				fontScale=1, color=(0, 0, 0), thickness=1)

	return img

def gen_frames():
	pTime, pTimeL = 0, 0
	previous = time.time()
	detector = faceDetector(minDetectionCon=0.5)
	a, b = 0, 0
	expTime = 0
	while True:
		_, img_orig = cap.read()
		img = cv2.flip(img_orig, 1)

		bboxs = detector.findFaces(img)

		curTime = time.time()
		expTime = curTime - expTime
		expTime = curTime
		# if expTime > 3 and expTime < 4:
		filename = str(int(expTime))
		if len(bboxs) != 0:
			x, y, w, h = bboxs[0][1]
			# img_pred = img_orig[y:y + h, x:x + w]
			img_pred = img_orig[y:y + h, x:x + w]
			img_pred = cv2.resize(img_pred, (224, 224))
			# if b < 100:
			cv2.imwrite(f"./RealTimeDetections/{filename}.jpg", img_pred)
				# b += 1

		# # FPS
		cTimeL = time.time()

		cTime = time.time()
		if (cTime - pTime) != 0:
			fps = 1 / (cTime - pTime)
			latency = np.round((cTimeL - pTimeL), 4)
			pTime, pTimeL = cTime, cTimeL
			a += 1

			img = FPS(img, fps, latency)

		# Drawing boxes
		img = detector.fancyDraw(img, bboxs[0][1])
		# if len(bboxs) != 0:
		# 	print(bboxs[0][1])

		# Video stream
		ret, buffer = cv2.imencode('.jpg', img)
		img = buffer.tobytes()
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


IMG_SIZE = (224, 224)
class_names_gender = ['man', 'woman']
skin_type = ['dry', 'oily']
skin_disease = ['Acne', 'Normal']

# Loading and reading the images
def load_and_prep(filepath):
  img = tf.io.read_file(filepath)
  img = tf.io.decode_image(img)
  img = tf.image.resize(img, IMG_SIZE)

  return img

def Model_predictions():
	imgs = os.listdir("RealTimeDetections")
	filepath = f"RealTimeDetections\{imgs[-4]}"

	img = load_and_prep(filepath)

	# Gender Prediction
	pred_prob_gender = gender_model.predict(tf.expand_dims(img, axis=0))
	pred_class_gender = class_names_gender[pred_prob_gender.argmax()]

	# Skin prediction
	pred_prob_skin = skin_model.predict(tf.expand_dims(img, axis=0))
	pred_class_skin = skin_type[pred_prob_skin.argmax()]

	# Skin disease prediction
	pred_prob_skin_disease = skin_disease_model.predict(tf.expand_dims(img, axis=0))
	pred_class_skin_disease = skin_disease[pred_prob_skin_disease.argmax()]

	return pred_class_gender, pred_class_skin, pred_class_skin_disease


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict')
def model_preds():
	global pred_class_gender, pred_class_skin, pred_class_skin_disease
	pred_class_gender, pred_class_skin, pred_class_skin_disease = Model_predictions()
	return render_template('analysis.html', gender=pred_class_gender, skinType=pred_class_skin,skinDisease=pred_class_skin_disease)

@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/model-predictions')
def preds():
	return jsonify({'gender': pred_class_gender,
                    'skin-type': pred_class_skin,
					'skin-disease': pred_class_skin_disease})

if __name__ == "__main__":
	# app.run(debug=True)
    webbrowser.open_new('http://127.0.0.1:2000/')
    app.run(debug=True, port=2000)