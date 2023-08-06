import tensorflow as tf

IMG_SIZE = (224, 224)
class_names_gender = ['man', 'woman']
skin_type = ['dry', 'oily']

# loading the model
gender_model = tf.keras.models.load_model("Gender-Recognition Model")
skin_model = tf.keras.models.load_model("Skin-Type-Recognition-Degraded")

# Loading and reading the images
def load_and_prep(filepath):
  img = tf.io.read_file(filepath)
  img = tf.io.decode_image(img)
  img = tf.image.resize(img, IMG_SIZE)

  return img

filepath = "./RealTimeDetections/1652850345.jpg"
img = load_and_prep(filepath)

# Gender Prediction 
pred_prob_gender = gender_model.predict(tf.expand_dims(img, axis=0)) 
pred_class_gender = class_names_gender[pred_prob_gender.argmax()]

# Skin prediction
pred_prob_skin = skin_model.predict(tf.expand_dims(img, axis=0))
pred_class_skin = skin_type[pred_prob_skin.argmax()]

print(pred_class_gender, pred_class_skin)

import os

imgs = os.listdir("RealTimeDetections")
filepath = f"RealTimeDetections\{imgs[-4]}"
print(filepath)