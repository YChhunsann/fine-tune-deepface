import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

model = load_model("baby_crying_detector.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0 
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]  
    
 
    if prediction > 0.5:
        predicted_class = "sleeping"  
    else:
        predicted_class = "crying"  


    confidence = max(prediction, 1 - prediction)

    print(f"Predicted Class: {predicted_class} with confidence {confidence:.2f}")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{predicted_class} ({confidence:.2f})")
    plt.show()

    return predicted_class, confidence


test_image_path = "sleeping.jpg"
predict_image(test_image_path)
