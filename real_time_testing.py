import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

model = load_model("baby_crying_detector.h5")

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_array = preprocess_image(frame)

    prediction = model.predict(img_array)[0][0]  

    if prediction > 0.5:
        predicted_class = "sleeping"
    else:
        predicted_class = "crying"

    cv2.putText(frame, f"Prediction: {predicted_class} ({prediction:.2f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Real-Time Baby Crying Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
