E-gent: Baby Crying and Sleeping Classifier
This project focuses on fine-tuning the VGG-Face model from DeepFace to classify images of babies as either "Crying" or "Sleeping." The model is trained using a custom dataset with two main categories: "crying" and "sleeping."

Key Features:
Fine-tuned VGG-Face: Utilizes the pre-trained VGG-Face model for facial recognition and fine-tunes it to classify baby emotions (crying vs sleeping).
Binary Classification: A simple binary classification task using the sigmoid activation function for output.
Data Augmentation: Various augmentation techniques like rotation, width shift, height shift, zoom, etc., are applied to the dataset to improve model generalization.
Real-Time Prediction: The trained model can be used for real-time detection of crying and sleeping behaviors from images or video feeds.
Project Objective:
To detect and classify the emotional state of babies in images based on the given categories, helping to monitor their well-being.
Dependencies:
DeepFace
TensorFlow
Keras
OpenCV
Numpy
Matplotlib
