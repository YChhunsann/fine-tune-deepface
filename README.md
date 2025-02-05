# E-gent: Baby Crying and Sleeping Classifier

This project focuses on fine-tuning the VGG-Face model from DeepFace to classify images of babies as either "Crying" or "Sleeping." The model is trained using a custom dataset with two main categories: "crying" and "sleeping."

## Key Features:
- **Fine-tuned VGG-Face**: Utilizes the pre-trained VGG-Face model for facial recognition and fine-tunes it to classify baby emotions (crying vs sleeping).
- **Binary Classification**: A simple binary classification task using the sigmoid activation function for output.
- **Data Augmentation**: Various augmentation techniques like rotation, width shift, height shift, zoom, etc., are applied to the dataset to improve model generalization.
- **Real-Time Prediction**: The trained model can be used for real-time detection of crying and sleeping behaviors from images or video feeds.

## Project Objective:
To detect and classify the emotional state of babies in images based on the given categories, helping to monitor their well-being.

## Dependencies:
- **DeepFace**
- **TensorFlow**
- **Keras**
- **OpenCV**
- **Numpy**
- **Matplotlib**

## How to Use:
1. Clone this repository.
2. Install the required libraries by running:
    ```
    pip install -r requirements.txt
    ```
3. Train the model by following the training script `fine_tune_deepface.py`.
4. Use the `test_manually.py` and `real_time_testing.py` script to test the model on new images or real-time video.

## Model:
- The VGG-Face model has been fine-tuned on a dataset of baby images labeled as "crying" and "sleeping."
- The model can be used for classification tasks and real-time detection.

## Example Usage:
- The trained model can be tested on images with the following command:
    ```python
    python test_manually.py --image_path <path_to_image>
    ```
    or
    ```
    python real_time_testing.py
    ```
