from deepface import DeepFace
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_model():
    base_model = DeepFace.build_model("VGG-Face").model  

    for layer in base_model.layers[:-5]:  
        layer.trainable = False  

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) 
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    predictions = Dense(1, activation='sigmoid')(x)  

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    "dataset/",            
    target_size=(224, 224),  
    batch_size=32,
    class_mode='binary', 
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    "dataset/",            
    target_size=(224, 224),  
    batch_size=32,
    class_mode='binary', 
    subset='validation'
)

crying_count, sleeping_count = 236, 176 
total_samples = crying_count + sleeping_count
weight_for_crying = total_samples / (2 * crying_count)
weight_for_sleeping = total_samples / (2 * sleeping_count)

class_weights = {0: weight_for_crying, 1: weight_for_sleeping}

model.fit(train_generator, validation_data=validation_generator, epochs=20, class_weight=class_weights)

model.save("baby_crying_detector.h5")
print("Model training complete and saved as baby_crying_detector.h5")
