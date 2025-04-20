import tensorflow as tf
from PIL import Image
import numpy as np

# Load model once at module level
MODEL_PATH = 'ai-models/room_classifier_final_model_v1.keras'
model = tf.keras.models.load_model(MODEL_PATH)

class RoomClassifier:
    class_names = ['Bathroom', 'Bedroom', 'Kitchen', 'Living Room', 'Office']  # Example classes

    @staticmethod
    def preprocess_image(image):
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_file):
        image = Image.open(image_file)
        processed = self.preprocess_image(image)
        preds = model.predict(processed)
        class_index = np.argmax(preds)
        return self.class_names[class_index]
