import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import logging
from threading import Lock
from typing import Dict
from DesignAI import settings
from core.ai_models.room_classifier_model import RoomClassifierModel
from core.services.img_predictor import Predictor

# Configure logging - only important messages
logger = logging.getLogger(__name__)

class RoomClassifier(Predictor):
    MODEL_PATH = os.path.join(settings.ML_MODELS_DIR, 'room_classifier_final_model.pth')
    class_names = ['Bathroom', 'Bedroom', 'Dinning', 'Kitchen', 'Living Room']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Singleton pattern variables
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:  # Thread safety
            if cls._instance is None:
                logger.info("Creating RoomClassifier singleton instance")
                cls._instance = super(RoomClassifier, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance

    def __init__(self):
        # Skip initialization if already done
        if hasattr(self, 'initialized') and self.initialized:
            return

        logger.info(f"Initializing RoomClassifier on device: {self.device}")

        try:
            # Set up transformation pipeline
            self.transform = transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            # Validate model path
            if not os.path.exists(self.MODEL_PATH):
                error_msg = f"Model file not found: {self.MODEL_PATH}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Load model
            logger.info(f"Loading room classifier model from {self.MODEL_PATH}")
            self.model = RoomClassifierModel.build_model(num_classes=len(self.class_names), device=self.device)
            self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            self.initialized = True
            logger.info("RoomClassifier initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RoomClassifier: {str(e)}")
            raise

    def get_name(self) -> str:
        return "room_classifier"

    def predict(self, image_file) -> Dict[str, float]:
        """
        Predict room type with confidence scores for all classes

        Args:
            image_file: Path to image file or file-like object

        Returns:
            Dictionary mapping room types to confidence scores
        """
        if not hasattr(self, 'initialized') or not self.initialized:
            logger.error("Model not properly initialized")
            raise RuntimeError("Model not properly initialized")

        try:
            # Open and preprocess image
            image = Image.open(image_file)
            processed = self._preprocess_image(image).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(processed)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

            # Format results
            result = {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            }

            # Log only the top prediction
            top_prediction = max(result.items(), key=lambda x: x[1])
            logger.info(f"Predicted room type: {top_prediction[0]} ({top_prediction[1]:.4f})")

            return result

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _preprocess_image(self, image):
        """
        Process input image for model prediction

        Args:
            image: PIL Image object

        Returns:
            Tensor ready for model input
        """
        try:
            image = image.convert('RGB')
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise