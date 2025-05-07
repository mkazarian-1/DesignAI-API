import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from DesignAI import settings
from core.ai_models.design_classifier_model import DesignClassifierModel
from core.services.img_predictor import Predictor
import logging
from threading import Lock
from typing import Dict

# Configure logging
logger = logging.getLogger(__name__)

class DesignClassifier(Predictor):
    MODEL_PATH = os.path.join(settings.ML_MODELS_DIR, 'design_classifier_final_model.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['Asian', 'Contemporary', 'Craftsman', 'Eclectic', 'Farmhouse',
                   'Industrial', 'Mediterranean', 'Mid-Century', 'Modern',
                   'Rustic', 'Scandinavian', 'Traditional']

    # Singleton pattern variables
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:  # Ensure thread safety
            if cls._instance is None:
                logger.info("Creating new DesignClassifier instance")
                cls._instance = super(DesignClassifier, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance

    def __init__(self):
        # Skip initialization if already done
        if hasattr(self, 'initialized') and self.initialized:
            return

        logger.info(f"Initializing DesignClassifier on device: {self.device}")

        try:
            self._validate_model_path()
            self._load_model()
            self._setup_transform()

            self.initialized = True
            logger.info("DesignClassifier initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _validate_model_path(self):
        """Validate that the model file exists."""
        if not os.path.exists(self.MODEL_PATH):
            error_msg = f"Model file not found: {self.MODEL_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _load_model(self):
        """Load the model from the specified path."""
        try:
            self.model = DesignClassifierModel.build_model(num_classes=len(self.class_names), device=self.device)
            self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _setup_transform(self):
        """Setup image transformation pipeline."""
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_name(self) -> str:
        return "design_classifier"

    def predict(self, image_file) -> Dict[str, float]:
        """
        Predict the design style of the given image.

        Args:
            image_file: Path to the image file or file-like object

        Returns:
            Dictionary mapping class names to probabilities
        """
        if not hasattr(self, 'initialized') or not self.initialized:
            error_msg = "Model not properly initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Open and preprocess the image
            image = Image.open(image_file)
            processed = self._preprocess_image(image).to(self.device)

            # Generate predictions
            with torch.no_grad():
                outputs = self.model(processed)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

            # Format results
            result = {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            }

            # Log top prediction only
            top_prediction = max(result.items(), key=lambda x: x[1])
            logger.info(f"Predicted design style: {top_prediction[0]} ({top_prediction[1]:.4f})")

            return result

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def _preprocess_image(self, image):
        """
        Preprocess the image for model input.

        Args:
            image: PIL Image object

        Returns:
            Tensor ready for model input
        """
        try:
            image = image.convert('RGB')
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise