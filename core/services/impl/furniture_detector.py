import os
import torch
import logging
from threading import Lock
from ultralytics import YOLO
from PIL import Image

from DesignAI import settings
from core.services.img_predictor import Predictor

logger = logging.getLogger(__name__)

class FurnitureDetector(Predictor):
    MODEL_PATH = os.path.join(settings.ML_MODELS_DIR, 'yolo_furniture_detector.pt')
    classes = ['Bed', 'Chair', 'Cupboard', 'Nightstand', 'Sideboard', 'Sofa', 'Table']
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FurnitureDetector, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance

    def __init__(self):
        if self.initialized:
            return

        try:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")

            self.model = YOLO(self.MODEL_PATH)
            logger.info(f"Model loaded from {self.MODEL_PATH}")

            # Set confidence and IOU thresholds
            self.conf_threshold = 0.25
            self.iou_threshold = 0.45

            self.initialized = True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def get_name(self) -> str:
        return "furniture_detector"

    def predict(self, image_file):
        if not self.initialized:
            logger.error("Model not initialized")
            return []

        try:
            image = Image.open(image_file)
            results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)

            detections = []
            for result in results:
                boxes = result.boxes
                detections = {
                    self.classes[int(box.cls[0].item())]:float(box.conf[0])
                    for box in boxes
                }

            return detections

        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []