import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from DesignAI import settings
from core.ai_models.design_classifier_model import DesignClassifierModel
from core.services.img_predictor import Predictor

class DesignClassifier(Predictor):
    MODEL_PATH = os.path.join(settings.ML_MODELS_DIR, 'design_classifier_final_model.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['Asian', 'Contemporary', 'Craftsman', 'Eclectic', 'Farmhouse',
                   'Industrial', 'Mediterranean', 'Mid-century-modern', 'Modern',
                   'Rustic', 'Scandinavian', 'Traditional']

    def __init__(self):
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {self.MODEL_PATH}")

        self.model = DesignClassifierModel.build_model(num_classes=len(self.class_names), device=self.device)
        self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_name(self) -> str:
        return "design_classifier"

    def predict(self, image_file):
        image = Image.open(image_file)

        processed = self._preprocess_image(image).to(self.device)

        with torch.no_grad():
            outputs = self.model(processed)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        result = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        return result

    def _preprocess_image(self, image):
        image = image.convert('RGB')
        return self.transform(image).unsqueeze(0)