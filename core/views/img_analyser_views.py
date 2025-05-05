from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from core.exceptions import PredictorFailure
from core.serializers import ImageUploadSerializer
from core.services.img_predictor import Predictor
from core.services.impl.design_classifier import DesignClassifier
from core.services.impl.furniture_detector import FurnitureDetector
from core.services.impl.room_classifier import RoomClassifier

class ImgAnalyserView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictors = [RoomClassifier(), DesignClassifier(), FurnitureDetector()]

    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            result = {}

            for predictor in self.predictors:
                try:
                    result[predictor.get_name()] = predictor.predict(image)
                except Exception as e:
                    raise PredictorFailure(f"{predictor.get_name()} failed: {str(e)}")

            return Response(result)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)