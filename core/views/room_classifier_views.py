from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from core.serializers import ImageUploadSerializer
from core.services.room_classifier import RoomClassifier

classifier = RoomClassifier()

class RoomClassifierView(APIView):
    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            prediction = classifier.predict(image)
            return Response({"prediction": prediction})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)