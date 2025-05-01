from rest_framework import status
from rest_framework.exceptions import APIException


class PredictorFailure(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Prediction failed.'
    default_code = 'prediction_error'