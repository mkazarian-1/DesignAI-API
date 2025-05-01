from django.urls import path
from core.views.img_analyser_views import ImgAnalyserView

urlpatterns = [
    path('base_img_analyse/', ImgAnalyserView.as_view(), name='img_analyser')
]
