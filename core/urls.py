from django.urls import path
from core.views.room_classifier_views import RoomClassifierView

urlpatterns = [
    path('classify-room/', RoomClassifierView.as_view(), name='classify-room')
]
