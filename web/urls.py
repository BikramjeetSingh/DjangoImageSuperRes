
from django.urls import path

from web.views import ImageUpscaleView

urlpatterns = [
    path("", ImageUpscaleView.as_view()),
]
