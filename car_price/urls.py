from django.urls import path
from . import views 

urlpatterns = [
    path('', views.car_price_predictor_view, name='car_predict'),
]