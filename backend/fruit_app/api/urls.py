from django.urls import include
from django.urls import path, include
from api.api_views import *


urlpatterns = [
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.jwt')),
    path('fresh-stale-predictions/', FreshStaleFruitAPIView.as_view(), name='fresh_stale_prediction'),
    path('fruit-classification/', FruitClassificationAPIView.as_view(), name='fruit_classification'),
    path('edibility-prediction/', EdibilityAPIView.as_view(), name='edibility_prediction'),
    path('disease-prediction/', DiseasePredictionAPIView.as_view(), name='disease_prediction'),
    path('ripened-method-prediction/', RipenedMethodPredictionAPIView.as_view(), name='ripened_method_prediction'),
    path('grade/', GradingAPIView.as_view(), name='grading'),
    path('fruit-price-prediction/', FruitPricePredictionAPIView.as_view(), name='fruit_price_prediction'),
    path('price-forecast/', PriceTimeSeriesAPIView.as_view(), name='price_forecast'),
]
