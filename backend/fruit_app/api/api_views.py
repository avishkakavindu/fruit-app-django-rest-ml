from datetime import datetime
from datetime import timedelta

from django.core.files.storage import default_storage
from rest_framework.response import Response
from rest_framework.views import APIView
from api.models import *
from rest_framework import mixins, generics, status
from rest_framework_simplejwt.authentication import JWTTokenUserAuthentication
from rest_framework import authentication, permissions
from django.db.models import Q
from rest_framework import generics
from tensorflow.keras.models import load_model
import cv2
import numpy as np

FRESH_STALE_LABELS = ['fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum', 'fresh_orange', 'fresh_tomato', 'stale_apple', 'stale_banana', 'stale_bitter_gourd', 'stale_capsicum', 'stale_orange', 'stale_tomato']
FRUIT_CLASSIFICATION_LABELS = ['Banana', 'Mango', 'Papaya', 'Watermelon']
EDIBILITY_CLASSIFICATION_LABELS = ['Not_edible', 'edible']


def get_processed_input_img(image_path, size=224):
    """
        Does the required preprocessing operations to an image
            :param image_path: Image file path
            :type image: str
            :param size: Size of an image(width, height both have equal value)
            :type size: int

        :rtype: <class 'numpy.ndarray'>
    """
    test_img = cv2.imread(image_path)
    # cv2.imshow('xx', test_img)
    # cv2.waitKey(0)
    test_img = cv2.resize(test_img, dsize=(size, size), interpolation=cv2.INTER_AREA)

    test_img = test_img.reshape((1, size, size, 3)).astype(np.float32)

    return test_img / 255


def get_predictions(image, model_path, labels, num_of_predictions):
    """
        Get predictions from the provided model using provided resource

            :param image: Image file from request
            :type image: <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>
            :param model_path: Path to trained model
            :type model_path: str
            :param labels: Model trained labels
            :type labels: list
            :param num_of_predictions: Number of predicted results needed(Ex: num_of_predictions = 3 means need result of top three predictions)
            :type num_of_predictions: int

        :rtype: object(dict)
    """

    # save image
    file_name = default_storage.save('image.png', image)
    # get path
    file_path = default_storage.path(file_name)

    processed_img = get_processed_input_img(str(file_path))

    loaded_model = load_model(model_path)

    pred = loaded_model.predict(processed_img)
    print('\n\n\n\n', pred, '\n\n\n')

    best = (-pred).argsort()[0]

    sorted_preds = [labels[i] for i in best]

    context = {
        'top_3_predictions': sorted_preds[:num_of_predictions],
        'probabilities': [pred[0][i] for i in best[:num_of_predictions]]
    }

    return context


class FreshStaleFruitAPIView(APIView):
    """ Fresh stale fruit predictions """

    authentication_classes = [JWTTokenUserAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']

        model_path = 'api/trained_models/fresh_stale_model-epoch_mnet.h5'
        predictions = get_predictions(image_file, model_path, FRESH_STALE_LABELS, 3)


        context = {
            'top_3_predictions': predictions['top_3_predictions'],
            'probabilities': predictions['probabilities']
        }

        return Response(context, status=status.HTTP_200_OK)


class FruitClassificationAPIView(APIView):
    """ Fruit classification ['Banana', 'Mango', 'Papaya', 'Watermelon'] """

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']

        model_path = 'api/trained_models/fruit_classification_trained_model.h5'
        predictions = get_predictions(image_file, model_path, FRUIT_CLASSIFICATION_LABELS, 1)

        context = {
            'top_3_predictions': predictions['top_3_predictions'],
            'probabilities': predictions['probabilities']
        }

        return Response(context, status=status.HTTP_200_OK)


class EEdibilityAPIView(APIView):
    """ Edibility classification """

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']

        model_path = 'api/trained_models/fruit_condition_edible_notedible_trained_model.h5'
        predictions = get_predictions(image_file, model_path, EDIBILITY_CLASSIFICATION_LABELS, 1)

        context = {
            'top_3_predictions': predictions['top_3_predictions'],
            'probabilities': predictions['probabilities']
        }

        return Response(context, status=status.HTTP_200_OK)