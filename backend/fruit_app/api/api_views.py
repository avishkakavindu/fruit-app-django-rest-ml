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
import pandas as pd
from sklearn import preprocessing
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
import glob
import os

FRESH_STALE_LABELS = ['fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum', 'fresh_orange', 'fresh_tomato', 'stale_apple', 'stale_banana', 'stale_bitter_gourd', 'stale_capsicum', 'stale_orange', 'stale_tomato']
FRUIT_CLASSIFICATION_LABELS = ['Banana', 'Mango', 'Papaya', 'Watermelon']
EDIBILITY_CLASSIFICATION_LABELS = ['Not_edible', 'edible']
DISEASE_LABELS = ['Anthracnose', 'Black_rot_canker', 'Mold', 'Scab', 'Sooty_blotch']
RIPENED_METHOD_LABELS = ['Artificial', 'Natural']

# contains [model_path, num_of_predictions, labels]
MODEL_PATHS = {
    'fresh_stale_food_model': ['api/trained_models/fresh_stale_model-epoch_mnet.h5', 3, FRESH_STALE_LABELS],
    'fruit_classification_model': ['api/trained_models/fruit_classification_trained_model.h5', 1, FRUIT_CLASSIFICATION_LABELS],
    'disease_model': ['api/trained_models/fruit_diseases_check_trained_model.h5', 3, DISEASE_LABELS],
    'edibility_model': ['api/trained_models/fruit_condition_edible_notedible_trained_model.h5', 1, EDIBILITY_CLASSIFICATION_LABELS],
    'fruit_ripening_model': ['api/trained_models/fruit_ripened_method_trained_model.h5', 1, RIPENED_METHOD_LABELS]
}


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
        'top_n_predictions': sorted_preds[:num_of_predictions],
        'probabilities': [pred[0][i] for i in best[:num_of_predictions]]
    }

    return context


class FruitClassificationAPIView(APIView):
    """ Member 01: Fruit classification ['Banana', 'Mango', 'Papaya', 'Watermelon'] """

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']

        model = MODEL_PATHS['fruit_classification_model']
        model_path = model[0]
        num_of_predictions = model[1]
        predictions = get_predictions(image_file, model_path, FRUIT_CLASSIFICATION_LABELS, num_of_predictions)

        context = {
            'top_n_predictions': predictions['top_n_predictions'],
            'probabilities': predictions['probabilities']
        }

        return Response(context, status=status.HTTP_200_OK)


class DiseasePredictionAPIView(APIView):
    """ Member 01: Disease identification ['Anthracnose', 'Black_rot_canker', 'Mold', 'Scab', 'Sooty_blotch'] """

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']

        model = MODEL_PATHS['disease_model']
        model_path = model[0]
        num_of_predictions = model[1]
        predictions = get_predictions(image_file, model_path, DISEASE_LABELS, num_of_predictions)

        context = {
            'top_n_predictions': predictions['top_n_predictions'],
            'probabilities': predictions['probabilities']
        }

        return Response(context, status=status.HTTP_200_OK)


class EdibilityAPIView(APIView):
    """ Member 01: Edibility classification """

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']

        model = MODEL_PATHS['edibility_model']
        model_path = model[0]
        num_of_predictions = model[1]
        predictions = get_predictions(image_file, model_path, EDIBILITY_CLASSIFICATION_LABELS, num_of_predictions)

        context = {
            'top_n_predictions': predictions['top_n_predictions'],
            'probabilities': predictions['probabilities']
        }

        return Response(context, status=status.HTTP_200_OK)


class RipenedMethodPredictionAPIView(APIView):
    """ Member 01: Ripened method identification ['Artificial', 'Natural'] """

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']

        model = MODEL_PATHS['fruit_ripening_model']
        model_path = model[0]
        num_of_predictions = model[1]
        predictions = get_predictions(image_file, model_path, RIPENED_METHOD_LABELS, num_of_predictions)

        context = {
            'top_n_predictions': predictions['top_n_predictions'],
            'probabilities': predictions['probabilities']
        }

        return Response(context, status=status.HTTP_200_OK)


class GradingAPIView(APIView):
    """ Member 01: Handles the grading based on predictions """

    def get_grading(self, image_file):
        """
            GEt grading for provided image

            :param image_file: Uploaded file for grading
            :type image_file: <class 'django.core.files.uploadedfile.InMemoryUploadedFile'>

            :return: Grade(A, B, C)
            :rtype: str
        """

        predictions = {}

        model_names = [key for key in MODEL_PATHS.keys() if key != 'fruit_classification_model']

        for model in model_names:
            model_path = MODEL_PATHS[model][0]
            num_of_predictions = MODEL_PATHS[model][1]
            labels = MODEL_PATHS[model][2]

            preds = get_predictions(image_file, model_path, labels, num_of_predictions)

            # adding the prediction and probabilities to prediction dict
            predictions[model] = {
                'top_n_predictions': preds['top_n_predictions'],
                'probabilities': preds['probabilities']
            }

        score = []

        fresh_stale_prefix = predictions['fresh_stale_food_model']['top_n_predictions'][0].split('_')[0]
        if fresh_stale_prefix == 'fresh':
            score.append(1)
        diseases = predictions['disease_model']['top_n_predictions']
        diseases_probs = predictions['disease_model']['probabilities']
        ideal_state = 'Healthy'
        if ideal_state in diseases:
            if diseases.index(ideal_state) == 0:
                score.append(1)
            elif diseases.index(ideal_state) in [1, 2] and diseases_probs[diseases.index(ideal_state)] > 0.9:
                score.append(1)
            else:
                score.append(0)
        else:
            score.append(0)

        edible = predictions['edibility_model']['top_n_predictions']
        if 'edible' in edible:
            score.append(1)
        else:
            score.append(0)

        ripen = predictions['fruit_ripening_model']['top_n_predictions']
        if 'Natural' in ripen:
            score.append(1)
        else:
            score.append(0.5)

        # list content score = [fresh_stale, disease, edible, ripen]
        if score.count(0) > 2:
            grade = 'C'
        elif score.count(0) > 1:
            grade = 'B'
        else:
            grade = 'A'

        return grade

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']

        grade = self.get_grading(image_file)

        return Response(grade, status=status.HTTP_200_OK)


class FreshStaleFruitAPIView(APIView):
    """ Member 02: Fresh stale fruit predictions """

    authentication_classes = [JWTTokenUserAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['image']
        temperature = request.POST['temperature']
        humidity = request.POST['humidity']

        model = MODEL_PATHS['fresh_stale_food_model']
        model_path = model[0]
        num_of_predictions = model[1]
        predictions = get_predictions(image_file, model_path, FRESH_STALE_LABELS, num_of_predictions)

        gav = GradingAPIView()
        grading = gav.get_grading(image_file)

        context = {
            'top_n_predictions': predictions['top_n_predictions'],
            'probabilities': predictions['probabilities'],
            'grade': grading
        }

        return Response(context, status=status.HTTP_200_OK)


class FruitPricePredictionAPIView(APIView):
    """ Member 03: Fruit price prediction """

    # OneHotEncoding for nominal columns
    def clean_nominal(self, dataset, columns_nominal):
        """ Method to OneHot Encode columns in dataset selectively"""

        cleaned_dataset = dataset
        onehot_encoder = preprocessing.OneHotEncoder(drop='first')

        for column in columns_nominal:
            X = onehot_encoder.fit_transform(dataset[column].values.reshape(-1, 1)).toarray()
            # create dataframe from encoded data
            dataset_onehot = pd.DataFrame(X, columns=[column + '_' + str(i) for i in range(X.shape[1])])
            # update dataset -> cleaned_dataset
            cleaned_dataset = pd.concat(
                [dataset_onehot.reset_index(drop=True), cleaned_dataset.reset_index(drop=True)],
                axis=1)
            # remove encoded column from dataset
            del cleaned_dataset[column]
        return cleaned_dataset

    # Label Encoding for ordinal columns
    def clean_ordinal(self, dataset, columns_ordinal):
        """ Method to Label Encode columns in dataset selectively"""

        label_encoder = preprocessing.LabelEncoder()
        for column in columns_ordinal:
            dataset[column] = label_encoder.fit_transform(dataset[column])
        return dataset

    def temp_preprocess(self, val, mean):
        """ Encode Temp column value to numerical based on > mean or < mean """
        if val < mean:
            return 0
        return 1

    def post(self, request, *args, **kwargs):
        fruit = request.POST['fruit'].lower()
        season = request.POST['season'].lower()
        month = request.POST['month'].lower()
        temp = request.POST['temperature']
        dhil3m = request.POST['disaster_happen_in_last_3month']
        condition = request.POST['fruit_condition']

        df = pd.read_csv('api/datasets/dataset_price_prediction.csv')
        df.drop('Unnamed: 0', axis=1, inplace=True)

        data_df = pd.DataFrame(
            [[fruit, season, month, temp, dhil3m, condition, None]],
            columns=df.columns
        )
        df = df.append(data_df)

        cleaned_df = self.clean_nominal(df, ['Season', 'Fruit', 'Deasaster Happen in last 3month', 'Fruit condition'])
        cleaned_df = self.clean_ordinal(cleaned_df, ['Month'])

        cleaned_df['Temp'] = pd.to_numeric(cleaned_df['Temp'])

        temp_mean = cleaned_df['Temp'].mean()

        cleaned_df['Temp'] = cleaned_df['Temp'].apply(lambda x: self.temp_preprocess(x, temp_mean))

        x = cleaned_df.drop("Price per kg", axis=1)

        record = x.tail(1)

        model_path = 'api/trained_models/fruit_price_predicter.sav'

        import pickle

        rf = pickle.load(open(model_path, 'rb'))

        preds = rf.predict(
            np.array(record.values.tolist()).reshape(1, -1)
        )[0]

        context = {
            'predicted_price': round(preds, 2)
        }

        return Response(context, status=status.HTTP_200_OK)


class PriceTimeSeriesAPIView(APIView):
    """ Forecast for given time """

    def create_df(self, base_path):
        all_files = glob.glob(os.path.join(base_path, "*.csv"))
        df_from_each_file = []
        for f in all_files:
            tmp_df = pd.read_csv(f)

            # Data seperated by ';' we need to create seperate columns
            tmp_df = pd.DataFrame(
                tmp_df['Date;Price min;Price max'].str.split(';').tolist(),
                columns=['date', 'min_price', 'max_price']
            )
            # add item column
            item = f.strip().split('/')[-1].split('\\')[-1].split('.')[0]
            print('Item: ', f.strip().split('/')[-1].split('\\')[-1].split('.')[0], '\n\n\n\n')
            tmp_df['item'] = item
            df_from_each_file.append(tmp_df)

        df = pd.concat(df_from_each_file, ignore_index=True)

        return df

    def get_forecasts(self, food_item, start_date, end_date, predicted='min_price'):
        model = Prophet(interval_width=0.95)
        df = self.create_df('api/datasets/sale_forecast/')

        df = df.loc[df['item'] == food_item]
        df.drop('item', axis=1, inplace=True)

        # target variable
        if predicted == 'min_price':
            df.drop('max_price', axis=1, inplace=True)
        elif predicted == 'max_price':
            df.drop('min_price', axis=1, inplace=True)
        # rename columns
        df.rename(columns={predicted: 'y', 'date': 'ds'}, inplace=True)
        # dtype to datetime
        df['ds'] = pd.DatetimeIndex(df['ds'])

        # training
        model.fit(df)

        # to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        # get number of days between two dates
        num_of_days = (end_date - start_date).days
        # make list of dates between two dates
        date_list = [start_date + timedelta(days=i) for i in range(num_of_days + 1)]

        print(date_list)
        # dates into df
        future_dates = {'ds': date_list}
        future_dates = pd.DataFrame(data=future_dates)
        # get forecasts
        forecast = model.predict(future_dates)
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

        return forecast

    def post(self, request, *args, **kwargs):
        product = request.POST['product']
        start_date = request.POST['start_date']
        end_date = request.POST['end_date']

        forecast = self.get_forecasts(product, start_date, end_date)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast.rename(
            columns={
                'ds': 'date',
                'yhat': 'predictions',
                'yhat_lower': 'lower_bound',
                'yhat_upper': 'higher_bound'
            }, inplace=True
        )
        context = {
            'detail': forecast
        }
        return Response(context)



