from django.db import models


class Image(models.Model):
    """ keeps images """
    image = models.ImageField(upload_to='images')

