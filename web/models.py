import os

from django.db import models


class UpscaledImageModel(models.Model):
    title = models.CharField(max_length=80)
    filename = models.CharField(max_length=80)
    lr_image = models.ImageField(upload_to='lr_images')
    hr_image = models.ImageField(blank=True, null=True)

    def __str__(self):
        return self.title

    def delete(self, using=None, keep_parents=False):
        if os.path.isfile(self.hr_image.path):
            os.remove(self.hr_image.path)
        if os.path.isfile(self.lr_image.path):
            os.remove(self.lr_image.path)

        return super(UpscaledImageModel, self).delete()
