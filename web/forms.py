from django import forms

from web.models import UpscaledImageModel


class ImageForm(forms.ModelForm):

    class Meta:
        model = UpscaledImageModel
        fields = ('title', 'lr_image')
