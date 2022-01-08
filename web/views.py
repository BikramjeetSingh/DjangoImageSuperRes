import os

from django import views
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render

from upscaler.predict import predict
from web.forms import ImageForm


def _get_file_name(f):
    """
    The Django ImageFieldFile.name attribute contains not only the filename but also the folder names after
    settings.MEDIA_ROOT. So for instance, if the image is present in media/lr_images and its name is foo.png, then
    the attribute will return lr_images/foo.png. So we need this function to return only the filename, without
    the folders.
    """
    return os.path.basename(f)


class ImageUpscaleView(views.View):

    def get(self, request, *args, **kwargs):
        return render(request, 'index.html', {'form': ImageForm()})

    def post(self, request, *args, **kwargs):
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()

            img_obj = form.instance
            img_obj.filename = _get_file_name(img_obj.lr_image.name)
            predict(
                img_path=img_obj.lr_image.path,
                output_path=f"{settings.MEDIA_ROOT}/hr_images/{img_obj.filename}",
            )
            img_obj.hr_image = f"hr_images/{img_obj.filename}"
            img_obj.save()

            with open(img_obj.hr_image.path, 'rb') as hr_image:
                response = HttpResponse(hr_image.read(), content_type="image/*")
                response['Content-Disposition'] = 'attachment; filename=' + _get_file_name(img_obj.hr_image.name)

            img_obj.delete()
            return response
