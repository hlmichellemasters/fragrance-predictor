from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse


class Perfume(models.Model):
    name = models.CharField(max_length=100)
    house = models.CharField(max_length=100)
    description = models.TextField()
    added_date = models.DateTimeField(default=timezone.now)
    added_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name + ' by ' + self.house

    def get_absolute_url(self):
        return reverse('perfume-detail', kwargs={'pk':self.pk})





















































