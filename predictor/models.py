from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse


class House(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class Perfume(models.Model):
    name = models.CharField(max_length=100)
    house = models.CharField(max_length=100)
    house_link = models.ForeignKey(House, related_name='perfumes', null=True, on_delete=models.CASCADE)
    description = models.TextField()
    added_date = models.DateTimeField(default=timezone.now)
    added_by = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['name', 'house'], name='house_perfume_unique')
        ]

    def __str__(self):
        return self.name + ' by ' + self.house

    def get_absolute_url(self):
        return reverse('perfume-detail', kwargs={'pk':self.pk})


class Preference(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    perfume = models.ForeignKey(Perfume, on_delete=models.CASCADE)
    love = models.BooleanField()
    comment = models.CharField(max_length=200)
    review_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        love_string = ""
        if self.love:
            love_string = " loves "
        else:
            love_string = " doesn't love "
        return self.user.username + love_string + self.perfume.name






















































