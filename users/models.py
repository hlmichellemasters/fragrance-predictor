from django.db import models
from django.contrib.auth.models import User
from pandas import DataFrame
from predictor.models import Preference


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='profile_pics/default.jpg', upload_to='profile_pics')

    def __str__(self):
        return f'{self.user.username} Profile'

    def preference_dataframe(self) -> DataFrame:
        # Get all review data, as a Pandas DataFrame object, for a given user
        return DataFrame.from_records(Preference.objects.filter(user=self.user).values('user_id', 'perfume_id', 'love'))



