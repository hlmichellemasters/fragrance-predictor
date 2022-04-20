import os
import sys
import pandas as pd
import django


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fragrance_project.settings')
django.setup()

os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from predictor.models import Perfume, Preference
from django.contrib.auth.models import User

# This script loads Perfumes from a csv file.
# Current set-up is to run "python load_perfumes.py staticfiles/perfumes.csv"


def save_preference_from_row(row):
    preference = Preference()
    print("row: " + str(row[0]))
    preference.user = User.objects.get(username=row[1])
    preference.perfume = Perfume.objects.get(name=row[2], house=row[3])
    preference.love = row[5]
    preference.save()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Reading from file " + str(sys.argv[1]))
        preferences_df = pd.read_csv(sys.argv[1])

        print(preferences_df)

        preferences_df.apply(
            save_preference_from_row,
            axis=1
        )

        print("There are now {} preferences".format(Preference.objects.count()))

    else:
        print("Please, provide preference file path")

