import os
import sys
import pandas as pd


import django
django.setup()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fragrance_project.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from predictor.models import Perfume
from django.contrib.auth.models import User


def save_perfume_from_row(row):
    perfume = Perfume()
    perfume.name = row[1]
    perfume.house = row[2]
    perfume.description = row[3]
    default_user = User.objects.get(username='michellem')
    perfume.added_by = default_user
    perfume.save()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Reading from file " + str(sys.argv[1]))
        perfumes_df = pd.read_csv(sys.argv[1])
        print(perfumes_df)

        perfumes_df.apply(
            save_perfume_from_row,
            axis=1
        )

        print("There are {} perfumes".format(Perfume.objects.count()))

    else:
        print("Please, provide Perfume file path")

