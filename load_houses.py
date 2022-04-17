import os
import sys
import pandas as pd
import django

django.setup()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fragrance_project.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from predictor.models import Perfume, House

# This script loads Perfumes from a csv file.
# Current set-up is to run "python load_perfumes.py staticfiles/perfumes.csv"


def save_house_from_row(row):
    house = House()
    house.name = row[1]
    house.save()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Reading from file " + str(sys.argv[1]))
        perfumes_df = pd.read_csv(sys.argv[1])
        print(perfumes_df)

        perfumes_df.apply(
            save_house_from_row,
            axis=1
        )

        print("There are {} houses ".format(House.objects.count()))

    else:
        print("Please, provide Perfume file path")

