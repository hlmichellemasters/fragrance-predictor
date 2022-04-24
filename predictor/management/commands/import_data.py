import json
import os


from django.core.management.base import BaseCommand
from django.conf import settings

from predictor.models import Perfume


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "db", type=str, help="database name without extention")

    def handle(self, **kwargs):
        file_name = kwargs["db"]
        f = file_name + ".json"
        file_path = os.path.join(settings.BASE_DIR, "database", f)
        # f = open("data.json")
        f = open(file_path, encoding="utf-8")

        # returns JSON object as
        # a dictionary
        data = json.load(f)
        for i in data:
            print(i.get('id'))

            Perfume.objects.create(name=i.get('name'),
                                   house=i.get('house'),
                                   house_link=i.get('house_link', None),
                                   description=i.get('description'),
                                   added_date=i.get(
                                       'added_date', '2022-04-24 12:35:20'),
                                   added_by_id=i.get('added_by', 1))

        f.close()
