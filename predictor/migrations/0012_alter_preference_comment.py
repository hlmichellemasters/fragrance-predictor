# Generated by Django 4.0.3 on 2022-04-20 00:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0011_auto_20220417_1830'),
    ]

    operations = [
        migrations.AlterField(
            model_name='preference',
            name='comment',
            field=models.CharField(max_length=200, null=True),
        ),
    ]
