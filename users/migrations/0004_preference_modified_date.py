# Generated by Django 4.0.3 on 2022-04-04 21:35

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_preference'),
    ]

    operations = [
        migrations.AddField(
            model_name='preference',
            name='modified_date',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
