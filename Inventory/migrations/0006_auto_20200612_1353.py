# Generated by Django 3.0.6 on 2020-06-12 05:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inventory', '0005_auto_20200610_2058'),
    ]

    operations = [
        migrations.AlterField(
            model_name='item_status',
            name='price',
            field=models.FloatField(),
        ),
    ]
