# Generated by Django 3.0.6 on 2020-12-10 10:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inventory', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='item_status',
            old_name='price',
            new_name='retail_price',
        ),
        migrations.AddField(
            model_name='item_status',
            name='stock_price',
            field=models.FloatField(null=True),
        ),
    ]
