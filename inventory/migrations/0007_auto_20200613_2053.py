# Generated by Django 3.0.6 on 2020-06-13 12:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Inventory', '0006_auto_20200612_1353'),
    ]

    operations = [
        migrations.AlterField(
            model_name='supplier',
            name='supplier_name',
            field=models.CharField(max_length=500),
        ),
    ]
