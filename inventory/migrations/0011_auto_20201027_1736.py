# Generated by Django 3.0.6 on 2020-10-27 09:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inventory', '0010_post'),
    ]

    operations = [
        migrations.AlterField(
            model_name='supplier',
            name='supplier_name',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='supplier',
            name='supplier_phone',
            field=models.TextField(),
        ),
    ]
