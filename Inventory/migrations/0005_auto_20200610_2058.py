# Generated by Django 3.0.6 on 2020-06-10 12:58

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('inventory', '0004_auto_20200610_2057'),
    ]

    operations = [
        migrations.AlterField(
            model_name='reorder',
            name='item_code',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='reorderitem', to='inventory.item_status'),
        ),
        migrations.AlterField(
            model_name='sales_record',
            name='item_code',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='salesitem', to='inventory.item_status'),
        ),
    ]
