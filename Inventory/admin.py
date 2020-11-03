from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import *
# Register your models here.

# admin.site.register(item)
@admin.register(item_status,supplier, sales_record,reorder)
class ViewAdmin(ImportExportModelAdmin):
    pass
