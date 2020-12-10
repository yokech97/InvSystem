from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import *
from django.contrib.auth.models import User
import pandas as pd
selection=item_status.objects.all().values_list('item_code',flat=True)
# sales=pd.DataFrame.from_records(sales_record.objects.filter(item_code=data).values('item_code_id'))
choice=[('','Select an Item Code')]
for x in selection:
    choice.append(tuple([x,x])) 
# choice=[('a','a'),('b','b'),('c','c'),('d','d')]
class HomeForm(forms.ModelForm):
    # post = forms.CharField(widget=forms.TextInput(
    #     attrs={
    #         'class': 'form-control',
    #         'placeholder': 'Input an Item Code Value'
    #     }
    # ))
    post= forms.CharField(widget=forms.Select(choices=choice))
    # post = forms.ChoiceField(choices=choice)
    class Meta:
        model = Post
        fields = ('post',)



class item_statusForm(forms.ModelForm):
    class Meta:
        model = item_status
        fields = ('item_code', 'item_name', 'type', 'retail_price','stock_price','status','item_quantity_available','issues',)


class SupplierForm(forms.ModelForm):
    class Meta:
        model = supplier
        fields = ('supplier_id', 'supplier_name', 'supplier_phone',)
        

class StaffForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'password')

class Sales_recordForm(forms.ModelForm):
    class Meta:
        model = sales_record
        fields = ('record_id', 'item_quantity_sold', 'item_code',)

class ReorderForm(forms.ModelForm):
    class Meta:
        model = reorder
        fields = ('order_id', 'date_reorder', 'quantity_reorder', 'date_of_receive','quantity_receive','item_code','supplier','remarks',)
