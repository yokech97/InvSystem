from django.shortcuts import render
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import update_session_auth_hash
from django.db.models import F, Sum
from django.contrib.auth.decorators import login_required
from django.db import transaction

# Create your views here.

from inventory.models import *

from inventory.forms import *


@login_required
def display_item_status(request):
    items = item_status.objects.all()
    itemlist=['item_code','item_name','type','price','status','item_quantity_available','issues']
    context={
        'list' : itemlist,
        'header': 'Item Status',
        'items1' :items,
    }

    return render(request,'inv/index.html',context)
@login_required
def display_supplier(request):
    items = supplier.objects.all()
    itemlist=['supplier_id','supplier_name','supplier_phone']
    context={
        'list' : itemlist,
        'header': 'Supplier',
        'items2' :items,
    }

    return render(request,'inv/index.html',context)
@login_required
def display_staff_member(request):
    itemlist=['id','username','password']
    items = User.objects.all()
    context={
        'list':itemlist,
        'header': 'Staff Member',
        'items3' :items,
    }

    return render(request,'inv/index.html',context)
@login_required
def display_sales_record(request):
  
        itemlist=['record_id','item_quantity_before_sales','item_quantity_sold','item_quantity_after_sales','date_sold','item_code','item_quantity_available']
        items = sales_record.objects.all().values('record_id','item_quantity_before_sales','item_quantity_sold','item_quantity_after_sales','date_sold','item_code','item_code__item_quantity_available')
        context={
        'list':itemlist,
        'header': 'Sales Record',
        'items4' : items,
  
    }

        return render(request,'inv/index.html',context)
@login_required
def display_reorder(request):
    itemlist=['order_id','date_reorder','quantity_reorder','date_of_receive','quantity_receive','item_code','supplier','remarks','item_quantity_available']
    items = reorder.objects.select_related().values('order_id','date_reorder','quantity_reorder','date_of_receive','quantity_receive','item_code','supplier','remarks','item_code__item_quantity_available')
    context={
        'list':itemlist,
        'header': 'Reorder',
        'items5' :items,
 
    }

    return render(request,'inv/index.html',context)
@login_required
def add_item(request, cls):
    if request.method == "POST":
        form = cls(request.POST)

        if form.is_valid():
            if form.save():
                return redirect('/', messages.success(request, 'Order was successfully created.', 'alert-success'))
            else:
                return redirect('/', messages.error(request, 'Data is not saved', 'alert-danger'))
        else:
            return redirect('/', messages.error(request, 'Form is not valid', 'alert-danger'))

    else:
        form = cls()
        if cls==item_statusForm:
            return render(request, 'inv/add_new.html', {'form' : form, 'header':'Item Status',})
        if cls==SupplierForm:
            return render(request, 'inv/add_new.html', {'form' : form, 'header':'Supplier',})
        if cls==Sales_recordForm:
            return render(request, 'inv/add_new.html', {'form' : form, 'header':'Sales Record',})


@login_required
def add_item_status(request):
    return add_item(request,item_statusForm)

@login_required
def add_supplier(request):
    return add_item(request,SupplierForm)


# def add_staff_member(request):
#         if request.method =='POST':
#             form = UserCreationForm(request.POST)
#             if form.is_valid():
#                 form.save()
#                 username = form.cleaned_data.get('username')
#                 messages.success(request,f'Account Successfully Created for {username}!')
#                 return redirect('inv/index.html')
#         else:
#             form = UserCreationForm()
#         return render(request,'inv/add_new.html',{'form':form})    
  
@login_required
def add_sales_record(request):
    return add_item(request,Sales_recordForm)

@login_required
def add_reorder(request):
    if request.method == "POST":
        form = ReorderForm(request.POST)
        if form.is_valid():
            if form.save():
                return redirect('/', messages.success(request, 'Order was successfully created.', 'alert-success'))
            else:
                return redirect('/', messages.error(request, 'Data is not saved', 'alert-danger'))
        else:
            return redirect('/', messages.error(request, 'Form is not valid', 'alert-danger'))
    else:
        form = ReorderForm()
        return render(request, 'inv/add_new_reorder.html', {'form':form, 'header':'Re-order',})

@login_required
def edit_item(request, pk, model, cls):
    item = get_object_or_404(model, pk=pk)
    if cls== Sales_recordForm:
        # itemss=sales_record.objects.select_related().get(pk=pk)
        # iterm=getattr(item,'item_code')
        before=item.item_quantity_sold
    if request.method == "POST":
        form = cls(request.POST, instance=item)
        if form.is_valid():    
            
            form.save()
     
            if cls==Sales_recordForm:
                with transaction.atomic():
                    af= sales_record.objects.get(pk=pk)
                    after=af.item_quantity_sold
                    x= item_status.objects.get(item_code=getattr(af.item_code,'item_code'))
                    new=x.item_quantity_available
                    if before>after:
                        value=before-after
                        x.item_quantity_available=new+value
                        x.save()
                    else:
                        value=after-before
                        x.item_quantity_available=new-value
                        x.save()
                return redirect('/sales', messages.success(request, 'Order was successfully updated.', 'alert-success'))
                
                
            if cls==SupplierForm:
                return redirect('/supplier', messages.success(request, 'Order was successfully updated.', 'alert-success'))
            if cls==item_statusForm:
                return redirect('/item', messages.success(request, 'Order was successfully updated.', 'alert-success'))
    else:
        form = cls(instance=item)

        return render(request, 'inv/edit_item.html', {'form': form})


@login_required
def edit_item_status(request, pk):
    return edit_item(request, pk, item_status, item_statusForm)

@login_required
def edit_supplier(request, pk):
    return edit_item(request, pk, supplier, SupplierForm)

# def edit_staff_member(request, pk):
#     if request.method == 'POST':
#         form = PasswordChangeForm(request, request.POST)
#         if form.is_valid():
#             user = form.save()
#             update_session_auth_hash(request, user)  # Important!
#             messages.success(request, 'Your password was successfully updated!')
#             return redirect('change_password')
#         else:
#             messages.error(request, 'Please correct the error below.')
#     else:
#         form = PasswordChangeForm(request.user)




    # return edit_item(request, pk, staff_member, StaffForm)
@login_required
def edit_sales_record(request, pk):
    return edit_item(request, pk, sales_record, Sales_recordForm)

@login_required
def edit_reorder(request, pk):
    item = get_object_or_404(reorder, pk=pk)
    before=item.quantity_receive
    if request.POST:
        form = ReorderForm(request.POST, instance=item)
        if form.is_valid():
            if form.save():
                
                with transaction.atomic():
                    af= reorder.objects.get(pk=pk)
                    after=af.quantity_receive
                    x= item_status.objects.get(item_code=getattr(af.item_code,'item_code'))
                    new=x.item_quantity_available
                    if before>after:
                        value=before-after
                        x.item_quantity_available=new+value
                        x.save()
                    if before==None:
                        x.item_quantity_available=new+after
                        x.save()
                    if before<after:
                        value=after-before
                        x.item_quantity_available=new-value
                        x.save()
                return redirect('/reorder', messages.success(request, 'Order was successfully updated.', 'alert-success'))
            else:
                return redirect('/reorder', messages.error(request, 'Data is not saved', 'alert-danger'))
        else:
            return redirect('/reorder', messages.error(request, 'Form is not valid', 'alert-danger'))
    else:
        form = ReorderForm(instance=item)
        return render(request, 'inv/edit_itemreorder.html', {'form':form})
    


def delete_item_status(request, pk):

    template = 'inv/index.html'
    item_status.objects.filter(item_code=pk).delete()

    items = item_status.objects.all()

    context = {
        'items': items,
    }

    return render(request, template, context)


def delete_supplier(request, pk):

    template = 'inv/index.html'
    supplier.objects.filter(supplier_id=pk).delete()

    items = supplier.objects.all()

    context = {
        'items': items,
    }

    return render(request, template, context)


# def delete_staff_member(request, pk):

#     template = 'inv/index.html'
#     User.objects.filter(id=pk).delete()

#     items = User.objects.all()

#     context = {
#         'items': items,
#     }

#     return render(request, template, context)


def delete_sales_record(request, pk):

    template = 'inv/index.html'
    sales_record.objects.filter(record_id=pk).delete()

    items = sales_record.objects.all()

    context = {
        'items': items,
    }

    return render(request, template, context)

def delete_reorder(request, pk):

    template = 'inv/index.html'
    reorder.objects.filter(order_id=pk).delete()

    items = reorder.objects.all()

    context = {
        'items': items,
    }

    return render(request, template, context)

 
@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, 'Your password was successfully updated!')
            return redirect('change_password')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        form = PasswordChangeForm(request.user)
    
    
    return render(request, 'inv/change_password.html', {'form': form})
def register (request):
    if request.method =='POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request,f'Account Successfully Created for {username}!')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request,'inv/register.html',{'form':form})


