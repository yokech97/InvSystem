from django.conf.urls import url
from .views import *
from django.contrib.auth import views as auth_views
urlpatterns = [
    url(r'^$', index, name='index'),
    url(r'^item$', display_item_status, name="display_item_status"),
    url(r'^supplier$', display_supplier, name="display_supplier"),
    url(r'^staff$', display_staff_member, name="display_staff_member"),
    url(r'^sales$', display_sales_record, name="display_sales_record"),
    url(r'^reorder$', display_reorder, name="display_reorder"),

    url(r'^add_item$', add_item_status, name="add_item"),
    url(r'^add_supplier$', add_supplier, name="add_supplier"),
    # url(r'^add_staff_member$', add_staff_member, name="add_staff_member"),
    url(r'^add_sales_record$', add_sales_record, name="add_sales_record"),
    url(r'^add_reorder$', add_reorder, name="add_reorder"),

    url(r'^item/edit_item/(?P<pk>\d+)$', edit_item_status, name="edit_item_status"),
    url(r'^supplier/edit_item/(?P<pk>\d+)/$', edit_supplier, name="edit_supplier"),
    # url(r'^staff/edit_item/(?P<pk>\d+)/$', edit_staff_member, name="edit_staff_member"),
    url(r'^sales/edit_item/(?P<pk>\d+)/$', edit_sales_record, name="edit_sales_record"),
    url(r'^reorder/edit_item/(?P<pk>\d+)/$', edit_reorder, name="edit_reorder"),

    url(r'^item/delete/(?P<pk>\d+)/$', delete_item_status, name="delete_item_status"),
    url(r'^supplier/delete/(?P<pk>\d+)/$', delete_supplier, name="delete_supplier"),
    # url(r'^staff/delete/(?P<pk>\d+)/$', delete_staff_member, name="delete_staff_member"),
    url(r'^sales/delete/(?P<pk>\d+)/$', delete_sales_record, name="delete_sales_record"),
    url(r'^reorder/delete/(?P<pk>\d+)/$', delete_reorder, name="delete_reorder"),
    url(r'^inv/login/$', auth_views.LoginView.as_view(template_name= 'inv/login.html'), name='login'),
    url(r'^inv/logout/$', auth_views.LogoutView.as_view(template_name= 'inv/logout.html'), name='logout'),
    url(r'^register/$',register, name='register'),
    url(r'^password/$',change_password, name='change_password'),

]
