from django.db import models


# Create your models here.
class item_status(models.Model):
    item_code=models.AutoField(primary_key=True,unique=True)
    # item_code=models.CharField(max_length=30,unique=True,primary_key=True,blank=False)
    item_name=models.CharField(max_length=50,blank=True)
    type= models.CharField(max_length=100,blank=True)  
    price=models.FloatField()
    choices =({'AVAILABLE','Item ready to be purchased'},{'SOLD','Item Sold'},{'RESTOCKING','Item restocking in few days'})
    status=models.CharField(max_length=50,choices=choices,default="AVAILABLE") #Available,Sold, Restocking
    item_quantity_available=models.IntegerField()
    issues=models.CharField(max_length=100,default="No issues")
    def __unicode__(self):
        return self.item_code
    # class Meta:
    #     abstract = True

class supplier(models.Model):
    supplier_id=models.AutoField(primary_key=True,unique=True)
    supplier_name=models.CharField(max_length=500)
    supplier_phone=models.CharField(max_length=15)
# class staff_member(models.Model):
#     member_id=models.AutoField(primary_key=True,unique=True)
#     member_username=models.CharField(max_length=20,unique=True)
#     member_password=models.CharField(max_length=20,unique=True)

class sales_record(models.Model):
    record_id=models.AutoField(primary_key=True,unique=True)
    item_quantity_sold=models.IntegerField(null=True)
    item_quantity_before_sales=models.IntegerField(null=True)
    item_quantity_after_sales=models.IntegerField(null=True)
    item_code=models.ForeignKey(item_status,blank=True,null=True, on_delete=models.SET_NULL ,related_name='salesitem')
    date_sold=models.DateField(auto_now_add=True)
    
    # staff=models.ForeignKey(staff_member,blank=False,null=True,on_delete=models.SET_NULL)
    # def __unicode__(self):
    #     return self.item_code, record_id
class reorder(models.Model):
    order_id=models.AutoField(primary_key=True,unique=True)
    date_reorder= models.DateField(blank=True,null=True);
    quantity_reorder=models.IntegerField()
    date_of_receive=models.DateField(blank=True,null=True)
    quantity_receive=models.IntegerField(blank=True,null=True)
    item_code=models.ForeignKey(item_status,blank=False,null=True,on_delete=models.SET_NULL, related_name='reorderitem')
    supplier=models.ForeignKey(supplier,blank=False,null=True,on_delete=models.SET_NULL, related_name='supplier')
    # staff=models.ForeignKey(staff_member,blank=False,null=True,on_delete=models.SET_NULL)
    remarks=models.CharField(max_length=100,default="No issue")


    
class Post(models.Model):
    post=models.TextField()

