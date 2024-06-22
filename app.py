from shipment.exception import ShippingException
import sys

try:
    1/0
except Exception as e:
    raise ShippingException(e,sys)