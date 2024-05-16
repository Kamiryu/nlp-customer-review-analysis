import pandas as pd
import ast as ast
import numpy as np
from typing import *


def extract_weekend_value(hours_dictionary:Dict) -> bool:
    days =  hours_dictionary.keys()
    if 'Saturday' in days  or 'Sunday' in days :
        return True
    return False

def extract_hours_value(hours_dictionary:Dict) -> List:
    values = dict(zip(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], list([-1, -1]  for i in range (7))))
    
    for (k,v) in hours_dictionary.items():
        vals = v.split('-')
        vals_int = [ int(x.split(':')[0]) + (int(x.split(':')[1])/60) for x in vals]
        values[k] = vals_int

    return list(values.values())


def extract_attributes_value(attr_dictionary: Dict) -> List:
    values = []
    for (k, v) in attr_dictionary.items():
        if v:
            values.append(k)
    return values
