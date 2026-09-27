import numpy as np
import re

#------------------------ Dimensional verification ------------------------
def dimensional_verification(individual, dict_of_dimension, num_units):

    def as_dim_vec(x):
        if isinstance(x, bool):
            return x
        elif isinstance(x, (int, float, np.integer, np.floating)):
            return [0] * num_units
        return x
    
    def my_add(*args):
        if any(isinstance(arg, bool) for arg in args):
            return False
        args = [as_dim_vec(a) for a in args]
        if all(arg == args[0] for arg in args):
            return args[0]
        return False

    def my_sub(a,b):    
        if isinstance(a,bool) or isinstance(b,bool):
            return False
        a = as_dim_vec(a)
        b = as_dim_vec(b)
        if a == b:
            return a
        return False

    def my_mul(a,b):
        if isinstance(a,bool) or isinstance(b,bool):
            return False
        a = as_dim_vec(a)
        b = as_dim_vec(b)   
        return [x + y for x, y in zip(a, b)]

    def my_protected_div(a,b):     
        if isinstance(a,bool) or isinstance(b,bool):
            return False
        a = as_dim_vec(a)
        b = as_dim_vec(b)
        return [x - y for x, y in zip(a, b)]

    ns = {
        'my_add': my_add,
        'my_sub': my_sub,
        'my_mul': my_mul,
        'my_protected_div': my_protected_div,
        'linker_add': my_add,  
    }
    ns.update(dict_of_dimension)
 
    term_dims = []
    for term in individual:
        expr = term.__str__().replace('\t','').replace('\n','')
        expr = re.sub(r'(?<!\w)(?<!linker_)add\(', 'my_add(', expr)
        expr = re.sub(r'(?<!\w)(?<!linker_)sub\(', 'my_sub(', expr)
        expr = re.sub(r'(?<!\w)(?<!linker_)mul\(', 'my_mul(', expr)
        expr = re.sub(r'(?<!\w)(?<!linker_)protected_div\(', 'my_protected_div(', expr) 
        dim = eval(expr, {}, ns)
        dim = as_dim_vec(dim)
        term_dims.append(dim)         
    return term_dims



