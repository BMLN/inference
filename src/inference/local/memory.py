from .models import Model

from functools import wraps
from types import MethodType











def on_demand(model):
    assert isinstance(model, Model)
    
    def decorator(model, method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            result = method(*args, **kwargs)
            del model.model
            model.model = None
            return result
        return wrapped


    func = getattr(type(model), "inference")
    func = decorator(model, func)
    binding = MethodType(func, model)
    setattr(model, "inference", binding)
    

    return model

                