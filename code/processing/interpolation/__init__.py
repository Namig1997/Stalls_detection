from .custom import InterpolatorCustom
from .regulargrid import InterpolatorRegularGrid

__interpolators__ = {
    "custom": InterpolatorCustom,
    "regulargrid": InterpolatorRegularGrid,
}

def get(name):
    return __interpolators__[name]