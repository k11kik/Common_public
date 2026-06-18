"""
cdf > pytplot.py
"""
import os
from . import cdfdata
from .cdfdata import cdffile_to_dict
from common import pytplot, display


def to_pytplot_single_var(
        cdffile, 
        var_times, 
        var,
        var_freqs=None,
        store_name=None
    ):
    if store_name is None:
        store_name = var
    times = cdfdata.get_data(cdffile, var_times)
    var_data = cdfdata.get_data(cdffile, var)
    if var_freqs is None:
        pytplot.store_data(store_name, {'x': times, 'y': var_data})
    else:
        freqs = cdfdata.get_data(cdffile, var_freqs)
        pytplot.store_data(store_name, {'x': times, 'y': var_data, 'v': freqs})
    return

