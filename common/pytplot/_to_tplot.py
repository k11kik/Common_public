from common import display
from ._core import store_data

def dict_to_tplot(
        dict_data,
        var_time,
        vars_to_tplot,
        prefix=''
):
    if not isinstance(dict_data, dict):
        display.warning('Input data must be dict')
        return
    
    keys = dict_data.keys()
    if var_time in keys:
        times = dict_data[var_time]
    else:
        display.warning(f'{var_time} is not in data')
        return
    
    if not isinstance(vars_to_tplot, list):
        vars_to_tplot = [vars_to_tplot]
    
    for i, var_to_tplot in enumerate(vars_to_tplot):
        if var_to_tplot in keys:
            store_data(f'{prefix}{var_to_tplot}', {'x': times, 'y': dict_data[var_to_tplot]})

    return