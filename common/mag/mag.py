import numpy as np
from common import pytplot, mathpy, quant, display


def normalize_vec(
        varname_vec,
        newname,
):
    if not pytplot.exist_vars(varname_vec):
        return
    
    dat_vec = pytplot.get_data(varname_vec)
    times = dat_vec.times
    comp_vec = dat_vec.y
    if comp_vec.ndim == 1:
        display.warning('ndim must be larger than 1')
        return
    elif comp_vec.ndim == 2:
        norm_value = np.linalg.norm(comp_vec, axis=1)
        pytplot.store_data(newname, {'x': times, 'y': norm_value})
    return


def calculate_fcp(
        varname_mag_norm,
        varname_ref_times=None,
        average_window_sec=10,
        newname='fcp'
):
    if not pytplot.exist_vars(varname_mag_norm):
        return
    dat_mag_norm = pytplot.get_data(varname_mag_norm)
    times, mag_norm = dat_mag_norm.times, dat_mag_norm.y
    mag_norm_ave = mathpy.moving_average_by_time(times, mag_norm, average_window_sec)
    fcp = quant.cyclotron_frequency(1, mag_norm_ave * 1e-9)
    pytplot.store_data(newname, {'x': times, 'y': fcp})

    if not varname_ref_times is None:
        if not pytplot.exist_vars(varname_ref_times):
            return
        pytplot.interp(varname_ref_times, newname, newname=newname)
    return
