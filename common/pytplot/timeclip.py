import logging
import numpy as np
from common import time, display, util
from .time_double import time_double
from ._core import get_data, store_data

def timeclip(
        var_name: str,
        epoch_clip: list,
        new_name: str | None = None,
        replace: bool = False,
        exclude_nan: bool = False,
        mode='absolute'
):
    """
    clip the data by epoch_clip, and exclude nan from the data
    * new_name: default: {var_name}_clip
    * mode: 'absolute', 'wide', 'narrow'
    """
    if mode not in ['absolute', 'wide', 'narrow']:
        display.warning(f'Invalid mode: {mode}')
        return
    dat_var = get_data(var_name)
    if dat_var is None:
        display.info('No data to clip')
        return False
    
    if var_name == new_name:
        replace = True
        
    times, dat = dat_var.times, dat_var.y

    times_clip_s, times_clip_e = time.convert(epoch_clip, frm='str', into='unix')

    if mode == 'absolute':
        mode_s = 'absolute'
        mode_e = 'absolute'
    elif mode == 'wide':
        mode_s = 'under'
        mode_e = 'over'
    elif mode == 'narrow':
        mode_s = 'over'
        mode_e = 'under'
    
    ids = util.get_closest_idx(times, times_clip_s, mode=mode_s)
    ide = util.get_closest_idx(times, times_clip_e, mode=mode_e)

    # ids = np.abs(times - times_clip_s).argmin()
    # ide = np.abs(times - times_clip_e).argmin()

    # ids = np.abs(times - time_double(epoch_clip[0])).argmin()
    # ide = np.abs(times - time_double(epoch_clip[1])).argmin()

    # ids = np.searchsorted(times, times_clip_s, side='left')
    # ide = np.searchsorted(times, times_clip_e, side='right')

    times_clipped = times[ids:ide]

    times_range = time.convert([times[0], times[-1]], frm='unix', into='str')

    if len(times_clipped) == 0:
        display.warning(f'No clipped data for {var_name}: {times_range=}, {epoch_clip=}')
        return False

    # if new_name == var_name:
    #     replace = True
    # else:
    #     replace = False

    if not hasattr(dat, 'v'):
        dat_clipped = dat[ids:ide]
        if new_name is None:
            new_name = f'{var_name}_clip'
        if exclude_nan:
            if dat_clipped.ndim == 1:
                idx_not_nan = ~np.isnan(dat_clipped)
                times_exnan = times_clipped[idx_not_nan]
                dat_exnan = dat_clipped[idx_not_nan]
            else:
                idx_not_nan = ~np.isnan(dat_clipped)[:, 0]
                times_exnan = times_clipped[idx_not_nan]
                dat_exnan = dat_clipped[idx_not_nan, :]
            if len(times_exnan) != 0:
                store_data(new_name, {'x': times_exnan, 'y': dat_exnan}, replace=replace)
        else:
            if len(times[ids:ide]) != 0:
                store_data(new_name, {'x': times[ids:ide], 'y': dat_clipped}, replace=replace)

    else:
        dat_clipped = dat[ids:ide, :]
        if new_name is None:
            new_name = f'{var_name}_clip'
        if exclude_nan:
            idx_not_nan = ~np.isnan(dat_clipped)[:, 0]
            times_exnan = times_clipped[idx_not_nan]
            dat_exnan = dat_clipped[idx_not_nan, :]
            if len(times_exnan) != 0:
                store_data(new_name, {'x': times_exnan, 'y': dat_exnan, 'v': dat_var.v}, replace=replace)
        else:
            if len(times[ids:ide]) != 0:
                store_data(new_name, {'x': times[ids:ide], 'y': dat_clipped, 'v': dat_var.v}, replace=replace)

    return True
