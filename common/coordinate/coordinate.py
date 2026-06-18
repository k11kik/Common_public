import numpy as np
from common import display, pytplot
from ._to_mfa import convert_to_mfa, convert_to_mfa_fluct
from .sm2rmlatmlt import _sm2rmlatmlt, _rmlatmlt2sm
from ._to_mbe import convert_to_mbe


def to_mfa(
        varname_mag: str,
        varname_orb: str,
        varname_mag_ambient: str | None = None,
        res=None,
        window_size=None,
        info: bool = True,
        varname_out: str | None = None
):
    """
    Parameters
    -----------
    * res: window second [s] for coverting into MFA

    """
    display.info('Converting to MFA...')
    
    dat_mag = pytplot.get_data(varname_mag)
    dat_orb = pytplot.get_data(varname_orb)
    if dat_mag is None:
        display.error('dat_mag is None')
        return None
    if dat_orb is None:
        display.error('dat_orb is None')
        return None
    
    if res is None and window_size is None:
        display.error('res or window_size must be given.')
        return None
    elif window_size is None and res is not None:
        dt = np.median(np.diff(dat_mag.times))
        window_size = int(res / dt)
    elif window_size is not None and res is None:
        pass
    else:
        display.warning('either res or window_size should be given -> using window_size value')
        pass
    
    if varname_mag_ambient is None:
        dict_mfa = convert_to_mfa(
            dat_mag.times,
            dat_mag.y,
            dat_orb.times,
            dat_orb.y,
            window_size_mfa=window_size,
            window_mfa_sec=res
        )
    
    else:
        dat_mag_ambient = pytplot.get_data(varname_mag_ambient)
        if dat_mag_ambient is None:
            display.error('mag_ambient is None')
            return None
        dict_mfa = convert_to_mfa_fluct(
            dat_mag.times,
            dat_mag.y,
            dat_mag_ambient.times,
            dat_mag_ambient.y,
            dat_orb.times,
            dat_orb.y,
            window_size_mfa=window_size
        )
    
    if dict_mfa is None:
        display.error('dict_mfa is None')
        return None

    # store data
    if varname_out is None:
        varname_out = f'{varname_mag}_mfa'
    pytplot.store_data(varname_out, {'x': dat_mag.times, 'y': dict_mfa['mag_mfa']})
    if varname_mag_ambient is None:
        pytplot.store_data(f'{varname_mag}_ave', {'x': dat_mag.times, 'y': dict_mfa['mag_ave']})
    else:
        pytplot.store_data(f'{varname_mag_ambient}_ave', {'x': dat_mag.times, 'y': dict_mfa['mag_ave']})

    return


def sm2rmlatmlt(
        varname,
        to='rmlatmlt',
        varname_out=None,
):
    dat = pytplot.get_data(varname)
    if dat is None:
        display.warning(f"'{varname}' is None")
        return
    
    valid_to = ['gsm', 'rmlatmlt']
    if not to in valid_to:
        display.warning(f"Invalid 'to': {to}")
        return
    
    data = dat.y
    data_converted = np.zeros_like(data)
    if to == 'gsm':
        data_converted_x, data_converted_y, data_converted_z = _rmlatmlt2sm(data[:, 0], data[:, 1], data[:, 2])
        data_converted[:, 0] = data_converted_x
        data_converted[:, 1] = data_converted_y
        data_converted[:, 2] = data_converted_z
    elif to == 'rmlatmlt':
        data_converted_x, data_converted_y, data_converted_z = _sm2rmlatmlt(data[:, 0], data[:, 1], data[:, 2])
        data_converted[:, 0] = data_converted_x
        data_converted[:, 1] = data_converted_y
        data_converted[:, 2] = data_converted_z
    else:
        display.warning(f'Unsupported type: {to=}')
    
    if varname_out is None:
        varname_out = f'{varname}_{to}'
    pytplot.store_data(varname_out, {'x': dat.times, 'y': data_converted})
    return


def to_mbe(
        varname_trans: str,
        varname_mag: str,
        window_sec,
        varname_out: str | None = None
):
    vars_tplot = pytplot.tplot_names(quiet=True)
    if not varname_trans in vars_tplot:
        display.warning(f'Not found in tplot: {varname_trans}')
        return
    if not varname_mag in vars_tplot:
        display.warning(f'Not found in tplot: {varname_mag}')
        return
    
    dat_trans = pytplot.get_data(varname_trans)
    dat_mag = pytplot.get_data(varname_mag)

    dict_mbe = convert_to_mbe(
        dat_trans.times,
        dat_trans.y,
        dat_mag.times,
        dat_mag.y,
        window_sec=window_sec
    )    
    
    if dict_mbe is None:
        display.error('dict_mbe is None')
        return None

    # store data
    if varname_out is None:
        varname_out = f'{varname_trans}_mbe'
    pytplot.store_data(varname_out, {'x': dat_mag.times, 'y': dict_mbe['values_mbe']})
    pytplot.store_data(f'{varname_mag}_ave', {'x': dat_mag.times, 'y': dict_mbe['mag_ave']})

    return