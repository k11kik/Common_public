import numpy as np
from common import pytplot, display
from ._base import xyz_to_polar, polar_to_xyz, rmlatmlt_to_polar, polar_to_rmlatmlt
from ._plot_orbit import plot_orbit, plot_orbit_polar, plot_orbit_rmlatmlt, plot_orbit_rmlatmlt_itself


def xyz2polar(
        varname,
        varname_out: str | None = None,
        to = 'polar', # 'xyz' or 'polar'
        info: bool = False,
):
    # check if valid
    # ---------------
    list_to = ['xyz', 'polar']
    if not to in list_to:
        display.error('orbit/xyz2polar', f'Invalid name: {to=}. Should be used: {list_to}')
        return None
    # ---------------

    if info:
        if to == 'polar':
            head_comment = 'Converting xyz to polar...'
        else:
            head_comment = 'Converting polar to xyz...'
        display.current_time_comment(comment=head_comment)

    dat = pytplot.get_data(varname)
    if dat is None:
        display.error('orbit/xyz2polar', 'dat is None')
        return None

    data = dat.y
    if data.ndim != 2 or data.shape[1] != 3:
        display.error('orbit/xyz2polar', 'Invalid data shape')
        return None
    
    data_x = data[:, 0]
    data_y = data[:, 1]
    data_z = data[:, 2]

    if to == 'polar':
        data_x_converted, data_y_converted, data_zconverted = xyz_to_polar(data_x, data_y, data_z)
    elif to == 'xyz':
        data_x_converted, data_y_converted, data_zconverted = polar_to_xyz(data_x, data_y, data_z)
    else:
        display.error(f'Invalid name: {to=}')
        return None
    
    data_converted = np.stack([data_x_converted, data_y_converted, data_zconverted], axis=1)

    # store
    if varname_out is None:
        varname_out = varname + '_' + to
    
    pytplot.store_data(varname_out, {'x': dat.times, 'y': data_converted})
    return


def rmlatmlt2polar(
        varname: str,
        varname_out: str | None = None,
        to='polar',  # 'rmlatmlt', 'polar'
):
    # check
    # --------
    valid_to = ['rmlatmlt', 'polar']
    if not to in valid_to:
        display.error(f'Invalid name: {to=}')
        return None
    
    # main
    dat = pytplot.get_data(varname)
    if dat is None:
        display.error('dat is None')
        return None

    data = dat.y
    if data.ndim != 2 or data.shape[1] != 3:
        display.error('orbit/xyz2polar', 'Invalid data shape')
        return None
    
    data_x = data[:, 0]
    data_y = data[:, 1]
    data_z = data[:, 2]

    if to == 'polar':
        data_x_converted, data_y_converted, data_zconverted = rmlatmlt_to_polar(data_x, data_y, data_z)
    elif to == 'rmlatmlt':
        data_x_converted, data_y_converted, data_zconverted = polar_to_rmlatmlt(data_x, data_y, data_z)
    else:
        display.warning('orbit/rmlatmlt2polar', 'to=rmlatmlt is not currently supported')
        return None
    
    data_converted = np.stack([data_x_converted, data_y_converted, data_zconverted], axis=1)

    # store
    if varname_out is None:
        varname_out = varname + '_' + to
    
    pytplot.store_data(varname_out, {'x': dat.times, 'y': data_converted})
    return


def plot(
        varname: str,
        type: str = 'xyz',
        savefig: str | None = None,
        suptitle: str = 'orbit plot'
):
    """
    Params
    -----
    * type: 'xyz', 'polar', 'rmlatmlt', 'rmlatmlt_itself'
    """
    # check
    # ------------
    valid_type = ['xyz', 'polar', 'rmlatmlt', 'rmlatmlt_itself']
    if not type in valid_type:
        display.warning(f'Invalid type: {type}. Should be used: {valid_type}')
    
    # main
    dat = pytplot.get_data(varname)
    if dat is None:
        display.warning('dat is None')
        return None
    
    orb = dat.y
    if orb.ndim != 2 or orb.shape[1] != 3:
        display.warning('Invalid data shape')
        return None
    
    if type == 'xyz':
        plot_orbit(
            orb,
            savefig=savefig, 
            suptitle=suptitle
        )
    
    elif type == 'polar':
        plot_orbit_polar(
            orb,
            savefig=savefig,
            suptitle=suptitle
        )
    
    elif type == 'rmlatmlt':
        plot_orbit_rmlatmlt(
            orb,
            savefig=savefig,
            suptitle=suptitle
        )
    
    elif type == 'rmlatmlt_itself':
        plot_orbit_rmlatmlt_itself(
            orb,
            savefig=savefig,
            suptitle=suptitle
        )


    return
