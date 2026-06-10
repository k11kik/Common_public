from common import pytplot, display, mathpy


def moving_average(
        varname: str,
        varname_out: str | None = None,
        window_size=256
):
    dat = pytplot.get_data(varname)
    if dat is None:
        display.error('data_process/moving_average', 'dat is None')
        return None
    averaged_data = mathpy.moving_average_vec(dat.y, window_size=window_size)

    # store
    if varname_out is None:
        varname_out = f'{varname}_ave'
    pytplot.store_data(varname_out, {'x': dat.times, 'y': averaged_data})
    return
