import os
from common import time, display, cdf, pytplot


def get_cdf_filepath(
        trange,
        basedir_cdf
):
    trange_list = time.make_time_list(trange, 1, 'months', getdata=True)
    cdf_filepaths = []
    for i, trange_i in enumerate(trange_list):
        dt_start = time.convert(trange_i[0], frm='str', into='datetime')
        year = dt_start.year
        month = dt_start.month
        cdf_filepath = os.path.join(
            basedir_cdf,
            f'{year:04}',
            f'dst_index_{year:04}{month:02}.cdf'
        )
        if not os.path.exists(cdf_filepath):
            display.warning(f'Not found: {cdf_filepath}')
            continue
        cdf_filepaths.append(cdf_filepath)
    return cdf_filepaths


def getdata(
        trange,
        basedir_cdf
):
    """
    Variables
    * dst_index: (n_times, )
    """
    cdf_filepaths = get_cdf_filepath(trange, basedir_cdf)
    vars = [
        'times',
        'dst_index'
    ]
    dict_data = cdf.read_and_combine_cdf_files(cdf_filepaths, vars)
    if dict_data is None:
        display.warning('No dst data')
        return
    
    pytplot.store_data('dst_index', {'x': dict_data['times'], 'y': dict_data['dst_index']})
    ret = pytplot.timeclip('dst_index', trange, new_name='dst_index', replace=True)
    if not ret:
        display.warning('No clipped data -> trange is recommended at least for 1 hour. Dst index is 1-hour resolution')
        pytplot.del_data('dst_index')
    return

