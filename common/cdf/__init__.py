from . import (
    # cdf, # to be deleted
    cdfdata,
    # _to_pytplot
)

from .cdfdata import (
    read_and_combine_cdf_files,
    get_data,
    # get_cdf_to_read,
    info,
    variable_list,
    dict_to_cdf,
    dict_to_cdffile,
    cdffile_to_dict,
    check_cdf_variables
)
