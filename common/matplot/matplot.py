import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil


def clear_cache():
    print('Clearing matplotlib cache ...')
    cache_dir = mpl.get_cachedir()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("  ✔ Matplotlib font cache were removed")
        print("  ※ Please restart the editer")


def init():
    plt.rcParams['font.family'] = 'Roboto Condensed'
    plt.rcParams['font.size'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    return