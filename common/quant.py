"""
17.oct.2024
estimate some quantities
"""
import numpy as np
import pandas as pd
from common.base import display

# ----- constants -----
pi = 3.141592
mu0 = 4 * np.pi * 1e-7  # permeability
mp = 1.6726e-27  # proton mass [kg]
e = 1.602e-19  # charge element [C]
c = 3e8  # light speed [m/s]
epsilon0 = 8.85e-12  # 真空誘電率 [F/m]
me_per_mp = 1 / 1836


# information of planets
info_planet = {
    "mercury": {
        "radius": 2440,
        "mag": 1e-2,
        "mag_eq": 3e-7,
    },
    "earth": {
        "radius": 6400,
        "mag": 1,
        "mag_eq": 3e-5,
        "mag_moment": 7.86432e+22,  # 赤道上で30,000 nTと仮定した場合のdipole磁気モーメント
    },
    "jupiter": {
        "radius": 70000,
        "mag": 2e+4,
        "mag_eq": 4.2e-4,
    }
}

# DataFrameに変換
df = pd.DataFrame(info_planet)
# df = pd.DataFrame(info_planet).T  # 転置して列に惑星名が来るようにする

# -----

# ----- helper function ------
def get_planet_info(data_frame, planet_name, var=None):
    """
    dfから惑星情報を取得する
    """
    if var is None:
        return data_frame[planet_name]
    else:
        return data_frame.at[var, planet_name]  # [index, column]


def deg_to_rad(degree):
    """
    degree -> radian
    """
    return (degree / 180) * np.pi

# -----


def l_shell(name_planet, altitude, mlat):
    """
    Calculate L-shell based on altitude and magnetic latitude.

    Parameters:
    ----------
    name_planet: str
        The name of the planet, e.g., "earth", "mercury", "jupiter".
    altitude: float
        Altitude in kilometers [km].
    mlat: float
        Magnetic latitude in degrees [°].

    Returns:
    -------
    ls: float
    The calculated L-shell value.

    :param name_planet: str, valid option: "earth", "mercury", "jupiter"
    :param altitude: [km]
    :param mlat: magnetic latitude [degree]
    :return: ls: L-shell
    """
    if np.any(altitude < 0):
        raise ValueError(f"altitude must be positive. Given: {altitude}")

    radius_planet = get_planet_info(df, name_planet, "radius")
    mlat = deg_to_rad(mlat)  # [deg] -> [rad]
    ls = (altitude + radius_planet) / radius_planet / np.cos(mlat) ** 2
    return ls


def altitude_from_ls(name_planet, ls, mlat):
    """
    Ls -> altitude
    :param name_planet: str
    :param ls: L-shell
    :param mlat: magnetic latitude [degree]
    :return: altitude [km]
    """
    if np.any(ls < 1):
        raise ValueError(f"L-shell values must be larger than 1. Given: {ls}")

    radius_planet = get_planet_info(df, name_planet, "radius")
    mlat = deg_to_rad(mlat)  # [deg] -> [rad]
    alt = (ls * np.cos(mlat) ** 2 - 1) * radius_planet
    return alt


def mag_dipole(name_planet, ls, mlat, qtype="eq"):
    """
    Calculate the magnetic dipole's magnetic field strength.

    Parameters:
    ----------
    name_planet: str
        The name of the planet for which to calculate the magnetic field.
    ls: float
        The L-shell value.
    mlat: float
        The magnetic latitude in degrees.
    qtype: str, optional
        Type of calculation: "eq" for equatorial, "abs" for absolute.
        "eq" -> use data of the mag field at the equator.
        "abs" -> use data of relative strength of the mag field to the earth.
        Normally, the result between "eq" and "abs" isn't so different, but in case of mercury, "eq" is better.

    Returns:
    -------
    float
        The calculated magnetic field strength.

    :param name_planet: str
    :param ls: L-shell
    :param mlat: [degree]
    :param qtype: "abs", "eq", Default: "eq"
    :return: magnetic field
    """
    if np.any(ls < 1):
        raise ValueError(f"L-shell values must be larger than 1. Given: {ls}")

    mag = 0
    radius_planet = get_planet_info(df, name_planet, "radius")
    altitude = altitude_from_ls(name_planet, ls, mlat)
    mlat = deg_to_rad(mlat)  # [deg] -> [rad]
    radius = radius_planet + altitude  # 惑星中心からの距離 [km]

    if qtype == "abs":
        radius *= 1e+3  # [m]
        mag_mom = df[name_planet].mag * get_planet_info(df, "earth", "mag_moment")
        mag = mu0 * mag_mom / (4 * np.pi * radius ** 3) * np.sqrt(1 + 3 * np.sin(mlat) ** 2)

    if qtype == "eq":
        mag_eq = df[name_planet].mag_eq
        mag = mag_eq * np.sqrt(1 + 3 * np.sin(mlat) ** 2) / (radius / radius_planet) ** 3

    return mag


def mag_absolute(name_planet, mag_eq_surface):
    """
    惑星赤道表面での磁場の強さから magnetic dipole の式の m を求める
    :param name_planet: str
    :param mag_eq_surface: 赤道表面での磁場強度
    :return:
    """
    radius_planet = get_planet_info(df, name_planet, "radius")
    mag_abs = 4 * np.pi * radius_planet ** 3 * mag_eq_surface / mu0
    return mag_abs


def cyclotron_frequency(mass_per_charge, mag, hz=True):
    """
    cyclotron frequency
    :param mass_per_charge: mass per proton mass / charge per charge element
    :param mag: [T]
    :param hz: if True, unit to return is [Hz], Default: True
    :return:
    """
    mass_per_charge *= mp / e
    fc = mag / mass_per_charge
    if hz:
        fc *= 1 / (2 * np.pi)
    return fc


def fc_planet(name_planet, mass_per_charge=1, ls=1, mlat=0, qtype="eq", hz=True):
    """
    cyclotron frequency of planets
    :param name_planet: str
    :param mass_per_charge: mass per proton mass / charge per charge element, Default: 1
    :param ls: L-shell, Default: 1
    :param mlat: magnetic latitude, Default: 0
    :param qtype: "eq"
    :param hz: if True, unit to return is [Hz], Default: True
    :return:
    """
    mag = mag_dipole(name_planet, ls, mlat, qtype)
    fc = cyclotron_frequency(mass_per_charge, mag, hz)
    return fc

def plasma_frequency(dens, charge_per_e, mass_per_mp, hz=True):
    const = e / np.sqrt(epsilon0 * mp)  # ~ 1.3167
    omega_p = const * np.sqrt(dens * charge_per_e ** 2 / mass_per_mp)
    if hz:
        return omega_p / (2 * pi)
    else:
        return omega_p


if __name__ == '__main__':
    fp = plasma_frequency(1, 1, 1, hz=False)
    print(f'{fp=}')

    print(f'{e / mp = }')
