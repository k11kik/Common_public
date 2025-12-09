"""
Dispersion Relation

Contents
    * draw dispersion relation (omega-k diagram)
"""
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import suptitle

from common import quant, util
from common.base import path, display


def init_dict_ions():
    dict_ions = {
        "e-": {"mass_per_mp": quant.me_per_mp, 'charge_per_e': -1, "alpha": -1, 'density': 1},
        "H+": {"mass_per_mp": 1, 'charge_per_e': 1, "alpha": 1},
        "He+": {"mass_per_mp": 4, 'charge_per_e': 1, "alpha": 0},
        "O+": {"mass_per_mp": 16, 'charge_per_e': 1, "alpha": 0},
        "D+": {"mass_per_mp": 2, 'charge_per_e': 1, "alpha": 0},
        "He++": {"mass_per_mp": 4, 'charge_per_e': 2, "alpha": 0},
    }
    # ne = dict_ions['e-']['density']
    # for ion_name, ion_info in dict_ions.items():
    #     dens = ion_info['n_per_ne'] * ne
    #     mass_per_charge = ion_info['mass_per_mp'] / ion_info['charge_per_e']
    #     charge_sign = np.sign(ion_info['charge_per_e'])
    #     ion_info['density'] = dens
    #     ion_info['M/Q'] = mass_per_charge
    #     ion_info['charge_sign'] = charge_sign

    return dict_ions


def update_derived_quantities(dict_ions):
    ne = dict_ions['e-']['density']  # electron density
    for ion_info in dict_ions.values():
        charge = ion_info['charge_per_e']
        ion_info['n_per_ne'] = ion_info['alpha'] / charge
        ion_info['density'] = ion_info['n_per_ne'] * ne
        ion_info['M/Q'] = ion_info['mass_per_mp'] / charge if charge != 0 else np.inf
        ion_info['charge_sign'] = np.sign(charge)
    
    total_alpha = 0
    for ion_name, ion_info in dict_ions.items():
        total_alpha += ion_info['alpha']
    # total_ion_n_per_ne = total_n_per_ne - 1  # total_n_per_neはelectronのも含んでいるから「1」を引く
    if abs(total_alpha) > 1e-5:
        display.warning('dispersion/run_plot', f'Total relative ion density must be 0: {total_alpha=}')
    return dict_ions


def xy_value(
        dict_ions: dict,
        omega,
        mag = 1e-9
):
    required_keys = ['density', 'charge_per_e', 'mass_per_mp', 'charge_sign']
    for ion_name, ion_info in dict_ions.items():
        for key in required_keys:
            if key not in ion_info:
                logging.error(f"key '{key}' is missing in ion '{ion_name}'")
        # X value
        pi = quant.plasma_frequency(ion_info['density'], ion_info['charge_per_e'], ion_info['mass_per_mp'], hz=False)
        x_value = (pi / omega) ** 2

        # Y value
        omega_c = quant.cyclotron_frequency(ion_info['M/Q'], mag, hz=False)
        y_value = omega_c / omega

        # conserve
        ion_info['Pi'] = pi
        ion_info['Omega'] = omega_c
        ion_info['X'] = x_value
        ion_info['Y'] = y_value
    return


def rlpsd_value(
        dict_ions: dict,
        omega,
        mag = 1e-9
):
    xy_value(dict_ions, omega, mag)
    r_value = np.ones_like(omega)
    l_value = np.ones_like(omega)
    p_value = np.ones_like(omega)
    required_keys = ['X', 'Y']
    for ion_name, ion_info in dict_ions.items():
        # check keys
        for key in required_keys:
            if key not in ion_info:
                display.error('dispersion/rlpsd_value', f"key '{key}' is missing in ion '{ion_name}'")

        r_value -= ion_info['X'] / (1 + ion_info['Y'])
        l_value -= ion_info['X'] / (1 - ion_info['Y'])
        p_value -= ion_info['X']
    
    s_value = (r_value + l_value) / 2
    d_value = (r_value - l_value) / 2
    return {
        'R': r_value,
        'L': l_value,
        'P': p_value,
        'S': s_value,
        'D': d_value
    }



def rlp_value(
        dict_ions: dict,
        omega,
        mag = 1e-9
):
    xy_value(dict_ions, omega, mag)
    r_value = np.ones_like(omega)
    l_value = np.ones_like(omega)
    p_value = np.ones_like(omega)
    required_keys = ['X', 'Y']
    for ion_name, ion_info in dict_ions.items():
        # check keys
        for key in required_keys:
            if key not in ion_info:
                logging.error(f"key '{key}' is missing in ion '{ion_name}'")

        r_value -= ion_info['X'] / (1 + ion_info['Y'])
        l_value -= ion_info['X'] / (1 - ion_info['Y'])
        p_value -= ion_info['X']
    return r_value, l_value, p_value


def squared_refractive_index(
        r_value,
        l_value,
        p_value,
        theta_deg: float = 0
):
    theta = theta_deg * quant.pi / 180  # deg -> rad
    s_value = (r_value + l_value) / 2
    d_value = (r_value - l_value) / 2
    a = s_value * np.sin(theta) ** 2 + p_value * np.cos(theta) ** 2
    b = r_value * l_value * np.sin(theta) ** 2 + p_value * s_value * (1 + np.cos(theta) ** 2)
    f = np.sqrt((r_value * l_value - p_value * s_value) ** 2 * np.sin(theta) ** 4 + (2 * p_value * d_value * np.cos(theta)) ** 2)

    return (b + f) / (2 * a), (b - f) / (2 * a)


def wave_number(squared_n, omega):
    """
    Computes wave number k from squared refractive index and omega.
    If squared_n < 0, returns np.nan.
    """
    squared_k = squared_n * (omega / quant.c) ** 2
    k = np.sqrt(np.where(squared_k > 0, squared_k, np.nan))
    return k


def split_modes(squared_np, squared_nm, omega, dict_value):
    """Split into R and L mode components using q_value."""
    def q(squared_n):
        s, r, l, d = dict_value['S'], dict_value['R'], dict_value['L'], dict_value['D']
        return (squared_n * s - r * l) / d

    modes = {
        'kr+': wave_number(np.where((squared_np > 0) & (q(squared_np) > 0), squared_np, np.nan), omega),
        'kr-': wave_number(np.where((squared_nm > 0) & (q(squared_nm) > 0), squared_nm, np.nan), omega),
        'kl+': wave_number(np.where((squared_np > 0) & (q(squared_np) < 0), squared_np, np.nan), omega),
        'kl-': wave_number(np.where((squared_nm > 0) & (q(squared_nm) < 0), squared_nm, np.nan), omega),
    }
    return modes


def wave_number_old01(
        dict_value: dict,
        squared_n_plus,
        squared_n_minus,
        omega
):
    def q_value(squared_n, dict_value):
        s_value = dict_value['S']
        r_value = dict_value['R']
        l_value = dict_value['L']
        d_value = dict_value['D']
        return (squared_n * s_value - r_value * l_value) / d_value

    # judging R/L mode
    q_plus = q_value(squared_n_plus, dict_value)
    q_minus = q_value(squared_n_minus, dict_value)
    r_plus = (squared_n_plus > 0) & (q_plus > 0) 
    l_plus = (squared_n_plus > 0) & (q_plus < 0) 
    r_minus = (squared_n_minus > 0) & (q_minus > 0) 
    l_minus = (squared_n_minus > 0) & (q_minus < 0)
    squared_nr_plus = np.where(r_plus, squared_n_plus, np.nan)
    squared_nl_plus = np.where(l_plus, squared_n_plus, np.nan)
    squared_nr_minus = np.where(r_minus, squared_n_minus, np.nan)
    squared_nl_minus = np.where(l_minus, squared_n_minus, np.nan)

    squared_kr_plus = squared_nr_plus * (omega / quant.c) ** 2
    squared_kr_minus = squared_nr_minus * (omega / quant.c) ** 2
    squared_kl_plus = squared_nl_plus * (omega / quant.c) ** 2
    squared_kl_minus = squared_nl_minus * (omega / quant.c) ** 2

    krp = np.sqrt(squared_kr_plus)
    krm = np.sqrt(squared_kr_minus)
    klp = np.sqrt(squared_kl_plus)
    klm = np.sqrt(squared_kl_minus)


    # kr = np.lib.scimath.sqrt(squared_kr)
    # kl = np.lib.scimath.sqrt(squared_kl)
    # kr = np.sqrt(np.maximum(squared_kr, 0))
    # kl = np.sqrt(np.maximum(squared_kl, 0))
    # kr = np.where(kr == 0, np.nan, kr)
    # kl = np.where(kl == 0, np.nan, kl)
    return krp, krm, klp, klm


def plot_monotonic_segments(
        x,
        y,
        ax=None,
        label=None,
        color=None,
        linestyle=None,
        alpha=None
):
    if ax is None:
        fig, ax = plt.subplots()

    # Identify monotonic segments
    diff = np.diff(x)

    change_points = np.where(diff < 0)[0] + 1
    # print(f'{change_points=}')

    segment_starts = np.concatenate(([0], change_points))
    segment_ends = np.concatenate((change_points, [len(x)]))

    label_used = False

    for s, e in zip(segment_starts, segment_ends):
        # print(f'{s=}, {e=}')
        x_seg = x[s:e]
        y_seg = y[s:e]
        # print(f'{x_seg=}')
        # print(f'{y_seg=}')
        if len(x_seg) == 0 or np.all(np.isnan(x_seg)) or np.all(np.isnan(y_seg)):
            continue
        ax.plot(
            x_seg, y_seg,
            label=label if not label_used else None,
            color=color,
            linestyle=linestyle,
            alpha=alpha
        )
        label_used = True  # <- 一度使ったらフラグを立てる

    # if ax is None:
    #     fig, ax = plt.subplots()
    #
    # x = np.asarray(x)
    # y = np.asarray(y)
    #
    # # 前後の差分をとって単調増加でない点を探す
    # diff = np.diff(x)
    # change_points = np.where(diff < 0)[0] + 1  # 増加→減少の箇所の「次のインデックス」
    #
    # # セグメントの開始・終了インデックスを作る
    # segment_starts = np.concatenate(([0], change_points))
    # segment_ends = np.concatenate((change_points, [len(x)]))
    #
    # for s, e in zip(segment_starts, segment_ends):
    #     ax.plot(
    #         x[s:e], y[s:e],
    #         label=label,
    #         color=color,
    #         linestyle=linestyle,
    #         alpha=alpha
    #     )

    return


def plot_dispersion_relation(
        dict_ions,
        mag=1e-9,
        theta_deg: any=0,
        omega = np.logspace(-4, 1, 100000),
        normalize=True,
        xlim=None,
        ylim=None,
        xlog=False,
        ylog=False,
        xlabel=None,
        ylabel=None,
        dir_save_png=None,
        save_png=None,
        suptitle: str=None
):
    # R, L, P
    dict_value = rlpsd_value(dict_ions, omega, mag)
    r_value = dict_value['R']
    l_value = dict_value['L']
    p_value = dict_value['P']


    # constants for normalization
    fcp = dict_ions['H+']['Omega']
    comp_va = 0
    for ion_name, ion_info in dict_ions.items():
        comp_va += ion_info['density'] * ion_info['mass_per_mp']
    va = mag / np.sqrt(quant.mu0 * quant.mp) / np.sqrt(comp_va)

    fig, ax = plt.subplots(figsize=(6, 8))

    if isinstance(theta_deg, (int, float)):
        squared_np, squared_nm = squared_refractive_index(r_value, l_value, p_value, theta_deg=theta_deg)
        modes = split_modes(squared_np, squared_nm, omega, dict_value)
        krp = modes['kr+']
        krm = modes['kr-']
        klp = modes['kl+']
        klm = modes['kl-']
        # squared_nr, squared_nl = squared_refractive_index(r_value, l_value, p_value, theta_deg=theta_deg)
        # kr, kl = wave_number(squared_nr, squared_nl, omega)

        if normalize:
            # Normalized axes
            x_krp = va * krp / fcp
            x_krm = va * krm / fcp
            x_klp = va * klp / fcp
            x_klm = va * klm / fcp
            y = omega / fcp
        else:
            x_krp = krp
            x_krm = krm
            x_klp = klp
            x_klm = klm
            y = omega
        
        plot_monotonic_segments(x_krp, y, ax=ax, color='red')
        plot_monotonic_segments(x_krm, y, ax=ax, color='red')
        plot_monotonic_segments(x_klp, y, ax=ax, color='blue')
        plot_monotonic_segments(x_klm, y, ax=ax, color='blue')


    elif isinstance(theta_deg, (list, np.ndarray)):
        for i, theta_i in enumerate(theta_deg):
            if i == 0:
                ls_i = '--'
                ls_klm_i = '--'
                alpha_i = 1
                alpha_klm_i = 1
                label_r = "R-mode, θ=0°"
                label_l = "L-mode, θ=0°"
            elif i == len(theta_deg) - 1:
                ls_i = '-'
                ls_klm_i = 'dotted'
                alpha_i = 1
                alpha_klm_i = .2
                label_r = "R-mode, θ=90°"
                label_l = "L-mode, θ=90°"
            else:
                ls_i = 'dotted'
                ls_klm_i = 'dotted'
                alpha_i = .2
                alpha_klm_i = .2
                label_r = None
                label_l = None

            squared_np, squared_nm = squared_refractive_index(r_value, l_value, p_value, theta_deg=theta_i)
            modes = split_modes(squared_np, squared_nm, omega, dict_value)
            krp = modes['kr+']
            krm = modes['kr-']
            klp = modes['kl+']
            klm = modes['kl-']

            
            # krp, krm, klp, klm = wave_number(dict_value, squared_np, squared_nm, omega)

            # squared_nr, squared_nl = squared_refractive_index(r_value, l_value, p_value, theta_deg=theta_i)
            # kr, kl = wave_number(squared_nr, squared_nl, omega)
            if normalize:
                # Normalized axes
                x_krp = va * krp / fcp
                x_krm = va * krm / fcp
                x_klp = va * klp / fcp
                x_klm = va * klm / fcp
                y = omega / fcp
            else:
                x_krp = krp
                x_krm = krm
                x_klp = klp
                x_klm = klm
                y = omega


            # plot
            plot_monotonic_segments(x_krp, y, ax=ax, color='red', linestyle=ls_i, alpha=alpha_i)
            plot_monotonic_segments(x_krm, y, ax=ax, color='red', linestyle=ls_i, alpha=alpha_i)
            plot_monotonic_segments(x_klp, y, ax=ax, color='blue', linestyle=ls_i, alpha=alpha_i)
            plot_monotonic_segments(x_klm, y, ax=ax, color='blue', linestyle=ls_klm_i, alpha=alpha_klm_i)
            
    else:
        raise ValueError(f'Unsupposed type of theta_deg: {type(theta_deg)}')

    if not normalize:
        min_x_krp, max_x_krp = np.nanmin(x_krp), np.nanmax(x_krp)
        min_x_krm, max_x_krm = np.nanmin(x_krm), np.nanmax(x_krm)
        min_x_klp, max_x_klp = np.nanmin(x_klp), np.nanmax(x_klp)
        min_x_klm, max_x_klm = np.nanmin(x_klm), np.nanmax(x_klm)
        min_x = np.min([min_x_krp, min_x_krm, min_x_klp, min_x_klm])
        max_x = np.max([max_x_krp, max_x_krm, max_x_klp, max_x_klm])
        ax.hlines(fcp, min_x, max_x, color='gray', linestyles='dashed')
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.grid(ls="--")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.legend(loc='upper right')

    # title, label
    if xlabel is None:
        if normalize:
            ax.set_xlabel(r"$V_A k / \Omega_{H^+}$")
        else:
            ax.set_xlabel('k')
    else:
        ax.set_xlabel(xlabel)
    if ylabel is None:
        if normalize:
            ax.set_ylabel(r"$\omega / \Omega_{H^+}$")
        else:
            ax.set_ylabel('omega')
    else:
        ax.set_ylabel(ylabel)
    fig.suptitle(suptitle)
    path.savefig(save_png)

    return


def run_plot(
        dict_ions: dict = None,
        save_png: str = None,
        normalize: bool = True,
        theta_deg = None
):
    if dict_ions is None:
        dict_ions = init_dict_ions()
    # total_alpha = 0
    # for ion_name, ion_info in dict_ions.items():
    #     total_alpha += ion_info['alpha']
    # # total_ion_n_per_ne = total_n_per_ne - 1  # total_n_per_neはelectronのも含んでいるから「1」を引く
    # if abs(total_alpha) > 1e-5:
    #     display.warning('dispersion/run_plot', f'Total relative ion density must be 0: {total_alpha=}')
    dict_ions = update_derived_quantities(dict_ions)
    util.print_dict(dict_ions)

    mag = 1e-9
    omega = np.logspace(-4, 1, 100000)
    # omega = np.logspace(-4, 2, 100000)
    # theta_deg = 90
    # theta_deg = [0, 90]
    if theta_deg is None:
        theta_deg = np.linspace(0, 89.9, 100)

    title_ion_ratio = ''
    ion_names_exist = []
    for i, (ion_name, ion_info) in enumerate(dict_ions.items()):
        alpha_i = ion_info['alpha']
        if alpha_i != 0:
            ion_names_exist.append(ion_name)

    for i, ion_name in enumerate(ion_names_exist):
        title_ion_ratio += f'{ion_name}={abs(dict_ions[ion_name]['alpha']) * 100:.2f}%'
        if not i == (len(ion_names_exist) - 1):
            title_ion_ratio += ', '
    if isinstance(theta_deg, (int, float)):
        suptitle = (f'Dispersion Relation (ω-k diagram)\n'
                    f'{title_ion_ratio}\n'
                    f'{theta_deg=}')
    elif isinstance(theta_deg, (list, np.ndarray)):
        suptitle = (f'Dispersion Relation (ω-k diagram)\n'
                    f'{title_ion_ratio}\n'
                    f'theta_deg=[{theta_deg[0]}, {theta_deg[-1]}]')

    plot_dispersion_relation(
        dict_ions,
        mag=mag,
        omega=omega,
        theta_deg=theta_deg,
        normalize=normalize,
        # xlog=True,
        # ylog=True,
        xlim=[1e-2, 2],
        ylim=[0, 1],
        suptitle=suptitle,
        save_png=save_png
    )

    return

if __name__ == '__main__':
    run_plot()

    # isono(
    #     save_png=r'C:\Users\kriku\PyCharmProjects\TohokuUniv\pparc\erg\out\ErgAnalysis2\test_dispersion_relation_isono.png',
    #     # save_png='/Users/kikuchiriku/PythonPyCharm/Tohoku-Univ./pparc/erg/out/ErgAnalysis2/test_dispersion_relation_isono.png'
    # )

    # miyashita(
    #     # save_png=r'C:\Users\kriku\PyCharmProjects\TohokuUniv\pparc\erg\out\ErgAnalysis2\test_dispersion_relation_miyashita.png',
    #     save_png='/Users/kikuchiriku/PythonPyCharm/Tohoku-Univ./pparc/erg/out/ErgAnalysis2/test_dispersion_relation_miyashita.png'
    # )

    # mag = 1e-9
    #
    # omega = np.logspace(-4, 1, 100000)
    # # omega = np.linspace(1e-2, int(1e3), 1000)
    # # xy_value(dict_ions, omega, mag)
    # #
    # # fig, axes = plt.subplots(2, 1, sharex='all')
    # # for ion_name, ion_info in dict_ions.items():
    # #     axes[0].plot(omega, ion_info['X'], label=f'{ion_name}')
    # #     axes[1].plot(omega, np.abs(ion_info['Y']), label=f'{ion_name}')
    # # axes[0].set_xscale('log')
    # # axes[0].set_yscale('log')
    # # axes[0].set_ylabel('X value')
    # # axes[0].grid(ls='--')
    # # axes[0].legend(loc='upper right')
    # # axes[1].set_yscale('log')
    # # axes[1].set_ylabel('Y value')
    # # axes[1].grid(ls='--')
    # # axes[1].set_xlabel('omega')
    # # axes[1].legend(loc='upper right')
    #
    # # R, L, P
    # r, l, p = rlp_value(dict_ions, omega, mag)
    #
    # # fig, ax = plt.subplots()
    # # ax.plot(omega, r, label='R')
    # # ax.plot(omega, l, label='L')
    # # ax.plot(omega, p, label='P')
    # # ax.set_xscale('log')
    # # ax.set_yscale('log')
    # # ax.legend()
    # # ax.set_xlabel('omega')
    #
    # # n^2
    # squared_nr, squared_nl = squared_refractive_index(r, l, p, theta_deg=0)
    # squared_nr_perp, squared_nl_perp = squared_refractive_index(r, l, p, theta_deg=90)
    #
    # fig, ax = plt.subplots()
    # ax.plot(omega, squared_nr, label='R-mode')
    # ax.plot(omega, squared_nl, label='L-mode')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.legend()
    # ax.set_xlabel('omega')
    # ax.set_ylabel('n^2')
    #
    # # dispersion relation
    # fcp = dict_ions['H+']['Omega']
    # comp_va = 0
    # for ion_name, ion_info in dict_ions.items():
    #     comp_va += ion_info['density'] * ion_info['mass_per_mp']
    # va = mag / np.sqrt(quant.mu0 * quant.mp) / np.sqrt(comp_va)
    #
    #
    # squared_kr = squared_nr * (omega / quant.c) ** 2
    # squared_kl = squared_nl * (omega / quant.c) ** 2
    # kr = np.sqrt(np.maximum(squared_kr, 0))
    # kl = np.sqrt(np.maximum(squared_kl, 0))
    # kr = np.where(kr == 0, np.nan, kr)
    # kl = np.where(kl == 0, np.nan, kl)
    #
    # squared_kr_perp = squared_nr_perp * (omega / quant.c) ** 2
    # squared_kl_perp = squared_nl_perp * (omega / quant.c) ** 2
    # kr_perp = np.sqrt(np.maximum(squared_kr_perp, 0))
    # kl_perp = np.sqrt(np.maximum(squared_kl_perp, 0))
    # kr_perp = np.where(kr_perp == 0, np.nan, kr_perp)
    # kl_perp = np.where(kl_perp == 0, np.nan, kl_perp)
    #
    # # Normalized axes
    # x_kr = va * kr / fcp
    # x_kl = va * kl / fcp
    # y = omega / fcp
    #
    # x_kr_perp = va * kr_perp / fcp
    # x_kl_perp = va * kl_perp / fcp
    #
    # fig, ax = plt.subplots()
    # ax.plot(kr, omega, label='R-mode', color='red', linestyle='--')
    # ax.plot(kl, omega, label='L-mode', color='blue', linestyle='--')
    # ax.plot(kr_perp, omega, label='R-mode', color='red')
    # ax.plot(kl_perp, omega, label='L-mode', color='blue')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.grid(ls='--')
    # ax.set_xlabel('k')
    # ax.set_ylabel('omega')
    # ax.legend()
    #
    # # Plot
    # plt.figure(figsize=(6, 8))
    # plt.plot(x_kr, y, color='red', linestyle='--')
    # plt.plot(x_kl, y, color='blue', linestyle='--')
    # plt.plot(x_kr_perp, y, color='red')
    # plt.plot(x_kl_perp, y, color='blue')
    # # plt.xscale("log")
    # # plt.yscale("log")
    # plt.xlabel(r"$V_A k / \Omega_{H^+}$")
    # plt.ylabel(r"$\omega / \Omega_{H^+}$")
    # plt.title("Dispersion Relation (ω–k diagram)")
    # plt.grid(True, which="both", ls="--")
    # # plt.legend()
    # plt.tight_layout()
    # plt.xlim([0, 1.5])
    # plt.ylim([0, 1])
    #
    #
    # # for each theta
    # list_theta_deg = np.linspace(0, 90, 50)
    # fig, ax = plt.subplots(figsize=(6, 8))
    # for i, theta_i in enumerate(list_theta_deg):
    #     if i == 0:
    #         ls_i = '--'
    #         alpha_i = 1
    #     elif i == len(list_theta_deg) - 1:
    #         ls_i = '-'
    #         alpha_i = 1
    #     else:
    #         ls_i = 'dotted'
    #         alpha_i = .1
    #     squared_nr, squared_nl = squared_refractive_index(r, l, p, theta_deg=theta_i)
    #     kr, kl = wave_number(squared_nr, squared_nl, omega)
    #     # Normalized axes
    #     x_kr = va * kr / fcp
    #     x_kl = va * kl / fcp
    #     y = omega / fcp
    #     # plot
    #     plot_monotonic_segments(x_kr, y, ax=ax, color='red', linestyle=ls_i, alpha=alpha_i)
    #     plot_monotonic_segments(x_kl, y, ax=ax, color='blue', linestyle=ls_i, alpha=alpha_i)
    #     # ax.plot(x_kr, y, color='red', linestyle=ls_i)
    #     # ax.plot(x_kl, y, color='blue', linestyle=ls_i)
    #     # plt.xscale("log")
    #     # plt.yscale("log")
    #     ax.set_xlabel(r"$V_A k / \Omega_{H^+}$")
    #     ax.set_ylabel(r"$\omega / \Omega_{H^+}$")
    #     ax.set_title("Dispersion Relation (ω–k diagram)")
    #     ax.grid(True, which="both", ls="--")
    #     # plt.legend()
    #     ax.set_xlim([0, 2])
    #     ax.set_ylim([0, 1])
    #
    # plt.show()

# ----------------------------------------------------------------------------------------------------------------------

def isono(
        save_png: str = None
):
    # 光速
    c = 3e8

    # 質量
    # プラズマ周波数(電子のサイクロトロン周波数で規格化)
    # サイクロトロン周波数(電子のサイクロトロン周波数で規格化)

    # 辞書型を用いることで、パラメーターを直感的に扱えるようにする。

    mag = 1e-9
    oe = quant.cyclotron_frequency(1/1800, mag, hz=False)
    pe = quant.plasma_frequency(1, -1, 1/1800, hz=False)
    me = quant.mp / 1800

    # me = 9.1093837015e-31  # 電子
    # oe = 1.0  # 電子
    # pe = oe * 1.6  # 電子

    electrons = {"mass": me, "charge_sign": -1.0, "Cyclotron_freq": oe, "Plasma_freq": pe, "name": "electron"}

    ah = .8  # proton and oxygen ratio
    ahe = .2
    ao = 0

    mh = quant.mp
    oh = quant.cyclotron_frequency(1, mag, hz=False)
    ph = quant.plasma_frequency(ah, 1, 1, hz=False)

    # mh = 1.67262192e-27  # プロトン
    # oh = oe * me / mh  # プロトン
    # ph = pe * np.sqrt(a * me / mh)  # プロトン

    mhe = quant.mp * 4
    ohe = quant.cyclotron_frequency(4, mag, hz=False)
    phe = quant.plasma_frequency(ahe, 1, 4, hz=False)

    mo = quant.mp * 16
    oo = quant.cyclotron_frequency(16, mag, hz=False)
    po = quant.plasma_frequency(ao, 1, 16,  hz=False)

    # mo = mh * 16  # 酸素
    # oo = oe * me / mo  # 酸素
    # po = pe * np.sqrt((1.0 - a) * me / mo)  # 酸素

    protons = {"mass": mh, "charge_sign": 1.0, "Cyclotron_freq": oh, "Plasma_freq": ph, "name": "proton"}
    heliums = {"mass": mhe, "charge_sign": 1.0, "Cyclotron_freq": ohe, "Plasma_freq": phe, "name": "proton"}
    oxygens = {"mass": mo, "charge_sign": 1.0, "Cyclotron_freq": oo, "Plasma_freq": po, "name": "oxygen"}

    elements = {"electron": electrons, "proton": protons, 'helium': heliums,  "oxygen": oxygens}

    def filter_arrays(sorted_w, sorted_kp, scale_factor, ratio):

        # x軸方向で隣接する要素間の差を計算(logで)
        differences = np.diff(np.log10(sorted_kp))

        # 刻み幅の閾値を計算（scale_factorをそのまましきい値とする。）
        threshold = scale_factor

        # 閾値以下もしくは8の倍数インデックスを持つ要素のインデックスを取得
        valid_indices = [i for i, diff in enumerate(differences, 1) if diff > threshold or i % 4 == 0]

        # 有効な要素のみを抽出
        filtered_w = sorted_w[valid_indices]
        filtered_kp = sorted_kp[valid_indices]

        return filtered_w, filtered_kp

    # 分散関係の計算を関数化
    def calc_disp(theta, w):

        R = np.ones(w.shape[0])
        L = np.ones(w.shape[0])
        P = np.ones(w.shape[0])

        for element in elements.values():
            plasma_frequencies = element["Plasma_freq"]
            plasma_cycltrons = element["Cyclotron_freq"]
            charge_sign = element["charge_sign"]
            R += -plasma_frequencies ** 2 / (w * (w + plasma_cycltrons * charge_sign))
            L += -plasma_frequencies ** 2 / (w * (w - plasma_cycltrons * charge_sign))
            P += -plasma_frequencies ** 2 / (w * w)

        S = (R + L) / 2

        A = S * np.sin(theta) ** 2 + P * np.cos(theta) ** 2
        B = R * L * np.sin(theta) ** 2 + P * S * (1 + np.cos(theta) ** 2)
        C = P * R * L
        F = np.sqrt(B ** 2 - 4 * A * C)
        n2p = (B + F) / (2 * A)  # 複合プラス
        n2m = (B - F) / (2 * A)  # 複合マイナス
        # k=の形にする
        kpp = w / c * np.sqrt(n2p)
        #     kpm = -w/c*np.sqrt(n2p)
        kmp = w / c * np.sqrt(n2m)
        #     kmm = -w/c*np.sqrt(n2m)

        # 密集している領域の点を省く

        # Nanを排除
        non_nan_p_indices = np.where(~np.isnan(kpp))[0]
        non_nan_m_indices = np.where(~np.isnan(kmp))[0]

        # NaN がない要素のみで配列をフィルタリング
        nan_filtered_kpp = kpp[non_nan_p_indices]
        nan_filtered_pw = w[non_nan_p_indices]

        nan_filtered_kmp = kmp[non_nan_m_indices]
        nan_filtered_mw = w[non_nan_m_indices]

        # 波数順になるように並び替える。
        sorted_indices = np.argsort(nan_filtered_kpp)
        sorted_pw = nan_filtered_pw[sorted_indices]
        sorted_kpp = nan_filtered_kpp[sorted_indices]

        sorted_indices = np.argsort(nan_filtered_kmp)
        sorted_mw = nan_filtered_mw[sorted_indices]
        sorted_kmp = nan_filtered_kmp[sorted_indices]

        filtered_pw, filtered_kpp = sorted_pw, sorted_kpp
        filtered_mw, filtered_kmp = sorted_mw, sorted_kmp

        for i in range(2):
            filtered_pw, filtered_kpp = filter_arrays(filtered_pw, filtered_kpp, 0.0025, i)
            filtered_mw, filtered_kmp = filter_arrays(filtered_mw, filtered_kmp, 0.0025, i)

        return [filtered_kpp, filtered_kmp, filtered_pw, filtered_mw, theta]

    # w = np.arange(10 ** -6, 1, 10 ** -6)  # 変数は周波数
    # wlog = np.logspace(-6, -3, 10 ** 5)  # 対数の配列も生成
    # w = np.concatenate((w, wlog))  # 上の2つの配列を結合
    w = np.logspace(-4, 1, 100000)

    thetas = [0, np.pi/2]
    # thetas = [0, np.pi / 12, np.pi / 4, np.pi / 2.4, np.pi / 2]

    results = []
    for theta in thetas:
        results.append(calc_disp(theta, w))

    # 描画

    # 描画関数
    def plot_disp(
            results, w,
            colormap=None
    ):
        if colormap is None:
            colormap = ['pink', 'red', 'lightblue', 'blue']
        count_plot = 0

        fig = plt.figure(figsize=(6, 8))

        for i in [0, 1]:  # kpp & kmp
            if i == 0:
                label_mode = 'R-mode'
            else:
                label_mode = 'L-mode'
            for result in results:  # 各thetaごとに
                if i == 0:
                    color_i = 'red'
                else:
                    color_i = 'blue'
                if result[4] == 0:
                    marker_i = 'o'
                elif np.abs(result[4] - np.pi / 2) < 1e-5:
                    marker_i = 'x'
                else:
                    marker_i = '.'

                plt.scatter(result[i], result[i + 2], marker='.', color=colormap[count_plot], label=f'{label_mode}, theta={result[4]:.2f}', s=3)
                count_plot += 1
                # theta = result[4]
                # color = colors[i]
                # colorpower = (theta / (np.pi / 2))
                # if (theta != 0):
                #     if (color == 'cyan'):
                #         color = (0, colorpower, colorpower)
                #     if (color == 'magenta'):
                #         color = (colorpower, 0, colorpower)
                # if (i == 0):
                #     plt.scatter(result[i], result[i + 2], color=color, s=0.3,
                #                 label=r'$B-F$,$\theta=$' + str(theta / np.pi * 180))
                # else:
                #     plt.scatter(result[i], result[i + 2], color=color, s=0.3,
                #                 label=r'$B+F$,$\theta=$' + str(theta / np.pi * 180))

    # color_map = {'theta_0': ('blue', 'red'), 'theta_not_0': ('cyan', 'magenta')}

    # 内包リストを使って、result[4]←theta が0のリストとそれを省いたリストを生成。
    plot_disp([result for result in results], w)
    # plot_disp([result for result in results if result[4] != 0], w)
    # plot_disp([result for result in results if result[4] == 0], w)

    # plt.ylim(5*10**2,0.7*10**3)
    plt.xlim(10 ** -12, 10 ** -7)
    plt.xscale('log')
    plt.yscale('log')
    count = 0
    # for element in elements.values():
    #     plasma_frequencies = element["Plasma_freq"]
    #     plasma_cycltrons = element["Cyclotron_freq"]
    #     count += 1
    #     plt.axhline(y=plasma_frequencies, linestyle='dashdot', alpha=0.3 * count, color='k',
    #                 label=element['name'] + 'Plasma frequency')
    #     plt.axhline(y=plasma_cycltrons, linestyle='dashed', alpha=0.3 * count, color='k',
    #                 label=element['name'] + 'Cyclotron frequency')
    # lightcol = (0.7, 0.7, 0.0)
    # plt.plot(w / c, w, linestyle='dotted', alpha=1.0, color=lightcol, label='Light speed')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend()
    plt.xlim([5e-12, 5e-6])
    plt.grid(ls='--')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\omega$')
    plt.title('Dispersion relation for Cold Plasma (Ion+Electron)')
    path.savefig(save_png)
    return


def miyashita(
        save_png: str = None
):
    # 物理定数
    c = 3 * 10 ** 8  # 光速
    mu_0 = 4 * np.pi * 1e-7  # 真空の透磁率

    # 磁場強度と質量密度（仮定値）
    B = 1e-9  # 磁場強度 [T]
    rho = 1e-18  # 質量密度 [kg/m^3]

    # ion ratio
    ah = .8
    ahe = .2
    ao = 0

    # アルフベン速度とプロトンのサイクロトロン周波数での無次元化定数
    v_A = B / np.sqrt(mu_0 * rho)
    me = 9.1093837015e-31
    mh = 1.67262192e-27
    ce = 1
    ch = ce * me / mh

    # 電子の特性
    pe = ce * 1.5
    electrons = {"mass": me, "charge_sign": -1.0, "Cyclotron_freq": ce, "Plasma_freq": pe, "name": "electron"}

    # プロトンの特性
    ph = pe * np.sqrt(ah * me / mh)
    protons = {"mass": mh, "charge_sign": 1.0, "Cyclotron_freq": ch, "Plasma_freq": ph, "name": "proton"}

    # ヘリウムの特性
    mo = mh * 4
    co = ce * me / mo
    po = pe * np.sqrt(ahe * me / mo)
    heliums = {"mass": mo, "charge_sign": 1.0, "Cyclotron_freq": co, "Plasma_freq": po, "name": "helium"}

    # 酸素の特性
    mo2 = mh * 16
    co2 = ce * me / mo2
    po2 = pe * np.sqrt(ao * me / mo2)
    oxygens = {"mass": mo2, "charge_sign": 1.0, "Cyclotron_freq": co2, "Plasma_freq": po2, "name": "oxygen"}

    elements = {"electron": electrons, "proton": protons, "helium": heliums, "oxygen": oxygens}

    # 周波数範囲 (プロトンのサイクロトロン周波数で規格化)
    wmax = 4.50001 * ch
    wmin = 0.000001 * ch
    w = np.arange(wmin, wmax, 0.00001 * ch)

    # 計算の高速化
    plasma_freq_squared = {element: elements[element]["Plasma_freq"] ** 2 for element in elements}
    cyclotron_freq = {element: elements[element]["Cyclotron_freq"] for element in elements}
    charge_sign = {element: elements[element]["charge_sign"] for element in elements}

    # Σを使用してR, L, Pを計算
    R = 1 - sum(plasma_freq_squared[element] / (w * (w + cyclotron_freq[element] * charge_sign[element])) for element in
                elements)
    L = 1 - sum(plasma_freq_squared[element] / (w * (w - cyclotron_freq[element] * charge_sign[element])) for element in
                elements)
    P = 1 - sum(plasma_freq_squared[element] / (w ** 2) for element in elements)
    S = (R + L) / 2

    # クロスオーバー周波数を計算 (θ = 0)
    sin_theta_squared = 0
    cos_theta_squared = 1
    A = S * sin_theta_squared + P * cos_theta_squared
    B = R * L * sin_theta_squared + P * S * (1 + cos_theta_squared)
    C = P * R * L
    F = np.sqrt(np.maximum(B ** 2 - 4 * A * C, 0))
    n2p = (B + F) / (2 * A)
    n2m = (B - F) / (2 * A)

    difference = np.abs(n2p - n2m)
    min_index = np.argmin(difference)
    # crossover_frequency1 = w[min_index]
    crossover_frequency1 = 0.275 * ch
    He_cyc = 0.24 * ch

    thetas = [0]
    # プロット
    plt.figure(figsize=(6, 8))
    for theta in thetas:
        sin_theta_squared = np.sin(theta) ** 2
        cos_theta_squared = np.cos(theta) ** 2

        A = S * sin_theta_squared + P * cos_theta_squared
        B = R * L * sin_theta_squared + P * S * (1 + cos_theta_squared)
        C = P * R * L
        F = np.sqrt(np.maximum(B ** 2 - 4 * A * C, 0))

        n2p = (B + F) / (2 * A)
        n2m = (B - F) / (2 * A)

        kpp = (w / v_A) * np.sqrt(n2p) / ch
        kmp = (w / v_A) * np.sqrt(n2m) / ch

        # 各周波数帯について場合分け
        over_crossover_frq1 = w > 0.275 * ch
        he_cyclotron_to_crossover1 = np.logical_and(w >= 0.25 * ch, w <= 0.275 * ch)
        crossover2_to_he_cyclotron = np.logical_and(w >= 0.09 * ch, w <= 0.25 * ch)
        o_cyclotron_to_crossover2 = np.logical_and(w >= 0.0625 * ch, w <= 0.09 * ch)
        under_o_cyclotprn = w < 0.0625 * ch

        # 色分けプロット
        plt.scatter(kmp[over_crossover_frq1], w[over_crossover_frq1] / ch, color='b', s=0.7)
        plt.scatter(kmp[he_cyclotron_to_crossover1], w[he_cyclotron_to_crossover1] / ch, color='r', s=0.7)
        plt.scatter(kmp[crossover2_to_he_cyclotron], w[crossover2_to_he_cyclotron] / ch, color='b', s=0.7)
        plt.scatter(kmp[o_cyclotron_to_crossover2], w[o_cyclotron_to_crossover2] / ch, color='r', s=0.7)
        plt.scatter(kmp[under_o_cyclotprn], w[under_o_cyclotprn] / ch, color='b', s=0.7)

        plt.scatter(kpp[over_crossover_frq1], w[over_crossover_frq1] / ch, color='r', s=0.7)
        plt.scatter(kpp[he_cyclotron_to_crossover1], w[he_cyclotron_to_crossover1] / ch, color='b', s=0.7)
        plt.scatter(kpp[crossover2_to_he_cyclotron], w[crossover2_to_he_cyclotron] / ch, color='r', s=0.7)
        plt.scatter(kpp[o_cyclotron_to_crossover2], w[o_cyclotron_to_crossover2] / ch, color='b', s=0.7)
        plt.scatter(kpp[under_o_cyclotprn], w[under_o_cyclotprn] / ch, color='r', s=0.7)

    # クロスオーバー周波数を横線で表示
    # plt.axhline(y=crossover_frequency1 / ch, color='k', linestyle='--')
    plt.axhline(crossover_frequency1 / ch, color='y', linestyle='--')
    plt.axhline(0.09, color='y', linestyle='--')

    # 各粒子のサイクロトロン周波数をプロット
    # for element in elements:
    # plasma_cyclotrons = cyclotron_freq[element] / ch
    # plt.axhline(y=plasma_cyclotrons, linestyle='dashed', alpha=0.5, color='k', label=f'{elements[element]["name"]} Cyclotron frequency')

    plt.axhline(0.251, linestyle='dashed', alpha=0.5, color='k')
    plt.axhline(0.0625, linestyle='dashed', alpha=0.5, color='k')
    # 軸と凡例
    plt.xlim(xmin=2 * 10 ** -3, xmax=3 * 10 ** -1)
    plt.ylim(ymin=5 * 10 ** -2, ymax=10 ** 0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$V_Ak / \Omega_H$')
    plt.ylabel(r'$\omega / \Omega_{H}$')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    path.savefig(save_png)
    return


def matsuda():
    # ===== 設定定数 =====
    XAXIS = 0  # 0: Va k / Wh, 1: k/Wh*1e5
    YAXIS = 0
    V_C = 3.0e8
    PI = np.pi

    # ===== 物理定数・初期値 =====
    pf_e = 36720000.0
    cyc_h = 1.6021766208E-19 * 12397E-9 / 1.6726219E-27  # ~716.28
    cyc_e = cyc_h * 1836.0
    cyc_d = cyc_h / 2.0
    cyc_t = cyc_h / 3.0
    cyc_o = cyc_h / 16.0
    cyc_o2 = cyc_h / (16.0 / 2.0)
    cyc_o5 = cyc_h / (16.0 / 5.0)
    cyc_he = cyc_h / 4.0
    cyc_hepp = cyc_h / 2.0


    # ===== 分散関係計算 =====
    def get_graph(deg, param):
        a_he = param
        a_hepp = 0.00
        a_d = 0.00
        a_o = 0.00
        a_o2 = 0.00
        a_o5 = 0.000
        a_t = 0.00
        a_h = 1 - a_he - a_hepp - a_t - a_d - a_o - a_o2 - a_o5

        pf_h = np.sqrt((a_h * pf_e**2) / 1836.0)
        pf_d = np.sqrt((a_d * pf_e**2) / (2.0 * 1836.0))
        pf_hepp = np.sqrt((2 * a_hepp * pf_e**2) / (4.0 * 1836.0))
        pf_t = np.sqrt((a_t * pf_e**2) / (3.0 * 1836.0))
        pf_he = np.sqrt((a_he * pf_e**2) / (4.0 * 1836.0))
        pf_o = np.sqrt((a_o * pf_e**2) / (16.0 * 1836.0))
        pf_o5 = np.sqrt((5 * a_o5 * pf_e**2) / ((5.0 / 16.0) * 1836.0))
        pf_o2 = np.sqrt((2 * a_o2 * pf_e**2) / ((2.0 / 16.0) * 1836.0))

        factor = cyc_h / (2 * PI) if YAXIS == 1 else 1.0

        x_data, y_data = [], []
        dwdk_x, dwdk_y = [], []

        before_w = -1
        before_k = -1

        for w in np.arange(0.01, 2000, 0.01):
            x_e = pf_e ** 2 / w ** 2
            y_e = cyc_e / w
            x_h = pf_h ** 2 / w ** 2
            y_h = cyc_h / w
            x_he = pf_he ** 2 / w ** 2
            y_he = cyc_he / w
            x_hepp = pf_hepp ** 2 / w ** 2
            y_hepp = cyc_hepp / w
            x_d = pf_d ** 2 / w ** 2
            y_d = cyc_d / w
            x_o = pf_o ** 2 / w ** 2
            y_o = cyc_o / w
            x_o2 = pf_o2 ** 2 / w ** 2
            y_o2 = cyc_o2 / w

            r = 1.0 - (x_e / (1.0 - y_e)) - (x_h / (1.0 + y_h)) - (x_he / (1.0 + y_he)) \
                - (x_hepp / (1.0 + y_hepp)) - (x_d / (1.0 + y_d)) - (x_o / (1.0 + y_o)) - (x_o2 / (1.0 + y_o2))
            l = 1.0 - (x_e / (1.0 + y_e)) - (x_h / (1.0 - y_h)) - (x_he / (1.0 - y_he)) \
                - (x_hepp / (1.0 - y_hepp)) - (x_d / (1.0 - y_d)) - (x_o / (1.0 - y_o)) - (x_o2 / (1.0 - y_o2))

            p = 1 - x_e - x_h - x_he
            s = (r + l) / 2.0
            d = (r - l) / 2.0

            rad = deg * PI / 180.0
            sin2 = np.sin(rad)**2
            cos2 = np.cos(rad)**2

            a = s * sin2 + p * cos2
            b = r * l * sin2 + p * s * (1 + cos2)
            c = p * r * l
            f = np.sqrt((r * l - p * s) ** 2 * sin2 ** 2 + (2 * p * d * np.cos(rad)) ** 2)

            try:
                n2_1 = (b + f) / (2 * a)
                n2_2 = (b - f) / (2 * a)
            except ZeroDivisionError:
                continue

            for n2 in [n2_1, n2_2]:
                if n2 <= 0:
                    continue
                n = np.sqrt(n2)
                if XAXIS == 0:
                    x_val = (9.2252E-4 * n * w) / cyc_h * factor
                else:
                    x_val = (n * w) / cyc_h / V_C * 1.0E5 * factor
                y_val = w / cyc_h * factor

                x_data.append(x_val)
                y_data.append(y_val)

                if before_w > 0 and before_k > 0:
                    dwdk = (w - before_w) / (n * w - before_k)
                    if dwdk > 0:
                        dwdk_x.append(dwdk)
                        dwdk_y.append(y_val)

                before_w = w
                before_k = n * w

                if 0.999995 < x_val < 1.000005:
                    print(f"deg = {deg} (k = {x_val}) w = {y_val}")

        print(f"deg = {deg} (k = {x_val}) w = {y_val}")
        return x_data, y_data, dwdk_x, dwdk_y


    # ===== 実行部分 =====
    def main():
        while True:
            try:
                param = float(input("please input a parameter: "))
            except ValueError:
                continue

            fig, axs = plt.subplots(2, 1, figsize=(8, 10))

            for deg, color in zip([0, 90], ["red", "blue"]):
                x_data, y_data, dwdk_x, dwdk_y = get_graph(deg, param)
                axs[0].plot(x_data, y_data, label=f"theta={deg}", color=color)
                axs[1].plot(dwdk_x, dwdk_y, label=f"theta={deg}", color=color)

            axs[0].set_xlabel(r"$V_A k / \Omega_{H^+}$" if XAXIS == 0 else r"$k \times 10^5 / \Omega_{H^+}$")
            axs[0].set_ylabel(r"$\omega / \Omega_{H^+}$")
            axs[0].legend()
            axs[0].grid(True)
            axs[0].set_xlim([0, 1])
            axs[0].set_ylim([0, 1])

            axs[1].set_xlabel("Group velocity (dw/dk)")
            axs[1].set_ylabel(r"$\omega / \Omega_{H^+}$")
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            path.savefig('/Users/kikuchiriku/PythonPyCharm/Tohoku-Univ./pparc/erg/out/ErgAnalysis2/test/dispersion_relation_norm_matsuda.png')

            input("Press Enter to continue or Ctrl+C to exit...")
    main()


def xy_parameter(
        omega: np.ndarray,
        density: np.ndarray,
        charge_sign: np.ndarray,
        amu: np.ndarray,
        mag
):
    """
    TBI
    ------
    * omegaとdensityなどは次元が異なる -> どのように計算すべき? X, Yは(n_omega, n_ion)のようにする？

    :param omega: (n,)
    :param density: (n_ions,)
    :param charge_sign: (n_ions,)
    :param amu: (n_ions,)
    :param mag:
    :return: X, Y (n, n_ions)
    """
    charge = charge_sign * quant.e
    mass = amu * quant.mp
    omega_p = np.sqrt(density * charge ** 2 / quant.epsilon0 / mass)
    omega_c = charge * mag / mass
    X= np.zeros((len(omega), len(density)))
    Y= np.zeros((len(omega), len(density)))
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            X[i, j] = (omega_p[j] / omega[i]) ** 2
            Y[i, j] = omega_c[j] / omega[i]
    return X, Y


def rlp_parameter(
        X: np.ndarray,
        Y: np.ndarray,
        charge_sign
):
    """

    :param X: (n_omega, n_ions)
    :param Y: (n_omega, n_ions)
    :param charge_sign: (n_ions,)
    :return: R, L, P (n_omega,)
    """
    n_omega, n_ions = np.shape(X)
    r_comp = np.zeros(n_omega)
    l_comp = np.zeros(n_omega)
    for i in range(n_ions):
        r_comp = r_comp + X[:, i] / (np.ones(n_omega) + charge_sign[i] * np.abs(Y[:, i]))
        l_comp = l_comp + X[:, i] / (np.ones(n_omega) - charge_sign[i] * np.abs(Y[:, i]))
    R = np.ones(n_omega) - r_comp
    L = np.ones(n_omega) - l_comp
    P = np.ones(n_omega) - np.sum(X, axis=1)
    return R, L, P






def plot_dispersion_rel(dict_ions):
    # Ion parameters
    amu = []
    charge_sign = []
    density = []
    for ion, params in dict_ions.items():
        amu.append(params["amu"])
        charge_sign.append(params["charge sign"])
        density.append(params["density"])
    amu = np.array(amu)
    charge_sign = np.array(charge_sign)
    density = np.array(density)

    # Frequency range
    omega = np.logspace(-2, 1, 10000)  # rad/s
    B = 1e-9  # Tesla

    # Calculate X, Y, R, L, P
    X, Y = xy_parameter(omega, density, charge_sign, amu, B)
    R, L, P = rlp_parameter(X, Y, charge_sign)
    n2r, n2l = squared_refractive_index(R, L, P, theta_deg=89.9)  # quasi-perpendicular

    # Derived quantities
    Omega_H = quant.e * B / quant.mp
    n_p = density[0]  # assume H+ is first
    V_A = B / np.sqrt(quant.mu0 * quant.mp * n_p)

    k2r = omega ** 2 / quant.c ** 2 * n2r
    k2l = omega ** 2 / quant.c ** 2 * n2l
    kr = np.sqrt(np.maximum(k2r, 0))
    kl = np.sqrt(np.maximum(k2l, 0))

    # Normalized axes
    x_kr = V_A * kr / Omega_H
    x_kl = V_A * kl / Omega_H
    y = omega / Omega_H

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_kr, y, label="R-mode")
    plt.plot(x_kl, y, label="L-mode")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(r"$V_A k / \Omega_{H^+}$")
    plt.ylabel(r"$\omega / \Omega_{H^+}$")
    plt.title("Dispersion Relation (ω–k diagram)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.xlim([0, 1.5])
    plt.ylim([0, 1])
    plt.show()


# if __name__ == "__main__":
#     dict_ions = {
#         "e-": {"amu": 1 / 1800, "charge sign": -1, "density": 1},
#         "H+": {"amu": 1, "charge sign": 1, "density": .8},
#         "He+": {"amu": 4, "charge sign": 1, "density": .2},
#         "O+": {"amu": 16, "charge sign": 1, "density": 0},
#     }
#     plot_dispersion_rel(dict_ions)




