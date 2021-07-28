import sys
import os
import numpy as np
from scipy import stats
import json
import matplotlib.pyplot as plt
from matplotlib import rc
import collections

rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))


def param_to_smooth(x, y, radius):
    param_smoothed = []
    if radius == 0:
        param_smoothed = y
    else:
        radius = radius
        for i in range(radius):
            param_smoothed.append(y[i])
        for i in range(radius, len(y) - radius):
            slope = stats.linregress(x[i - radius: i + 1 + radius],
                                     y[i - radius: i + 1 + radius]).slope
            intercept = stats.linregress(x[i - radius: i + 1 + radius],
                                         y[i - radius: i + 1 + radius]).intercept
            param_smoothed.append(intercept + slope * x[i])
        for i in range(len(y) - radius, len(y)):
            param_smoothed.append(y[i])
    param_to_smooth = np.array(param_smoothed)

    return param_to_smooth


def process_data_vof(json_vof_name, smooth_radius):
    with open(json_vof_name) as f:
        data_vof = json.load(f)

    ref_vel_vof = data_vof['ref_u_mgn']
    f_0 = sorted(data_vof['times_F_avs'].items(), key=lambda x: float(x[0]))
    f_0 = [seq[1] for seq in f_0]
    av_vels = sorted(data_vof['times_u_mgn_avs'].items(), key=lambda x: float(x[0]))
    av_vels = [seq[1] for seq in av_vels]
    s_0_av = sorted(data_vof['times_alpha_avs'].items(), key=lambda x: float(x[0]))
    s_0_av = [seq[1] for seq in s_0_av]

    mu_0 = data_vof['nus']['1'] * data_vof['rhos']['1']
    mu_1 = data_vof['nus']['2'] * data_vof['rhos']['2']
    ref_mu = mu_1
    mu_ratio = mu_0 / mu_1
    gamma = data_vof['sigma']

    x = np.arange(len(s_0_av))
    s_0_av_smoothed = param_to_smooth(x, s_0_av, smooth_radius)
    s_0_av = s_0_av_smoothed

    av_vels_smoothed = param_to_smooth(s_0_av, av_vels, smooth_radius)
    av_vels = av_vels_smoothed

    fs_0_smoothed = param_to_smooth(s_0_av, f_0, smooth_radius)
    f_0 = fs_0_smoothed

    kr_1 = (1. - f_0) * av_vels / ref_vel_vof * (mu_1 / ref_mu)
    kr_0 = (mu_0 / mu_1) * (f_0 / (1. - f_0)) * kr_1

    return kr_0, kr_1, s_0_av, f_0, av_vels, mu_ratio, gamma


def process_data_pnm(data_vofpnm, smooth_radius):
    ref_vel_pnm = data_vofpnm['ref_u_mgn']
    f_0 = sorted(data_vofpnm['times_F_avs'].items(), key=lambda x: float(x[0]))
    f_0 = [seq[1] for seq in f_0]
    av_vels = sorted(data_vofpnm['times_u_mgn_avs'].items(), key=lambda x: float(x[0]))
    av_vels = [seq[1] for seq in av_vels]
    s_0_av = sorted(data_vofpnm['times_alpha_avs'].items(), key=lambda x: float(x[0]))
    s_0_av = [seq[1] for seq in s_0_av]

    mu_0 = data_vofpnm['mus']['1']
    mu_1 = data_vofpnm['mus']['2']
    ref_mu = mu_1
    mu_ratio = mu_0 / mu_1
    gamma = data_vofpnm['sigma']

    x = np.arange(len(s_0_av))
    s_0_av_smoothed = param_to_smooth(x, s_0_av, smooth_radius)
    s_0_av = s_0_av_smoothed

    av_vels_smoothed = param_to_smooth(s_0_av, av_vels, smooth_radius)
    av_vels = av_vels_smoothed

    fs_0_smoothed = param_to_smooth(s_0_av, f_0, smooth_radius)
    f_0 = fs_0_smoothed

    kr_1 = (1. - f_0) * av_vels / ref_vel_pnm * (mu_1 / ref_mu)
    kr_0 = (mu_0 / mu_1) * (f_0 / (1. - f_0)) * kr_1

    return kr_0, kr_1, s_0_av, f_0, av_vels, mu_ratio, gamma


def plot_rel_perms(ax, s_0, kr_0, kr_1, f_0, case_name):
    lns1 = ax.plot(s_0, f_0, ls="", marker="o", markersize=2,
                   color="black", label='$f^{\mathit0}$')
    lns2 = ax.plot(s_0, kr_0, ls="", marker="o", markersize=2,
                   color="tab:blue", label='$k_r^{\mathit0}$')
    lns3 = ax.plot(s_0, kr_1, ls="", marker="o", markersize=2,
                   color="tab:green", label='$k_r^{\mathit1}$')

    ax.set_xlabel('$S^{\mathit0}$')
    ax.set_ylabel('$k_r, f^{\mathit0}$')

    # plt.gca().set_ylim(bottom=0)
    # plt.gca().set_ylim(top=1)
    plt.xlim(0, 1)

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=9)

    plt.title(case_name)
    plt.show()


def compare_rel_perms(ax, s_0_vof, s_0_vofpnm, kr_0_vof, kr_1_vof, kr_0_vofpnm, kr_1_vofpnm,
                      case_name):
    lns1 = ax.plot(s_0_vof, kr_0_vof, ls="", marker="o", markersize=2,
                   color="tab:blue", label='vof')
    lns2 = ax.plot(s_0_vof, kr_1_vof, ls="", marker="o", markersize=2,
                   color="tab:blue")
    lns3 = ax.plot(s_0_vofpnm, kr_0_vofpnm, ls="", marker="o", markersize=2,
                   color="tab:orange", label='vofpnm')
    lns4 = ax.plot(s_0_vofpnm, kr_1_vofpnm, ls="", marker="o", markersize=2,
                   color="tab:orange")

    ax.set_xlabel('$S^{\mathit0}$')
    ax.set_ylabel('$k_r$')

    plt.gca().set_ylim(bottom=0)
    # plt.gca().set_ylim(top=1)
    plt.xlim(0, 1)

    lns = lns1 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=9)

    plt.title(case_name)
    plt.show()
