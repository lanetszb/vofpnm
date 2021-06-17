# MIT License
#
# Copyright (c) 2020 Aleksandr Zhuravlyov and Zakhar Lanets
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
import os
import matplotlib.pyplot as plt
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))


def plot_rel_perms(ax, av_sats, rel_perms_0, rel_perms_2, ca_numbers):
    ax.plot(av_sats, rel_perms_0, ls="", marker="o", markersize=2,
            color="tab:blue", label='0')
    ax.plot(av_sats, rel_perms_2, ls="", marker="o", markersize=2,
            color="tab:olive", label='1')
    ax1 = ax.twinx()

    # ax1.plot(av_sats, ca_numbers, ls="", marker="o", markersize=2,
    #          label='Ca')
    ax.set_xlabel('S0')
    ax.set_ylabel('Kr')
    # ax1.set_ylabel('Ca')
    ax.legend(loc=2)
    # ax1.legend(loc=1)


def plot_conesrvation_check(ax, times_accum, massrates_net_accum, masses_inside_accum,
                            massrates_net_curr):
    ax.plot(times_accum, massrates_net_accum, ls="", marker="o", markersize=2,
            color="tab:blue", label='massrate_net_accum')
    ax.set_xlabel('time, s')
    ax.set_ylabel('mass, kg')
    ax.plot(times_accum, masses_inside_accum, ls="", marker="o", markersize=2,
            color="tab:olive", label='mass_inside_accum')
    ax1 = ax.twinx()
    ax1.plot(times_accum, massrates_net_curr, ls="", marker="o", markersize=2,
             color="tab:purple", label='massrate_net_curr')
    ax1.set_ylabel('kg/sec')
    ax.legend(loc=2)
    ax1.legend(loc=1)


def plot_viscs_vels(ax, times, viscs, vels):
    ax.plot(times, viscs, ls="", marker="o", markersize=2,
            color="tab:blue", label='viscosities')
    ax.set_xlabel('time, s')
    ax.set_ylabel('Pa*s')
    ax1 = ax.twinx()
    ax1.plot(times, vels, ls="", marker="o", markersize=2,
             color="tab:olive", label='vol_rate_in')
    ax1.set_ylabel('m3/s')
    ax.legend(loc=2)
    ax1.legend(loc=1)


def plot_av_sat(ax, times, av_sats):
    ax.plot(times, av_sats, ls="", marker="o", markersize=2,
            color="tab:blue", label='sat')
    ax.set_xlabel('time, s')
    ax.set_ylabel('s0')
    ax.legend(loc=2)
    plt.legend()


def plot_capillary_pressures(ax, capillary_pressure_max, capillary_pressure_curr_func):
    sats_changes = np.linspace(-1.0, 1.0, 220)
    capillary_pressures = []

    for sat_change in sats_changes:
        capillary_pressures.append(capillary_pressure_curr_func(sat_change, capillary_pressure_max))

    ax.plot(sats_changes, capillary_pressures, ls="-", marker="o", markersize=0,
            color="tab:blue", label='capillary pressure')
    # ax.set_xlabel('$\Delta S$')
    # ax.set_ylabel('$b$')
    ax.set_xlabel('Delta S')
    ax.set_ylabel('b')
    # ax.legend(loc=2)
    # plt.legend()


def plot_capillary_pressure_curve(ax, av_sats, capillary_pressures):
    ax.plot(av_sats, capillary_pressures, ls="", marker="o", markersize=2,
            color="tab:purple", label='av_ca_pressure')
    ax.set_xlabel('S0')
    ax.set_ylabel('capillary pressure, Pa')
    ax.legend(loc=2)
    plt.legend()
