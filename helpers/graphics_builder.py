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

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))


def plot_rel_perms(ax, av_sats, rel_perms_0, rel_perms_2, ca_numbers):
    ax.plot(av_sats, rel_perms_0, ls="", marker="o", markersize=2,
            color="b", label='water')
    ax.plot(av_sats, rel_perms_2, ls="", marker="o", markersize=2,
            color="y", label='gas')
    ax1 = ax.twinx()

    ax1.plot(av_sats, ca_numbers, ls="", marker="o", markersize=2,
             label='Ca')
    ax.set_xlabel('Sw')
    ax.set_ylabel('Krw')
    ax1.set_ylabel('Ca')
    ax.legend(loc=2)
    ax1.legend(loc=1)


def plot_conesrvation_check(ax, times_accum, massrates_net_accum, masses_inside_accum,
                            massrates_net_curr):
    ax.plot(times_accum, massrates_net_accum, ls="", marker="o", markersize=2,
            color="b", label='massrate_net_accum')
    ax.set_xlabel('time')
    ax.set_ylabel('mass')
    ax.plot(times_accum, masses_inside_accum, ls="", marker="o", markersize=2,
            color="y", label='mass_inside_accum')
    ax1 = ax.twinx()
    ax1.plot(times_accum, massrates_net_curr, ls="", marker="o", markersize=2,
             color="k", label='massrate_net_curr')
    ax.legend(loc=2)
    ax1.legend(loc=1)


def plot_viscs_vels(ax, times, viscs, vels):
    ax.plot(times, viscs, ls="", marker="o", markersize=2,
            color="b", label='viscosities')
    ax1 = ax.twinx()
    ax1.plot(times, vels, ls="", marker="o", markersize=2,
             label='velocities')
    ax.set_xlabel('time')
    ax.set_ylabel('m/sec')
    ax1.set_ylabel('Pa*sec')
    ax.legend(loc=2)
    ax1.legend(loc=1)


def plot_av_sat(ax, times, av_sats):
    ax.plot(times, av_sats, ls="", marker="o", markersize=2,
            color="b", label='sat')
    ax.set_xlabel('time')
    ax.set_ylabel('s0')
    plt.legend()
