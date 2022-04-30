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
import numpy as np
import copy
import matplotlib.pyplot as plt
import time as tm
from matplotlib import rc

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))

from netgrid import save_files_collection_to_file
from vofpnm.cfd.ini_class import Ini
from vofpnm.cfd.cfd_class import Cfd
from vofpnm.helpers import plot_conesrvation_check, plot_viscs_vels, plot_av_sat

plt.style.use('dracula')

# rc('text', usetex=True)
# plt.rcParams["font.family"] = "Times New Roman"

start_time = tm.time()

ini = Ini('config/config.ini')
ini.initialize_sats()

cfd = Cfd(ini)

visc_0 = ini.paramsPnm['visc_0']
visc_1 = ini.paramsPnm['visc_1']
ini.throats_viscs = np.tile(visc_0, ini.netgrid.throats_N)
cfd.run_pnm()

throats_volumes = cfd.ini.throats_volumes
throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))

vels_by_vols = []
av_vel = np.sum(np.array(vels_by_vols)) / np.sum(throats_volumes)

throats_widths = np.absolute(np.array(list(cfd.ini.throats_widths.values())))

ini.flow_0_ref = cfd.calc_rel_flow_rate()

visc_1 = ini.paramsPnm['visc_1']
ini.throats_viscs = np.tile(visc_1, ini.netgrid.throats_N)
cfd.run_pnm()
ini.flow_1_ref = cfd.calc_rel_flow_rate()

cfd.calc_coupling_params()
cfd.run_pnm()

throats_volumes = cfd.ini.throats_volumes
throats_av_sats = cfd.ini.equation.throats_av_sats
dens_0 = cfd.ini.paramsPnm['dens_0']
mass_already_in = copy.deepcopy(np.sum(throats_volumes * throats_av_sats * dens_0))

mass_rates_in = []
mass_rates_out = []
masses_inside = []

times = []
viscs = []
av_sats = []
vol_rates_in = []
vol_rates_out = []

#################
# Paraview output
#################
os.system('rm -r inOut/*.vtu')
os.system('rm -r inOut/*.pvd')
sats_dict = dict()
file_name = 'inOut/collection.pvd'
files_names = list()
files_descriptions = list()

cells_arrays = cfd.process_paraview_data()
cfd.ini.netgrid.cells_arrays = cells_arrays
files_names.append(str(0) + '.vtu')
files_descriptions.append(str(0))
cfd.ini.netgrid.save_cells('inOut/' + files_names[-1])
save_files_collection_to_file(file_name, files_names, files_descriptions)
#################
#################

time = [0]
time_steps = []
cour_number = np.empty([])
time_curr = 0
time_step_curr = 0
time_output_freq = cfd.ini.time_period / 500.
round_output_time = int(ini.round_output_time)
output_time_step = ini.output_time_step
time_bound = output_time_step
is_output_step = False
is_last_step = False
out_idx = int(0)
while True:

    if cfd.ini.time_step_type == 'const':
        cfd.ini.time_step = cfd.ini.const_time_step
    elif cfd.ini.time_step_type == 'flow_variable':
        cfd.ini.time_step = cfd.ini.local.calc_flow_variable_time_step(
            cfd.ini.throats_velocities)
    elif cfd.ini.time_step_type == 'div_variable':
        cfd.ini.time_step = cfd.ini.local.calc_div_variable_time_step(
            cfd.ini.equation.sats[cfd.ini.equation.i_curr], cfd.ini.throats_velocities)

    time_step_curr = cfd.ini.time_step

    if time_curr + time_step_curr >= time_bound:
        time_step_curr = time_bound - time_curr
        time_bound += output_time_step
        is_output_step = True

    if time_curr + time_step_curr >= cfd.ini.time_period:
        is_last_step = True
        if not is_output_step:
            time_step_curr = cfd.ini.time_period - time_curr

    time_steps.append(time_step_curr)
    time_curr += time_step_curr

    cfd.ini.equation.cfd_procedure_one_step(cfd.ini.throats_velocities, time_step_curr)
    cfd.calc_coupling_params()

    mass_inside = copy.deepcopy(np.sum(throats_volumes * throats_av_sats * dens_0))
    masses_inside.append(mass_inside)

    vol_rate_in, vol_rate_out, vol_rate_in_0, vol_rate_out_1 = cfd.calc_flow_rates(mass_rates_in,
                                                                                   mass_rates_out)
    vol_rates_out.append(vol_rate_out_1)

    throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))
    throats_viscs = cfd.ini.throats_viscs
    visc = np.sum(cfd.ini.throats_volumes * throats_viscs) / np.sum(cfd.ini.throats_volumes)
    viscs.append(visc)
    vol_rates_in.append(vol_rate_in)
    cfd.calc_av_sat(av_sats)

    times.append(time_curr)

    print('time_step: ', round(time_step_curr, round_output_time))
    time.append(time_curr)
    cfd.ini.equation.print_cour_numbers(cfd.ini.throats_velocities, cfd.ini.time_step)
    print(' percentage executed:', round((time_curr / cfd.ini.time_period * 100.), 2), '%.', '\n')

    cfd.run_pnm()

    cells_arrays = cfd.process_paraview_data()
    if is_output_step:
        cfd.ini.netgrid.cells_arrays = cells_arrays
        files_names.append(str(round(time_curr, round_output_time)) + '.vtu')
        files_descriptions.append(str(round(time_curr, round_output_time)))
        cfd.ini.netgrid.save_cells('inOut/' + files_names[-1])
        save_files_collection_to_file(file_name, files_names, files_descriptions)
        out_idx += 1
        is_output_step = False

    if is_last_step:
        break

execution_time = tm.time() - start_time
print("--- %s seconds ---" % execution_time)

#############
# Plotting
#############

# Plotting
fig_width = 3.5
y_scale = 0.9
# fig, ax = plt.subplots(figsize=(fig_width, fig_width * y_scale),
#                        tight_layout=True)
fig, axs = plt.subplots(3, sharex='all')

# conservation
mass_in_accum = np.cumsum(np.array(mass_rates_in) * np.array(time_steps))
mass_out_accum = np.cumsum(np.array(mass_rates_out) * np.array(time_steps))
massrates_net_accum = mass_in_accum - mass_out_accum

masses_inside_accum = np.array(masses_inside) - mass_already_in

mass_in_curr = np.array(mass_rates_in)
mass_out_curr = np.array(mass_rates_out)
massrates_net_curr = mass_in_curr - mass_out_curr
#

plot_conesrvation_check(axs[0], times, massrates_net_accum,
                        masses_inside_accum, massrates_net_curr)
#
# viscosities velocities
plot_viscs_vels(axs[1], times, viscs, vol_rates_in)
# average saturation
plot_av_sat(axs[2], times, av_sats)
