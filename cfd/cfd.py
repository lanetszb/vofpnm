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
import json
import pandas as pd
import copy
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))

from netgrid import save_files_collection_to_file
from vofpnm.cfd.ini_class import Ini
from vofpnm.cfd.cfd_class import Cfd
from vofpnm.helpers import plot_rel_perms, plot_conesrvation_check, plot_viscs_vels, plot_av_sat, \
    plot_capillary_pressure_curve, plot_capillary_pressures

ini = Ini(config_file=sys.argv[1])

cfd = Cfd(ini)

visc_0 = ini.paramsPnm['visc_0']
visc_1 = ini.paramsPnm['visc_1']
ini.throats_viscs = np.tile(visc_0, ini.netgrid.throats_N)
cfd.run_pnm()

# ### validation with openFoam ###
two_phase_data = dict()
times_labels_alphas = dict()
times_labels_u_mgns = dict()

thrs_velocities_to_output = dict()
thrs_alphas_to_output = dict()

thrs_to_label = np.array([1, 2, 3, 4, 5, 6], dtype=int)
thrs_to_output = np.array([0, 3, 6, 11, 12, 13], dtype=int)

nus = {'1': visc_0, '2': visc_1}
rhos = {'1': ini.paramsPnm['b_dens_fluid1'], '2': ini.paramsPnm['b_dens_fluid1']}
two_phase_data['mus'] = nus
two_phase_data['rhos'] = rhos
two_phase_data['sigma'] = ini.ift

times_labels_u_mgns_one_phase = dict()
times_u_mgn_avs_one_phase = dict()

one_phase_data = dict()
one_phase_data['mu'] = nus['1']
one_phase_data['rho'] = rhos['1']

# ### validation with openfoam one-phase ###
for i in range(len(thrs_to_output)):
    thrs_velocities_to_output[str(thrs_to_label[i])] = np.abs(cfd.ini.throats_velocities[
                                                                  thrs_to_output[i]])
    thrs_alphas_to_output[str(thrs_to_label[i])] = cfd.ini.equation.throats_av_sats[
        thrs_to_output[i]]

throats_volumes = cfd.ini.throats_volumes
throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))

vels_by_vols = []
for throat in thrs_to_output:
    vels_by_vols.append(throats_volumes[throat] * throats_vels[throat])
av_vel = np.sum(np.array(vels_by_vols)) / np.sum(throats_volumes)

throats_widths = np.absolute(np.array(list(cfd.ini.throats_widths.values())))
width_by_vols = []
for throat in thrs_to_output:
    width_by_vols.append(throats_volumes[throat] * throats_widths[throat])
av_width = np.sum(np.array(width_by_vols)) / np.sum(throats_volumes)

one_phase_data['width'] = av_width
two_phase_data['width'] = av_width

times_labels_u_mgns_one_phase[str(0)] = thrs_velocities_to_output
times_u_mgn_avs_one_phase[str(0)] = av_vel
one_phase_data['times_labels_u_mgns'] = times_labels_u_mgns_one_phase
one_phase_data['times_u_mgn_av'] = av_vel

json_one_phase_data = 'inOut/validation/one_phase_data23.json'

with open(json_one_phase_data, 'w') as f:
    json.dump(one_phase_data, f, sort_keys=True, indent=4 * ' ', ensure_ascii=False)
# ### validation with openfoam one-phase ###

ini.flow_0_ref = cfd.calc_rel_flow_rate()

visc_1 = ini.paramsPnm['visc_1']
ini.throats_viscs = np.tile(visc_1, ini.netgrid.throats_N)
cfd.run_pnm()
ini.flow_1_ref = cfd.calc_rel_flow_rate()

cfd.calc_coupling_params()
cfd.run_pnm()

rel_perms_0 = []
rel_perms_1 = []
capillary_numbers = []
capillary_pressures = []
av_sats = []

throats_volumes = cfd.ini.throats_volumes
throats_av_sats = cfd.ini.equation.throats_av_sats
dens_0 = cfd.ini.paramsPnm['dens_0']
mass_already_in = copy.deepcopy(np.sum(throats_volumes * throats_av_sats * dens_0))

mass_rates_in = []
mass_rates_out = []
masses_inside = []

times = []
viscs = []
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
    cfd.calc_rel_perms(rel_perms_0, rel_perms_1, capillary_numbers, capillary_pressures,
                       av_sats, ini.flow_0_ref, ini.flow_1_ref, vol_rate_in_0)

    # cfd.calc_kunni_perms(rel_perms_0, rel_perms_1, capillary_numbers,
    #                      av_sats, ini.flow_0_ref, ini.flow_1_ref, vol_rate_in_0, vol_rate_in_1)

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

        ####### validation with openfoam #######
        for i in range(len(thrs_to_output)):
            thrs_velocities_to_output[str(thrs_to_label[i])] = np.abs(cfd.ini.throats_velocities[
                                                                          thrs_to_output[i]])
            thrs_alphas_to_output[str(thrs_to_label[i])] = cfd.ini.equation.throats_av_sats[
                thrs_to_output[i]]

        times_labels_alphas[str(round(time_curr, round_output_time))] = copy.deepcopy(thrs_alphas_to_output)
        times_labels_u_mgns[str(round(time_curr, round_output_time))] = copy.deepcopy(thrs_velocities_to_output)
        ####### validation with openfoam #######

    throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))
    throats_viscs = cfd.ini.throats_viscs
    visc = np.sum(cfd.ini.throats_volumes * throats_viscs) / np.sum(cfd.ini.throats_volumes)
    times.append(time_curr)
    viscs.append(visc)
    vol_rates_in.append(vol_rate_in)
    if is_last_step:
        break

#############
# Validation output
#############
two_phase_data['times_labels_alphas'] = times_labels_alphas
two_phase_data['times_labels_u_mgns'] = times_labels_u_mgns

json_file_u_mgns = 'inOut/validation/two_phase_data.json'

with open(json_file_u_mgns, 'w') as f:
    json.dump(two_phase_data, f, sort_keys=True, indent=4 * ' ', ensure_ascii=False)

#############
# Plotting
#############

# rel perms
# fig1, ax1 = plt.subplots()
# plot_rel_perms(ax1, av_sats, rel_perms_0, rel_perms_1,
#                capillary_numbers)
# # capillary pressure
# fig2, axs = plt.subplots(1, 2, sharey='all')
# plot_capillary_pressure_curve(axs[0], av_sats, capillary_pressures)
# plot_capillary_pressures(axs[1], np.min(capillary_pressures),
#                          cfd.calc_throat_capillary_pressure_curr)
# # conservation
# mass_in_accum = np.cumsum(np.array(mass_rates_in) * np.array(time_steps))
# mass_out_accum = np.cumsum(np.array(mass_rates_out) * np.array(time_steps))
# massrates_net_accum = mass_in_accum - mass_out_accum
#
# masses_inside_accum = np.array(masses_inside) - mass_already_in
#
# mass_in_curr = np.array(mass_rates_in)
# mass_out_curr = np.array(mass_rates_out)
# massrates_net_curr = mass_in_curr - mass_out_curr
# #
# fig, axs = plt.subplots(3, sharex='all')
# plot_conesrvation_check(axs[0], times, massrates_net_accum,
#                         masses_inside_accum, massrates_net_curr)
#
# # viscosities velocities
# plot_viscs_vels(axs[1], times, viscs, vol_rates_in)
# #
# # average saturation
# plot_av_sat(axs[2], times, av_sats)

# Data output to csv
# df = pd.DataFrame({'sat': av_sats, 'rel_perms_0': rel_perms_0, 'rel_perms_1': rel_perms_1})
# df = df.set_index('sat')
# df.to_csv('inOut/rel_perms_no_gz.csv')
#
# df_accum = pd.DataFrame(
#     {'times': times, 'vols_out': np.cumsum(np.array(vol_rates_out) * np.array(time_steps))})
# df_accum = df_accum.set_index('times')
#
# df_accum.to_csv('inOut/vol_out_no_gz.csv')
