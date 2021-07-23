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
import time as tm
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from matplotlib import rc

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))

from netgrid import save_files_collection_to_file
from matplotlib.ticker import FormatStrFormatter
from vofpnm.cfd.ini_class import Ini
from vofpnm.cfd.cfd_class import Cfd
from vofpnm.helpers.calc_relperms import param_to_smooth, process_data_pnm, process_data_vof, \
    plot_rel_perms, compare_rel_perms

# rc('text', usetex=True)
# plt.rcParams["font.family"] = "Times New Roman"
ini = Ini('config/config.ini')
results = dict()


def get_error(power_coeff):
    print('power_coeff', power_coeff)
    ini.initialize_sats()
    json_vof_name = 'demo_m1_ift_0.001_dp_200_drainage_vof.json'
    start_time = tm.time()

    cfd = Cfd(ini)

    visc_0 = ini.paramsPnm['visc_0']
    visc_1 = ini.paramsPnm['visc_1']
    ini.throats_viscs = np.tile(visc_0, ini.netgrid.throats_N)
    cfd.run_pnm()
    throats_volumes = cfd.ini.throats_volumes

    # ### validation with openFoam ###
    test_case_vofpnm = dict()
    times_alpha_avs = dict()
    times_u_mgn_avs = dict()
    times_F_avs = dict()
    times_F_avs_new = dict()
    times_V_in = dict()

    nus = {'1': visc_0, '2': visc_1}
    rhos = {'1': ini.paramsPnm['b_dens_fluid1'], '2': ini.paramsPnm['b_dens_fluid1']}
    test_case_vofpnm['mus'] = nus
    test_case_vofpnm['rhos'] = rhos
    test_case_vofpnm['sigma'] = ini.ift

    # ### validation with openfoam one-phase ###
    throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))

    u_mgn_av = np.sum((throats_volumes * throats_vels)) / np.sum(throats_volumes)
    test_case_vofpnm['ref_u_mgn'] = u_mgn_av

    throats_widths = np.absolute(np.array(list(cfd.ini.throats_widths.values())))
    av_width = np.sum((throats_volumes * throats_widths)) / np.sum(throats_volumes)
    test_case_vofpnm['width'] = av_width

    ini.flow_0_ref = cfd.calc_rel_flow_rate()

    visc_1 = ini.paramsPnm['visc_1']
    ini.throats_viscs = np.tile(visc_1, ini.netgrid.throats_N)
    cfd.run_pnm()
    ini.flow_1_ref = cfd.calc_rel_flow_rate()

    cfd.calc_coupling_params(power_coeff)
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
        cfd.calc_coupling_params(power_coeff)

        # coeffs = copy.deepcopy(ini.equation.throats_sats_grads)
        # pcs_max = ini.throats_capillary_pressures_max
        # ini.throats_capillary_pressures = cfd.calc_throat_capillary_pressure_curr(coeffs,
        #                                                                           pcs_max,
        #                                                                           power_coeff)

        mass_inside = copy.deepcopy(np.sum(throats_volumes * throats_av_sats * dens_0))
        masses_inside.append(mass_inside)

        vol_rate_in, vol_rate_out, vol_rate_in_0, vol_rate_out_1 = cfd.calc_flow_rates(
            mass_rates_in,
            mass_rates_out)
        vol_rates_out.append(vol_rate_out_1)

        cfd.calc_rel_perms(rel_perms_0, rel_perms_1, capillary_numbers, capillary_pressures,
                           av_sats, ini.flow_0_ref, ini.flow_1_ref, vol_rate_in_0)

        print('time_step: ', round(time_step_curr, round_output_time))
        time.append(time_curr)
        cfd.ini.equation.print_cour_numbers(cfd.ini.throats_velocities, cfd.ini.time_step)
        print(' percentage executed:', round((time_curr / cfd.ini.time_period * 100.), 2), '%.',
              '\n')
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
            throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))
            u_mgn_av = np.sum(throats_volumes * throats_vels) / np.sum(throats_volumes)
            alpha_av = np.sum(throats_volumes * throats_av_sats) / np.sum(throats_volumes)
            F_av = np.sum(throats_volumes * throats_vels * throats_av_sats) / np.sum(
                throats_volumes * throats_vels)

            times_u_mgn_avs[str(round(time_curr, round_output_time))] = u_mgn_av
            times_alpha_avs[str(round(time_curr, round_output_time))] = alpha_av
            times_F_avs[str(round(time_curr, round_output_time))] = F_av
            times_F_avs_new[str(round(time_curr, round_output_time))] = \
                (vol_rate_out - vol_rate_out_1) / vol_rate_out
            times_V_in[str(round(time_curr, round_output_time))] = vol_rate_in
            ####### validation with openfoam #######
            print(str(round(time_curr, round_output_time)), time_curr)

        throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))
        throats_viscs = cfd.ini.throats_viscs
        visc = np.sum(cfd.ini.throats_volumes * throats_viscs) / np.sum(cfd.ini.throats_volumes)
        times.append(time_curr)
        viscs.append(visc)
        vol_rates_in.append(vol_rate_in)
        if is_last_step:
            break

    execution_time = tm.time() - start_time
    print("--- %s seconds ---" % execution_time)
    #############
    # Rel perms validation output
    #############
    test_case_vofpnm['times_alpha_avs'] = times_alpha_avs
    test_case_vofpnm['times_u_mgn_avs'] = times_u_mgn_avs
    test_case_vofpnm['times_F_avs'] = times_F_avs
    test_case_vofpnm['times_F_avs_new'] = times_F_avs_new
    test_case_vofpnm['execution_time'] = execution_time
    test_case_vofpnm['time_step'] = cfd.ini.time_step
    test_case_vofpnm['grid_volume'] = cfd.ini.grid_volume
    test_case_vofpnm['total_volume'] = np.sum(throats_volumes)
    test_case_vofpnm['times_V_in'] = times_V_in

    smooth_radius = 0
    kr_0_pnm, kr_1_pnm, s_0_pnm, f_0_pnm, av_vels_pnm, mu_ratio_pnm, gamma_pnm = \
        process_data_pnm(test_case_vofpnm, smooth_radius)

    json_vof = 'inOut/optimisation/' + json_vof_name
    kr_0_vof, kr_1_vof, s_0_vof, f_0_vof, av_vels_vof, mu_ratio_vof, gamma_vof = \
        process_data_vof(json_vof, smooth_radius)

    results[power_coeff] = {}
    results[power_coeff]['s_0_pnm'] = list(s_0_pnm)
    results[power_coeff]['kr_0_pnm'] = list(kr_0_pnm)
    results[power_coeff]['kr_1_pnm'] = list(kr_1_pnm)
    results[power_coeff]['s_0_vof'] = list(s_0_vof)
    results[power_coeff]['kr_0_vof'] = list(kr_0_vof)
    results[power_coeff]['kr_1_vof'] = list(kr_1_vof)
    # fig1, ax1 = plt.subplots()
    # plot_rel_perms(ax1, s_0_pnm, kr_0_pnm, kr_1_pnm, f_0_pnm, '')
    #
    # fig2, ax2 = plt.subplots()
    # plot_rel_perms(ax2, s_0_vof, kr_0_vof, kr_1_vof, f_0_vof, '')
    # fig3, ax3 = plt.subplots()
    # compare_rel_perms(ax3, s_0_vof, s_0_pnm, kr_0_vof, kr_1_vof, kr_0_pnm, kr_1_pnm, '')

    print('len(kr_0_vof)', len(kr_0_vof))
    print('len(kr_0_pnm)', len(kr_0_pnm))
    kr0_error = np.mean(abs(kr_0_vof - kr_0_pnm))
    kr1_error = np.mean(abs(kr_1_vof - kr_1_pnm))
    kr_error = (kr0_error + kr1_error) / 2.
    results[power_coeff]['kr_error'] = kr_error
    # sat_error = np.mean(abs(s_0_vof - s_0_pnm))
    # av_error = (kr_error + sat_error) / 2.
    print('kr_error', kr_error)

    return kr_error


res = minimize_scalar(get_error, bounds=(0.5, 5), method='Bounded', options={'maxiter': 5,
                                                                             'xatol': 1e-04,
                                                                             'disp': 3})
results_output = 'inOut/optimisation/test_output.json'

with open(results_output, 'w') as f:
    json.dump(results, f, sort_keys=False, indent=4 * ' ', ensure_ascii=False)
with open(results_output) as f:
    results = json.load(f)
#
for key in results:
    plt.plot(results[key]['s_0_pnm'], results[key]['kr_0_pnm'], color="blue", linestyle='dashed',
             linewidth=0.5)
    plt.plot(results[key]['s_0_pnm'], results[key]['kr_1_pnm'], color="blue", linestyle='dashed',
             linewidth=0.5)
    plt.plot(results[key]['s_0_vof'], results[key]['kr_0_vof'], color="red")
    plt.plot(results[key]['s_0_vof'], results[key]['kr_1_vof'], color="red")

min_kr_error = 100.
coeff_miner = str()

for key in results:
    if results[key]['kr_error'] < min_kr_error:
        min_kr_error = results[key]['kr_error']
        coeff_miner = key

plt.plot(results[coeff_miner]['s_0_pnm'], results[coeff_miner]['kr_0_pnm'], color="blue")
plt.plot(results[coeff_miner]['s_0_pnm'], results[coeff_miner]['kr_1_pnm'], color="blue")

# json_file_u_mgns = 'inOut/validation/demo_m1_ift_0.001_dp_200_drainage_vofpnm.json'
#
# with open(json_file_u_mgns, 'w') as f:
#     json.dump(test_case_vofpnm, f, sort_keys=False, indent=4 * ' ', ensure_ascii=False)
