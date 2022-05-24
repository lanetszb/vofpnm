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
from matplotlib import rc

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))

from netgrid import save_files_collection_to_file
from matplotlib.ticker import FormatStrFormatter
from vofpnm.cfd.ini_class import Ini
from vofpnm.cfd.cfd_class import Cfd
from vofpnm.helpers import plot_rel_perms, plot_conesrvation_check, plot_viscs_vels, plot_av_sat, \
    plot_capillary_pressure_curve, plot_capillary_pressures

# rc('text', usetex=True)
# plt.rcParams["font.family"] = "Times New Roman"

start_time = tm.time()

# ini = Ini('config/config.ini')
ini = Ini(config_file=sys.argv[1])
ini.initialize_sats()
# ini = Ini(config_file=sys.argv[1])

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

thrs_velocities_to_output = dict()
thrs_alphas_to_output = dict()

nus = {'1': visc_0, '2': visc_1}
rhos = {'1': ini.paramsPnm['b_dens_fluid1'], '2': ini.paramsPnm['b_dens_fluid1']}
test_case_vofpnm['mus'] = nus
test_case_vofpnm['rhos'] = rhos
test_case_vofpnm['sigma'] = ini.ift

pc_max = 1.5 * max(ini.throats_capillary_pressures_max)
test_case_vofpnm['pc_max'] = pc_max

# ### validation with openfoam one-phase ###
throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))

u_mgn_av = np.sum((throats_volumes * throats_vels)) / np.sum(throats_volumes)

### calculating darcy porosity and permeability ###
vol_rate_in = 0
for throat in ini.inlet_throats:
    velocity = ini.throats_velocities[throat]
    area = ini.netgrid.throats_Ss[throat]
    density = ini.paramsPnm['dens_0']
    vol_rate_in += velocity * area

throat_inlet = list(ini.inlet_throats)[0]
throat_inlet_pore_1 = ini.throats_pores[throat_inlet][0]
throat_inlet_pore_2 = ini.throats_pores[throat_inlet][1]
throat_inlet_x_coord_min = min([ini.pores_coordinates[throat_inlet_pore_1][0],
                                ini.pores_coordinates[throat_inlet_pore_2][0]])

throat_outlet = list(ini.outlet_throats)[0]
throat_outlet_pore_1 = ini.throats_pores[throat_outlet][0]
throat_outlet_pore_2 = ini.throats_pores[throat_outlet][1]
throat_outlet_x_coord_max = max([ini.pores_coordinates[throat_outlet_pore_1][0],
                                 ini.pores_coordinates[throat_outlet_pore_2][0]])

L_z = 4.2e-6
L_y = 0.0025
vel_ref = vol_rate_in / L_y / L_z

u_mgn_x = 0
vols_by_vels_accum = 0
vols_accum = 0
for throat in range(ini.netgrid.throats_N):
    pore_1 = ini.throats_pores[throat][0]
    pore_2 = ini.throats_pores[throat][1]
    pore_1_x_coord = ini.pores_coordinates[pore_1][0]
    pore_2_x_coord = ini.pores_coordinates[pore_2][0]
    if pore_1_x_coord != pore_2_x_coord:
        vols_by_vels_accum += throats_volumes[throat] * throats_vels[throat]
        vols_accum += throats_volumes[throat]
    if pore_1_x_coord == pore_2_x_coord:
        vols_by_vels_accum += 0
        vols_accum += throats_volumes[throat]
u_mgn_x = vols_by_vels_accum / vols_accum

poro = vel_ref / u_mgn_x

p_in_accum = 0
vol_rates_in_accum = 0
for throat in ini.inlet_throats:
    velocity = ini.throats_velocities[throat]
    area = ini.netgrid.throats_Ss[throat]
    vol_rate_in = velocity * area
    vol_rates_in_accum += vol_rate_in
    pore_1 = ini.throats_pores[throat][0]
    pore_2 = ini.throats_pores[throat][1]
    if ini.pores_coordinates[pore_1][0] <= ini.pores_coordinates[pore_2][0]:
        p_in_accum += list(ini.pnm.pressures)[pore_1] * vol_rate_in
    else:
        p_in_accum += list(ini.pnm.pressures)[pore_2] * vol_rate_in
pressure_in = p_in_accum / vol_rates_in_accum

L_x = throat_outlet_x_coord_max - throat_inlet_x_coord_min
dP = pressure_in - ini.pressure_out
permeability = vel_ref * visc_0 * L_x / dP

test_case_vofpnm['ref_u_mgn'] = u_mgn_av
print('ref_u_mgn', u_mgn_av)

throats_widths = np.absolute(np.array(list(cfd.ini.throats_widths.values())))
av_width = np.sum((throats_volumes * throats_widths)) / np.sum(throats_volumes)
test_case_vofpnm['width'] = av_width

ini.flow_0_ref = cfd.calc_rel_flow_rate()
print('flow_0_ref', ini.flow_0_ref)

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

times_u_mgn_x, times_pressure_in = [], []
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
    cfd.calc_coupling_params()

    mass_inside = copy.deepcopy(np.sum(throats_volumes * throats_av_sats * dens_0))
    masses_inside.append(mass_inside)

    vol_rate_in, vol_rate_out, vol_rate_in_0, vol_rate_out_1 = cfd.calc_flow_rates(mass_rates_in,
                                                                                   mass_rates_out)
    vol_rates_out.append(vol_rate_out_1)

    ### New params for history matching ###

    u_mgn_x = 0
    vols_by_vels_accum = 0
    vols_accum = 0
    for throat in range(ini.netgrid.throats_N):
        pore_1 = ini.throats_pores[throat][0]
        pore_2 = ini.throats_pores[throat][1]
        pore_1_x_coord = ini.pores_coordinates[pore_1][0]
        pore_2_x_coord = ini.pores_coordinates[pore_2][0]
        if pore_1_x_coord != pore_2_x_coord:
            vols_by_vels_accum += throats_volumes[throat] * throats_vels[throat]
            vols_accum += throats_volumes[throat]
        if pore_1_x_coord == pore_2_x_coord:
            vols_by_vels_accum += 0
            vols_accum += throats_volumes[throat]
    u_mgn_x = vols_by_vels_accum / vols_accum
    times_u_mgn_x.append(u_mgn_x)

    p_in_accum = 0
    vol_rates_in_accum = 0
    for throat in ini.inlet_throats:
        velocity = ini.throats_velocities[throat]
        area = ini.netgrid.throats_Ss[throat]
        vol_rate_in = velocity * area
        vol_rates_in_accum += vol_rate_in
        pore_1 = ini.throats_pores[throat][0]
        pore_2 = ini.throats_pores[throat][1]
        if ini.pores_coordinates[pore_1][0] <= ini.pores_coordinates[pore_2][0]:
            p_in_accum += list(ini.pnm.pressures)[pore_1] * vol_rate_in
        else:
            p_in_accum += list(ini.pnm.pressures)[pore_2] * vol_rate_in
    pressure_in = p_in_accum / vol_rates_in_accum
    times_pressure_in.append(pressure_in)

    ### New params for history matching ###

    cfd.calc_rel_perms(rel_perms_0, rel_perms_1, capillary_numbers, capillary_pressures,
                       av_sats, ini.flow_0_ref, ini.flow_1_ref, vol_rate_in_0)

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
        throats_vels = np.absolute(np.array(list(cfd.ini.throats_velocities.values())))
        u_mgn_av = np.sum(throats_volumes * throats_vels) / np.sum(throats_volumes)
        alpha_av = np.sum(throats_volumes * throats_av_sats) / np.sum(throats_volumes)
        F_av = np.sum(throats_volumes * throats_vels * throats_av_sats) / np.sum(
            throats_volumes * throats_vels)

        times_u_mgn_avs[str(round(time_curr, round_output_time))] = u_mgn_av
        times_alpha_avs[str(round(time_curr, round_output_time))] = alpha_av
        times_F_avs[str(round(time_curr, round_output_time))] = F_av
        times_F_avs_new[str(round(time_curr, round_output_time))] = (vol_rate_out - vol_rate_out_1) / vol_rate_out
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
test_case_vofpnm['poro'] = poro
test_case_vofpnm['permeability'] = permeability
test_case_vofpnm['times_alpha_avs'] = times_alpha_avs
test_case_vofpnm['times_u_mgn_avs'] = times_u_mgn_avs
test_case_vofpnm['times_u_mgn_x'] = times_u_mgn_x
test_case_vofpnm['times_pressure_in'] = times_pressure_in
test_case_vofpnm['times_F_avs'] = times_F_avs
test_case_vofpnm['execution_time'] = execution_time
test_case_vofpnm['time_step'] = cfd.ini.output_time_step
test_case_vofpnm['grid_volume'] = cfd.ini.grid_volume
test_case_vofpnm['total_volume'] = np.sum(throats_volumes)
test_case_vofpnm['times_V_in'] = times_V_in

json_file_u_mgns = 'inOut/validation/model_chess_pnm_2_imb.json'

with open(json_file_u_mgns, 'w') as f:
    json.dump(test_case_vofpnm, f, sort_keys=False, indent=4 * ' ', ensure_ascii=False)
