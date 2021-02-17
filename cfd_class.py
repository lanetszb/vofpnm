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
import math
import copy
import configparser
import json
from ast import literal_eval
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../'))

from netgrid import Netgrid, save_files_collection_to_file
from vofpnm import Pnm
from vofpnm import Props, Boundary, Local, Convective, Equation


class Cfd:
    def __init__(s, config_file):
        s.__config = configparser.ConfigParser()
        s.__config.read(config_file)
        get = s.__config.get

        ################################
        # creating grid with Netgrid
        #################################

        s.pores_coordinates = literal_eval(s.__config['Properties_grid']['pores_coordinates'])
        s.throats_pores = literal_eval(s.__config['Properties_grid']['throats_pores'])
        s.throats_widths = literal_eval(s.__config['Properties_grid']['throats_widths'])
        s.throats_depths = literal_eval(s.__config['Properties_grid']['throats_depths'])
        s.inlet_pores = literal_eval(s.__config['Properties_grid']['inlet_pores'])
        s.outlet_pores = literal_eval(s.__config['Properties_grid']['outlet_pores'])
        s.inlet_throats = literal_eval(s.__config['Properties_grid']['inlet_throats'])
        s.outlet_throats = literal_eval(s.__config['Properties_grid']['outlet_throats'])
        s.delta_V = float(get('Properties_grid', 'delta_V'))
        s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))

        # json_file_name = str(get('Properties_grid', 'case_name'))
        # with open('inOut/' + json_file_name) as f:
        #     data = json.load(f)
        #
        # s.pores_coordinates = {int(key): value for key, value in data['pores_coordinates'].items()}
        # s.throats_pores = {int(key): value for key, value in data['throats_pores'].items()}
        # s.throats_widths = {int(key): value for key, value in data['throats_widths'].items()}
        # s.throats_depths = {int(key): value for key, value in data['throats_depths'].items()}
        #
        # s.inlet_pores = set(data['boundary_pores']['inlet_pores'])
        # s.outlet_pores = set(data['boundary_pores']['outlet_pores'])
        # s.inlet_throats = set(data['boundary_throats']['inlet_throats'])
        # s.outlet_throats = set(data['boundary_throats']['outlet_throats'])
        #
        # s.delta_V = float(get('Properties_grid', 'delta_V'))
        # s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))

        s.netgrid = Netgrid(s.pores_coordinates, s.throats_pores,
                            s.throats_widths, s.throats_depths, s.delta_V, s.min_cells_N,
                            s.inlet_pores, s.outlet_pores)
        #############
        # PNM
        #############

        s.paramsPnm = {'a_gas_dens': float(get('Properties_gas', 'a_gas_dens')),
                       'b_gas_dens': float(get('Properties_gas', 'b_gas_dens')),
                       'gas_visc': float(get('Properties_gas', 'gas_visc')),
                       'liq_dens': float(get('Properties_liquid', 'liq_dens')),
                       'liq_visc': float(get('Properties_liquid', 'liq_visc')),
                       'pressure_in': float(get('Properties_simulation', 'pressure_in')),
                       'pressure_out': float(get('Properties_simulation', 'pressure_out')),
                       'it_accuracy': float(get('Properties_simulation', 'it_accuracy')),
                       'solver_method': str(get('Properties_simulation', 'solver_method'))}

        s.pore_n = s.netgrid.pores_N
        s.throats_denss = np.tile(s.paramsPnm['b_gas_dens'], s.netgrid.throats_N)
        s.throats_viscs = np.tile(s.paramsPnm['gas_visc'], s.netgrid.throats_N)
        s.capillary_pressures = np.tile(0., s.netgrid.throats_N)

        s.newman_pores_flows = {}
        s.dirichlet_pores_pressures = {}
        for pore in s.inlet_pores:
            s.dirichlet_pores_pressures[pore] = s.paramsPnm['pressure_in']
        for pore in s.outlet_pores:
            s.dirichlet_pores_pressures[pore] = s.paramsPnm['pressure_out']

        s.throats_volumes = np.array(list(dict(
            (throat, float(s.netgrid.throats_Ss[throat] * s.netgrid.throats_Ls[throat]))
            for throat in s.netgrid.throats_Ss).values()))

        s.pnm = Pnm(s.paramsPnm, s.netgrid)
        s.velocities = None

        #############
        # VOF
        #############
        s.sat_ini = float(get('Properties_vof', 'sat_ini'))
        s.sat_inlet = float(get('Properties_vof', 'sat_inlet'))
        s.sat_outlet = float(get('Properties_vof', 'sat_outlet'))

        # fill the first cell in the inlet throats
        s.sats_curr = np.tile(s.sat_ini, s.netgrid.cells_N)
        for i in s.netgrid.types_cells['inlet']:
            s.sats_curr[i] = s.sat_inlet
        # fully fill inlet throats
        # s.sats_curr = np.tile(s.sat_ini, s.netgrid.cells_N)
        # for throat in s.inlet_throats:
        #     for cell in s.netgrid.throats_cells[throat]:
        #         s.sats_curr[cell] = s.sat_inlet

        s.sats_prev = copy.deepcopy(s.sats_curr)
        s.sats_arrays = {"sats_curr": s.sats_curr,
                         "sats_prev": s.sats_prev}
        s.netgrid.cells_arrays = s.sats_arrays

        s.time_period = float(get('Properties_vof', 'time_period'))  # sec
        s.const_time_step = float(get('Properties_vof', 'const_time_step'))  # sec
        s.time_step_type = str(get('Properties_vof', 'time_step_type'))  # sec
        s.tsm = float(get('Properties_vof', 'tsm'))
        s.sat_trim = float(get('Properties_vof', 'sat_trim'))
        s.params = {'time_period': s.time_period, 'const_time_step': s.const_time_step,
                    'tsm': s.tsm, 'sat_trim': s.sat_trim}

        s.contact_angle = float(get('Properties_vof', 'contact_angle'))
        s.ift = float(get('Properties_vof', 'interfacial_tension'))

        s.props = Props(s.params)
        s.local = Local(s.props, s.netgrid)
        s.convective = Convective(s.props, s.netgrid)
        s.equation = Equation(s.props, s.netgrid, s.local, s.convective)

        s.equation.bound_groups_dirich = ['inlet']
        s.equation.sats_bound_dirich = {'inlet': s.sat_inlet}
        s.equation.bound_groups_newman = ['outlet']
        s.equation.sats = [s.sats_curr, s.sats_prev]
        s.sats_init = copy.deepcopy(s.equation.sats[s.equation.i_curr])
        s.sats_time = [s.sats_init]

        s.av_density = None
        s.av_viscosity = None
        s.av_sats = None

    def run_pnm(s):
        s.pnm.cfd_procedure(s.throats_denss, s.throats_viscs, s.capillary_pressures,
                            s.newman_pores_flows, s.dirichlet_pores_pressures)
        s.pnm.calc_throats_flow_rates(s.capillary_pressures)

        mass_flows = s.pnm.throats_mass_flows
        cross_secs = s.netgrid.throats_Ss
        s.velocities = dict((throat, float(mass_flows[throat]) / cross_secs[throat])
                            for throat in mass_flows)

    def calc_coupling_params(s):
        s.equation.calc_throats_av_sats()
        s.equation.calc_throats_sats_grads()

        s.av_sats = s.equation.throats_av_sats
        s.throats_denss = s.av_sats * s.paramsPnm['liq_dens'] + \
                          (1 - s.av_sats) * s.paramsPnm['b_gas_dens']
        s.throats_viscs = s.av_sats * s.paramsPnm['liq_visc'] + \
                          (1 - s.av_sats) * s.paramsPnm['gas_visc']

        coeff = 2. * abs(math.cos(s.contact_angle)) * s.ift
        # coeff = -0.01 * coeff
        # coeff = -1. * coeff
        coeff = 0
        # ToDo: make more readable

        s.capillary_pressures = np.array(list(dict((throat, (coeff / s.throats_widths[throat]) +
                                                    (coeff / s.throats_depths[throat]))
                                                   for throat in s.throats_widths).values()))

        capillary_coeffs = copy.deepcopy(s.equation.throats_sats_grads ** 3)
        # threshold = 0.1
        # capillary_coeffs = np.where(capillary_coeffs > threshold, 1, capillary_coeffs)
        # capillary_coeffs = np.where(capillary_coeffs <= threshold, 0, capillary_coeffs)
        # capillary_coeffs = np.where(capillary_coeffs < -threshold, -1, capillary_coeffs)
        s.capillary_pressures = np.multiply(capillary_coeffs, s.capillary_pressures)

    def throats_values_to_cells(s, array):
        cells_values = np.full(s.netgrid.cells_N, 0, dtype=np.float64)
        s.netgrid.throats_values_to_cells(array, cells_values)
        return cells_values

    def pores_values_to_cells(s, array):
        cells_values = np.full(s.netgrid.cells_N, 0, dtype=np.float64)
        s.netgrid.pores_values_to_cells(array, cells_values)
        return cells_values

    def calc_rel_perms(s, rel_perms_water_array, rel_perms_gas_array, capillary_number_array,
                       water_sats_array, gas_flow_in, gas_flow_out, flow_in, flow_out):

        water_av_sat = \
            np.sum(s.throats_volumes * s.equation.throats_av_sats) / np.sum(s.throats_volumes)

        capillary_numbers = s.throats_viscs * np.absolute(
            np.array(list(s.velocities.values()))) / s.ift
        capillary_number = np.sum(s.throats_volumes * capillary_numbers) / np.sum(s.throats_volumes)
        capillary_number_array.append(capillary_number)

        relat_flow_rate = (flow_in + flow_out) / (gas_flow_in + gas_flow_out)
        # 5.668081102853038e-09

        rel_perm_water = water_av_sat * s.paramsPnm['liq_visc'] * relat_flow_rate / s.paramsPnm[
            'gas_visc']

        rel_perm_gas = (1. - water_av_sat) * relat_flow_rate

        # rel_perm_water = water_av_sat * visc_av * relat_velocity / \
        #                  s.paramsPnm['gas_visc']
        # rel_perm_gas = (1. - water_av_sat) * visc_av * relat_velocity / s.paramsPnm[
        #     'gas_visc']

        # print('relat_velocity: ', relat_velocity)

        rel_perms_water_array.append(rel_perm_water)
        rel_perms_gas_array.append(rel_perm_gas)
        water_sats_array.append(water_av_sat)


if __name__ == '__main__':
    cfd = Cfd(config_file=sys.argv[1])

    cfd.run_pnm()
    gas_flow_in = 0
    for throat in cfd.inlet_throats:
        velocity = cfd.velocities[throat]
        S = cfd.netgrid.throats_Ss[throat]
        gas_flow_in += velocity * S

    gas_flow_out = 0
    for throat in cfd.outlet_throats:
        velocity = cfd.velocities[throat]
        S = cfd.netgrid.throats_Ss[throat]
        gas_flow_out += velocity * S

    cfd.calc_coupling_params()
    cfd.run_pnm()

    rel_perms_water_array = []
    rel_perms_gas_array = []
    capillary_number_array = []
    water_av_sats_array = []

    volume_already_in = copy.deepcopy(np.sum(cfd.throats_volumes * cfd.equation.throats_av_sats *
                                             cfd.paramsPnm['liq_dens']))
    flow_in_array = []
    flow_out_array = []
    volume_inside_array = []

    av_sats_array = cfd.throats_values_to_cells(cfd.equation.throats_av_sats)
    sats_grads_array = cfd.throats_values_to_cells(cfd.equation.throats_sats_grads)
    capillary_pressures_array = cfd.throats_values_to_cells(cfd.capillary_pressures)
    pressures_array = cfd.pores_values_to_cells(cfd.pnm.pressures)

    throats_idxs = np.arange(cfd.netgrid.throats_N, dtype=float)
    throats_idxs_array = cfd.throats_values_to_cells(throats_idxs)
    velocities = np.array(list(cfd.velocities.values()))
    velocities_array = cfd.throats_values_to_cells(velocities)
    densities_array = cfd.throats_values_to_cells(cfd.throats_denss)
    conductances = np.array(list(cfd.pnm.conductances.values()))
    conductances_array = cfd.throats_values_to_cells(conductances)

    delta_pressures = []
    for pores in cfd.netgrid.throats_pores.values():
        delta_pressures.append(cfd.pnm.pressures[pores[0]] - cfd.pnm.pressures[pores[1]])
    delta_pressures = np.array(delta_pressures)
    delta_pressures_array = cfd.throats_values_to_cells(delta_pressures)

    #################
    # Paraview output
    #################
    os.system('rm -r inOut/*.vtu')
    os.system('rm -r inOut/*.pvd')
    sats_dict = dict()
    file_name = 'inOut/collection_refined.pvd'
    files_names = list()
    files_descriptions = list()
    #################
    #################

    time = [0]
    time_steps = []
    cour_number = np.empty([])
    time_curr = 0
    time_output_freq = cfd.time_period / 100.
    time_bound = time_output_freq
    is_output_step = False
    is_last_step = False
    i = int(0)
    while True:

        if cfd.time_step_type == 'const':
            time_step = cfd.const_time_step
        elif cfd.time_step_type == 'flow_variable':
            time_step = cfd.local.calc_flow_variable_time_step(cfd.velocities)
        elif cfd.time_step_type == 'div_variable':
            time_step = cfd.local.calc_div_variable_time_step(
                cfd.equation.sats[cfd.equation.i_curr], cfd.velocities)

        if time_curr + time_step >= time_bound:
            time_step = time_bound - time_curr
            time_bound += time_output_freq
            is_output_step = True

        if time_curr + time_step >= cfd.time_period:
            time_step = cfd.time_period - time_curr
            is_last_step = True

        time_steps.append(time_step)
        time_curr += time_step

        cfd.equation.cfd_procedure_one_step(cfd.velocities, time_step)
        cfd.calc_coupling_params()

        volume_inside = copy.deepcopy(np.sum(cfd.throats_volumes * cfd.equation.throats_av_sats *
                                             cfd.paramsPnm['liq_dens']))
        volume_inside_array.append(volume_inside)

        flow_in = 0
        flow_in_rel_perm = 0
        for throat in cfd.inlet_throats:
            first_cell = cfd.netgrid.throats_cells[throat][0]
            velocity = cfd.velocities[throat]
            S = cfd.netgrid.throats_Ss[throat]
            density = cfd.paramsPnm['liq_dens']
            sat = cfd.equation.sats[cfd.equation.i_curr][first_cell]
            flow_in += velocity * S * sat * density
            flow_in_rel_perm += velocity * S * sat
        flow_in_array.append(flow_in)

        flow_out = 0
        flow_out_rel_perm = 0
        for throat in cfd.outlet_throats:
            last_cell = cfd.netgrid.throats_cells[throat][-1]
            velocity = cfd.velocities[throat]
            S = cfd.netgrid.throats_Ss[throat]
            density = cfd.paramsPnm['liq_dens']
            sat = cfd.equation.sats[cfd.equation.i_curr][last_cell]
            flow_out += velocity * S * sat * density
            flow_out_rel_perm += velocity * S * sat
        flow_out_array.append(flow_out)

        cfd.calc_rel_perms(rel_perms_water_array, rel_perms_gas_array, capillary_number_array,
                           water_av_sats_array, gas_flow_in, gas_flow_out, flow_in_rel_perm,
                           flow_out_rel_perm)

        av_sats_array = cfd.throats_values_to_cells(cfd.equation.throats_av_sats)
        sats_grads_array = cfd.throats_values_to_cells(cfd.equation.throats_sats_grads)
        capillary_pressures_array = cfd.throats_values_to_cells(cfd.capillary_pressures)
        velocities = np.array(list(cfd.velocities.values()))
        velocities_array = cfd.throats_values_to_cells(velocities)
        densities_array = cfd.throats_values_to_cells(cfd.throats_denss)
        conductances = np.array(list(cfd.pnm.conductances.values()))
        conductances_array = cfd.throats_values_to_cells(conductances)

        delta_pressures = []
        for pores in cfd.netgrid.throats_pores.values():
            delta_pressures.append(cfd.pnm.pressures[pores[0]] - cfd.pnm.pressures[pores[1]])
        delta_pressures = np.array(delta_pressures)
        delta_pressures_array = cfd.throats_values_to_cells(delta_pressures)

        print('time_step: ', int(time_curr / time_step))
        time.append(time_curr)
        cfd.equation.print_cour_numbers(cfd.velocities, time_step)
        print(' time:', round((time_curr / cfd.time_period * 100.), 2), '%.', '\n')
        cfd.run_pnm()
        pressures_array = cfd.pores_values_to_cells(cfd.pnm.pressures)
        sats_array = copy.deepcopy(cfd.equation.sats[cfd.equation.i_curr])
        output_1 = np.array(cfd.local.output_1, dtype=float)
        output_2 = np.array(cfd.local.output_2, dtype=float)
        if is_output_step:
            cfd.netgrid.cells_arrays = {
                'sat': sats_array,
                'sat_av': av_sats_array,
                'sat_grad': sats_grads_array,
                'output_1': output_1,
                'output_2': output_2,
                'capillary_Ps': capillary_pressures_array,
                'pressures': pressures_array,
                'velocities': velocities_array,
                'conductances': conductances_array,
                'delta_P': delta_pressures_array,
                'throats_idxs': throats_idxs_array}
            files_names.append(str(i) + '_refined.vtu')
            files_descriptions.append(str(i))
            cfd.netgrid.save_cells('inOut/' + files_names[-1])
            save_files_collection_to_file(file_name, files_names, files_descriptions)
            i += 1
            is_output_step = False

        if is_last_step:
            break
        #
    fig, ax1 = plt.subplots()
    ax1.plot(water_av_sats_array, rel_perms_water_array, ls="", marker="o", markersize=2,
             color="b", label='water')
    ax2 = ax1.twinx()
    ax1.plot(water_av_sats_array, rel_perms_gas_array, ls="", marker="o", markersize=2,
             color="y", label='gas')
    ax2.plot(water_av_sats_array, capillary_number_array, ls="", marker="o", markersize=2,
             label='Ca')
    ax1.set_xlabel('Sw')
    ax1.set_ylabel('Krw')
    ax2.set_ylabel('Ca')
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.show()

    flow_in_accum = np.cumsum(np.array(flow_in_array) * np.array(time_steps))
    flow_out_accum = np.cumsum(np.array(flow_out_array) * np.array(time_steps))
    volume_inside_accum = np.array(volume_inside_array) - volume_already_in
    time_accum = np.cumsum(time_steps)

    fig, ax2 = plt.subplots()

    ax2.plot(time_accum, flow_in_accum - flow_out_accum, ls="", marker="o", markersize=2,
             color="b", label='massrate_net')
    # ax2.plot(time_accum, flow_in_accum, ls="", marker="o", markersize=2, label='flowrate_in')
    # ax2.plot(time_accum, flow_out_accum, ls="", marker="o", markersize=2, label='flowrate_out')
    # ax2 = ax1.twinx()
    ax2.set_xlabel('time')
    ax2.set_ylabel('volume')
    ax2.plot(time_accum, volume_inside_accum, ls="", marker="o", markersize=2,
             color="y", label='mass_inside')
    plt.legend()
    plt.show()

    # # #################
    # # # Paraview output
    # # #################
    # os.system('rm -r inOut/*.vtu')
    # os.system('rm -r inOut/*.pvd')
    # sats_dict = dict()
    # file_name = 'inOut/collection_refined.pvd'
    # files_names = list()
    # files_descriptions = list()
    #
    # freq = 1000
    # for i in range(len(time)):
    #     if (i % freq) == 0:
    #         cfd.netgrid.cells_arrays = {'sat': cfd.sats_time[i],
    #                                     'sat_av': av_sats_array[i],
    #                                     'sat_grad': sats_grads_array[i],
    #                                     'capillary_Ps': capillary_pressures_array[i],
    #                                     'pressures': pressures_array[i],
    #                                     'velocities': velocities_array[i],
    #                                     'conductances': conductances_array[i],
    #                                     'delta_P': delta_pressures_array[i]}
    #         files_names.append(str(i) + '_refined.vtu')
    #         files_descriptions.append(str(i))
    #         cfd.netgrid.save_cells('inOut/' + files_names[-1])
    #         save_files_collection_to_file(file_name, files_names, files_descriptions)
    #
    # # for pores in cfd.netgrid.throats_pores.values():
    # # print('press_grad:', cfd.pnm.pressures[pores[0]] - cfd.pnm.pressures[pores[1]])
    # # print('pores: ', pores[0], pores[1])

    # np.set_printoptions(threshold=sys.maxsize)
    # print(cfd.equation.matrix.toarray()[cfd.netgrid.throats_cells[4][-1]])

    # zero matrix sum
    # np.set_printoptions(threshold=sys.maxsize)
    # zero_array = np.sum(cfd.equation.matrix.toarray(), axis=1) - np.array(cfd.local.alphas)
    # print('zero_array: ', zero_array)
    # print('non_zero_rows: ', np.nonzero(zero_array))
