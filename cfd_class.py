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

        # Test model Jing
        s.pores_coordinates = literal_eval(s.__config['Properties_grid']['pores_coordinates'])
        s.throats_pores = literal_eval(s.__config['Properties_grid']['throats_pores'])
        s.throats_widths = literal_eval(s.__config['Properties_grid']['throats_widths'])
        s.throats_depths = literal_eval(s.__config['Properties_grid']['throats_depths'])
        s.inlet_pores = literal_eval(s.__config['Properties_grid']['inlet_pores'])
        s.outlet_pores = literal_eval(s.__config['Properties_grid']['outlet_pores'])
        s.delta_L = float(get('Properties_grid', 'delta_L'))
        s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))

        s.inlet_throats = np.array([0, 1])
        s.outlet_throats = np.array([3, 4])

        # s.inlet_throats = np.array([0, 1, 2, 3, 4, 5, 6])
        # s.outlet_throats = np.array([13, 14, 15, 16, 17, 18])

        # Test model quadratic
        # s.inlet_throats = np.array([41])
        # s.outlet_throats = np.array([40])

        s.netgrid = Netgrid(s.pores_coordinates, s.throats_pores,
                            s.throats_widths, s.throats_depths, s.delta_L, s.min_cells_N,
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

        s.pore_n = len(s.pores_coordinates)
        s.throats_denss = np.tile(s.paramsPnm['b_gas_dens'], len(s.netgrid.throats_Ss))
        s.throats_viscs = np.tile(s.paramsPnm['gas_visc'], len(s.netgrid.throats_Ss))
        s.capillary_pressures = np.tile(0., len(s.netgrid.throats_Ss))
        s.newman_pores_flows = {}
        s.dirichlet_pores_pressures = {}

        for pore in s.inlet_pores:
            s.dirichlet_pores_pressures[pore] = s.paramsPnm['pressure_in']
        for pore in s.outlet_pores:
            s.dirichlet_pores_pressures[pore] = s.paramsPnm['pressure_out']

        s.throats_volumes = np.array(list(dict(
            (k, float(s.netgrid.throats_Ss[k] * s.netgrid.throats_Ls[k]))
            for k in s.netgrid.throats_Ss).values()))

        s.pnm = Pnm(s.paramsPnm, s.netgrid)
        s.velocities = None
        #############
        # VOF
        #############
        s.sat_ini = float(get('Properties_vof', 'sat_ini'))
        s.sat_inlet = float(get('Properties_vof', 'sat_inlet'))
        s.sat_outlet = float(get('Properties_vof', 'sat_outlet'))

        s.sats_curr = np.tile(s.sat_ini, s.netgrid.cells_N)
        for i in s.netgrid.types_cells['inlet']:
            s.sats_curr[i] = s.sat_inlet
        # s.sats_curr = np.tile(s.sat_ini, s.netgrid.cells_N)
        # for throat in s.inlet_throats:
        #     for cell in s.netgrid.throats_cells[throat]:
        #         s.sats_curr[cell] = s.sat_inlet

        s.sats_prev = copy.deepcopy(s.sats_curr)
        s.sats_arrays = {"sats_curr": s.sats_curr,
                         "sats_prev": s.sats_prev}
        s.netgrid.cells_arrays = s.sats_arrays

        s.time_period = float(get('Properties_vof', 'time_period'))  # sec
        s.time_step = float(get('Properties_vof', 'time_step'))  # sec
        s.params = {'time_period': s.time_period, 'time_step': s.time_step}

        s.contact_angle = float(get('Properties_vof', 'contact_angle'))
        s.ift = float(get('Properties_vof', 'interfacial_tension'))

        s.props = Props(s.params)
        s.local = Local(s.props, s.netgrid)
        s.local.calc_time_steps()
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

        mass_flows = s.pnm.throats_flow_rates
        cross_secs = s.netgrid.throats_Ss
        s.velocities = dict((k, float(mass_flows[k]) / cross_secs[k])
                            for k in mass_flows)
        # s.velocities = dict((k, float(mass_flows[k]) / cross_secs[k] / s.throats_denss[k])
        #                     for k in mass_flows)

    def calc_coupling_params(s):
        s.equation.calc_throats_av_sats()
        s.equation.calc_throats_sats_grads()

        # print('sats:', s.equation.sats[s.equation.i_curr])
        s.av_sats = s.equation.throats_av_sats
        s.throats_denss = s.av_sats * s.paramsPnm['liq_dens'] + \
                          (1 - s.av_sats) * s.paramsPnm['b_gas_dens']
        s.throats_viscs = s.av_sats * s.paramsPnm['liq_visc'] + \
                          (1 - s.av_sats) * s.paramsPnm['gas_visc']

        coeff = 2. * abs(math.cos(s.contact_angle)) * s.ift
        coeff = 0 * coeff
        # coeff = -1. * coeff
        # ToDo: make more readable
        s.capillary_pressures = dict((k, (coeff / s.throats_widths[k]) +
                                      (coeff / s.throats_depths[k]))
                                     for k in s.throats_widths)

        s.capillary_pressures = list(s.capillary_pressures.values())
        s.capillary_pressures = np.array(s.capillary_pressures)
        # ###########################
        capillary_coeffs = copy.deepcopy(s.equation.throats_sats_grads ** 3)
        # threshold = 0.1
        # capillary_coeffs = np.where(capillary_coeffs > threshold, 1, capillary_coeffs)
        # capillary_coeffs = np.where(capillary_coeffs <= threshold, 0, capillary_coeffs)
        # capillary_coeffs = np.where(capillary_coeffs < -threshold, -1, capillary_coeffs)
        s.capillary_pressures = np.multiply(capillary_coeffs, s.capillary_pressures)

    def throats_values_to_cells(s, array_arrays, array):
        cells_values = np.full(s.netgrid.cells_N, 0, dtype=np.float64)
        s.netgrid.throats_values_to_cells(array, cells_values)
        array_arrays.append(copy.deepcopy(cells_values))

    def pores_values_to_cells(s, array_arrays, array):
        cells_values = np.full(s.netgrid.cells_N, 0, dtype=np.float64)
        s.netgrid.pores_values_to_cells(array, cells_values)
        array_arrays.append(copy.deepcopy(cells_values))

    def calc_rel_perms(s, rel_perms_water_array, rel_perms_gas_array, capillary_number_array,
                       water_sats_array, gas_flow_in, gas_flow_out, flow_in, flow_out):

        water_av_sat = \
            np.sum(s.throats_volumes * s.equation.throats_av_sats) / np.sum(s.throats_volumes)

        capillary_numbers = s.throats_viscs * np.absolute(
            np.array(list(s.velocities.values()))) / s.ift
        capillary_number = np.sum(s.throats_volumes * capillary_numbers) / np.sum(s.throats_volumes)
        capillary_number_array.append(capillary_number)

        relat_flow_rate = (flow_in + flow_out) / (gas_flow_in + gas_flow_out)
        # print('flow_in + flow_out', flow_in + flow_out)
        # print('gas_flow_in + gas_flow_out', gas_flow_in + gas_flow_out)

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

    # cfd.newman_pores_flows = {6: 50., 15: 50.}
    # cfd.dirichlet_pores_pressures = {19: cfd.paramsPnm['pressure_out'],
    #                                  22: cfd.paramsPnm['pressure_out'],
    #                                  23: cfd.paramsPnm['pressure_out']}
    cfd.calc_coupling_params()
    cfd.run_pnm()
    pressures_array = []
    av_sats_array = []
    sats_grads_array = []
    capillary_pressures_array = []
    velocities_array = []
    densities_array = []
    conductances_array = []
    delta_pressures_array = []

    rel_perms_water_array = []
    rel_perms_gas_array = []
    capillary_number_array = []
    water_av_sats_array = []

    volume_already_in = copy.deepcopy(np.sum(cfd.throats_volumes * cfd.equation.throats_av_sats *
                                             cfd.paramsPnm['liq_dens']))
    flow_in_array = []
    flow_out_array = []
    volume_inside_array = []

    cfd.throats_values_to_cells(av_sats_array, cfd.equation.throats_av_sats)
    cfd.throats_values_to_cells(sats_grads_array, cfd.equation.throats_sats_grads)
    cfd.throats_values_to_cells(capillary_pressures_array, cfd.capillary_pressures)
    cfd.pores_values_to_cells(pressures_array, cfd.pnm.pressures)

    velocities = np.array(list(cfd.velocities.values()))
    cfd.throats_values_to_cells(velocities_array, velocities)
    cfd.throats_values_to_cells(densities_array, cfd.throats_denss)
    conductances = np.array(list(cfd.pnm.conductances.values()))
    cfd.throats_values_to_cells(conductances_array, conductances)

    delta_pressures = []
    for pores in cfd.netgrid.throats_pores.values():
        delta_pressures.append(cfd.pnm.pressures[pores[0]] - cfd.pnm.pressures[pores[1]])
        # delta_pressures = np.array(delta_pressures)
    delta_pressures = np.array(delta_pressures)
    cfd.throats_values_to_cells(delta_pressures_array, delta_pressures)

    time = [0]
    cour_number = np.empty([])
    time_curr = 0

    for cfd.time_step in cfd.local.time_steps:
        time_curr += cfd.time_step
        cfd.equation.cfd_procedure_one_step(cfd.velocities, cfd.time_step)

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
            flow_in_rel_perm += velocity * S
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
            flow_out_rel_perm += velocity * S
        flow_out_array.append(flow_out)

        cfd.calc_rel_perms(rel_perms_water_array, rel_perms_gas_array, capillary_number_array,
                           water_av_sats_array, gas_flow_in, gas_flow_out, flow_in_rel_perm,
                           flow_out_rel_perm)

        cfd.throats_values_to_cells(av_sats_array, cfd.equation.throats_av_sats)
        cfd.throats_values_to_cells(sats_grads_array, cfd.equation.throats_sats_grads)
        cfd.throats_values_to_cells(capillary_pressures_array, cfd.capillary_pressures)
        velocities = np.array(list(cfd.velocities.values()))
        cfd.throats_values_to_cells(velocities_array, velocities)
        cfd.throats_values_to_cells(densities_array, cfd.throats_denss)
        conductances = np.array(list(cfd.pnm.conductances.values()))
        cfd.throats_values_to_cells(conductances_array, conductances)
        cfd.sats_time.append(copy.deepcopy(cfd.equation.sats[cfd.equation.i_curr]))

        delta_pressures = []
        for pores in cfd.netgrid.throats_pores.values():
            delta_pressures.append(cfd.pnm.pressures[pores[0]] - cfd.pnm.pressures[pores[1]])
        delta_pressures = np.array(delta_pressures)
        cfd.throats_values_to_cells(delta_pressures_array, delta_pressures)

        print('time_step: ', int(time_curr / cfd.time_step))
        time.append(time_curr)
        cfd.equation.print_cour_numbers(cfd.velocities, cfd.time_step)
        print(' time:', round((time_curr / cfd.time_period * 1000 * 0.1), 2), '%.', '\n')
        cfd.run_pnm()
        cfd.pores_values_to_cells(pressures_array, cfd.pnm.pressures)

        # np.set_printoptions(threshold=sys.maxsize)
        # print(cfd.equation.matrix.toarray()[cfd.netgrid.throats_cells[4][-1]])

        # zero matrix sum
        # np.set_printoptions(threshold=sys.maxsize)
        # zero_array = np.sum(cfd.equation.matrix.toarray(), axis=1) - np.array(cfd.local.alphas)
        # print('zero_array: ', zero_array)
        # print('non_zero_rows: ', np.nonzero(zero_array))
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(water_av_sats_array, rel_perms_water_array, ls="", marker="o", markersize=2,
    #          color="b", label='water')
    # ax2 = ax1.twinx()
    # ax1.plot(water_av_sats_array, rel_perms_gas_array, ls="", marker="o", markersize=2,
    #          color="y", label='gas')
    # ax2.plot(water_av_sats_array, capillary_number_array, ls="", marker="o", markersize=2,
    #          label='Ca')
    # ax1.set_xlabel('Sw')
    # ax1.set_ylabel('Krw')
    # ax2.set_ylabel('Ca')
    # ax1.legend(loc=2)
    # ax2.legend(loc=1)
    # plt.show()
    #
    # flow_in_accum = np.cumsum(np.array(flow_in_array) * cfd.time_step)
    # flow_out_accum = np.cumsum(np.array(flow_out_array) * cfd.time_step)
    # volume_inside_accum = np.array(volume_inside_array) - volume_already_in
    # time_accum = np.cumsum(cfd.local.time_steps)
    #
    # fig, ax2 = plt.subplots()
    #
    # ax2.plot(time_accum, flow_in_accum - flow_out_accum, ls="", marker="o", markersize=2,
    #          color="b", label='massrate_net')
    # # ax2.plot(time_accum, flow_in_accum, ls="", marker="o", markersize=2, label='flowrate_in')
    # # ax2.plot(time_accum, flow_out_accum, ls="", marker="o", markersize=2, label='flowrate_out')
    # # ax2 = ax1.twinx()
    # ax2.set_xlabel('time')
    # ax2.set_ylabel('volume')
    # ax2.plot(time_accum, volume_inside_accum, ls="", marker="o", markersize=2,
    #          color="y", label='mass_inside')
    # plt.legend()
    # plt.show()
    #
    # #################
    # # Paraview output
    # #################
    os.system('rm -r inOut/*.vtu')
    os.system('rm -r inOut/*.pvd')
    sats_dict = dict()
    file_name = 'inOut/collection_refined.pvd'
    files_names = list()
    files_descriptions = list()

    freq = 1
    for i in range(len(time)):
        if (i % freq) == 0:
            cfd.netgrid.cells_arrays = {'sat': cfd.sats_time[i],
                                        'sat_av': av_sats_array[i],
                                        'sat_grad': sats_grads_array[i],
                                        'capillary_Ps': capillary_pressures_array[i],
                                        'pressures': pressures_array[i],
                                        'velocities': velocities_array[i],
                                        'conductances': conductances_array[i],
                                        'delta_P': delta_pressures_array[i]}
            files_names.append(str(i) + '_refined.vtu')
            files_descriptions.append(str(i))
            cfd.netgrid.save_cells('inOut/' + files_names[-1])
    save_files_collection_to_file(file_name, files_names, files_descriptions)

    # for pores in cfd.netgrid.throats_pores.values():
    # print('press_grad:', cfd.pnm.pressures[pores[0]] - cfd.pnm.pressures[pores[1]])
    # print('pores: ', pores[0], pores[1])

