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
        # Test model \/ | /\
        # s.pores_coordinates = {0: [1., 2.], 1: [1., -2.], 2: [5., 0.], 3: [7., 0.],
        #                        4: [9., 2.], 5: [9., -2.]}
        # s.throats_pores = {0: [0, 2], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [3, 5]}
        # s.throats_widths = {0: 0.1, 1: 0.1, 2: 0.25, 3: 0.15, 4: 0.15}
        # s.throats_depths = {0: 0.35, 1: 0.35, 2: 0.6, 3: 0.25, 4: 0.25}
        # s.delta_L = float(get('Properties_grid', 'delta_L'))
        # s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))
        #
        # s.inlet_pores = {0, 1}
        # s.outlet_pores = {4, 5}
        # Test model simple _ | | |
        # s.pores_coordinates = {0: [0., 0.], 1: [0., 4.], 2: [2., 4.], 3: [-2., 4.],
        #                        4: [0., 8.], 5: [2., 8.], 6: [-2., 8.]}
        # s.throats_pores = {0: [0, 1], 1: [1, 2], 2: [1, 3], 3: [1, 4], 4: [2, 5], 5: [3, 6]}
        # s.throats_widths = {0: 0.1, 1: 0.05, 2: 0.4, 3: 0.1, 4: 0.05, 5: 0.4}
        # s.throats_depths = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1}
        # s.delta_L = float(get('Properties_grid', 'delta_L'))
        # s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))
        #
        # s.inlet_pores = {0}
        # s.outlet_pores = {4, 5, 6}
        # s.inlet_throats = np.array([0])
        # s.outlet_throats = np.array([3, 4, 5])

        # Test model complex
        # s.pores_coordinates = {0: [28., 99.9], 1: [35., 69.], 2: [50., 58.], 3: [32., 53.],
        #                        4: [61., 57.], 5: [73., 60.], 6: [26., 100.], 7: [76., 55.],
        #                        8: [98., 56.], 9: [94., 74.], 10: [72., 72.], 11: [68., 84.],
        #                        12: [87., 84.], 13: [80., 96.], 14: [57., 100.9], 15: [56., 101.],
        #                        16: [74., 23.], 17: [83., 24.], 18: [38., 22.], 19: [76., 2.],
        #                        20: [38., 14.], 21: [23., 11.], 22: [25., 2], 23: [41., 1.]}
        #
        # s.throats_pores = {0: [0, 6], 1: [0, 1], 2: [1, 2], 3: [2, 4], 4: [4, 5], 5: [5, 10],
        #                    6: [10, 11], 7: [11, 13], 8: [10, 12], 9: [7, 8], 10: [8, 9],
        #                    11: [9, 12], 12: [12, 13], 13: [13, 14], 14: [14, 15], 15: [5, 7],
        #                    16: [4, 18], 17: [7, 17], 18: [16, 17], 19: [16, 19], 20: [20, 23],
        #                    21: [18, 20], 22: [3, 18], 23: [1, 3], 24: [20, 21], 25: [21, 22]}
        #
        # s.throats_widths = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0,
        #                     9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0,
        #                     17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0,
        #                     25: 1.0}
        #
        # s.throats_depths = {0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5, 7: 1.5,
        #                     8: 1.5, 9: 1.5, 10: 1.5, 11: 1.5, 12: 1.5, 13: 1.5, 14: 1.5,
        #                     15: 1.5, 16: 1.5, 17: 1.5, 18: 1.5, 19: 1.5, 20: 1.5, 21: 1.5,
        #                     22: 1.5, 23: 1.5, 24: 1.5, 25: 1.5}
        # s.delta_L = float(get('Properties_grid', 'delta_L'))
        # s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))
        #
        # s.inlet_pores = {6, 15}
        # s.outlet_pores = {19, 22, 23}
        # s.inlet_throats = {0, 14}

        # Test model quadratic

        s.pores_coordinates = {0: [0.5, 0.5], 1: [0.5, 1.5], 2: [0.5, 2.5], 3: [0.5, 3.5],
                               4: [0.5, 4.5], 5: [1.5, 0.5], 6: [1.5, 1.5], 7: [1.5, 2.5],
                               8: [1.5, 3.5], 9: [1.5, 4.5], 10: [2.5, 0.5], 11: [2.5, 1.5],
                               12: [2.5, 2.5], 13: [2.5, 3.5], 14: [2.5, 4.5], 15: [3.5, 0.5],
                               16: [3.5, 1.5], 17: [3.5, 2.5], 18: [3.5, 3.5], 19: [3.5, 4.5],
                               20: [4.5, 0.5], 21: [4.5, 1.5], 22: [4.5, 2.5], 23: [4.5, 3.5],
                               24: [4.5, 4.5], 25: [2.5, 0.1], 26: [2.5, 4.9]}

        s.throats_pores = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [5, 6], 5: [6, 7],
                           6: [7, 8], 7: [8, 9], 8: [10, 11], 9: [11, 12], 10: [12, 13],
                           11: [13, 14], 12: [15, 16], 13: [16, 17], 14: [17, 18], 15: [18, 19],
                           16: [20, 21], 17: [21, 22], 18: [22, 23], 19: [23, 24], 20: [0, 5],
                           21: [1, 6], 22: [2, 7], 23: [3, 8], 24: [4, 9], 25: [5, 10], 26: [6, 11],
                           27: [7, 12], 28: [8, 13], 29: [9, 14], 30: [10, 15], 31: [11, 16],
                           32: [12, 17], 33: [13, 18], 34: [14, 19], 35: [15, 20], 36: [16, 21],
                           37: [17, 22], 38: [18, 23], 39: [19, 24], 40: [10, 25], 41: [14, 26]}
        #
        s.throats_widths = {0: 0.105808943, 1: 0.100512363, 2: 0.140428113, 3: 0.185475502,
                            4: 0.160130653, 5: 0.191653093, 6: 0.259751009, 7: 0.299529478,
                            8: 0.13980895, 9: 0.118524175, 10: 0.146604534, 11: 0.242519626,
                            12: 0.146029201, 13: 0.147892087, 14: 0.112648064, 15: 0.204085648,
                            16: 0.224921242, 17: 0.183046901, 18: 0.174592434, 19: 0.159867198,
                            20: 0.230520478, 21: 0.223053132, 22: 0.126927328, 23: 0.19305168,
                            24: 0.296181542, 25: 0.235364853, 26: 0.241628436, 27: 0.182355375,
                            28: 0.139820093, 29: 0.111772433, 30: 0.119250727, 31: 0.189043356,
                            32: 0.237897479, 33: 0.103975472, 34: 0.259054376, 35: 0.194276711,
                            36: 0.286181091, 37: 0.22621814, 38: 0.161075372, 39: 0.261724302,
                            40: 0.1, 41: 0.1}

        s.throats_depths = {0: 0.106433886, 1: 0.137385778, 2: 0.140078374, 3: 0.195926926,
                            4: 0.113420209, 5: 0.142340574, 6: 0.144960077, 7: 0.189136225,
                            8: 0.15561773, 9: 0.108630577, 10: 0.143400167, 11: 0.196271333,
                            12: 0.127040187, 13: 0.153840534, 14: 0.142376393, 15: 0.197873038,
                            16: 0.133517971, 17: 0.137300739, 18: 0.137685504, 19: 0.17740006,
                            20: 0.132121331, 21: 0.14770719, 22: 0.117286486, 23: 0.139837154,
                            24: 0.188459309, 25: 0.12284481, 26: 0.132030567, 27: 0.168407173,
                            28: 0.14818333, 29: 0.158711059, 30: 0.152308724, 31: 0.168013797,
                            32: 0.191339743, 33: 0.105037598, 34: 0.189773116, 35: 0.194748039,
                            36: 0.185400963, 37: 0.143754305, 38: 0.12213411, 39: 0.142929008,
                            40: 0.1, 41: 0.1}

        s.throats_widths = {key: 0.2 for key in s.throats_widths.keys()}
        s.throats_widths[40] = 0.1
        s.throats_widths[41] = 0.1
        s.throats_depths = {key: 0.2 for key in s.throats_depths.keys()}
        s.throats_depths[40] = 0.1
        s.throats_depths[41] = 0.1

        s.delta_L = float(get('Properties_grid', 'delta_L'))
        s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))

        s.inlet_pores = {26}
        s.outlet_pores = {25}
        s.inlet_throats = np.array([41])
        s.outlet_throats = np.array([40])

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
        s.velocities = dict((k, float(mass_flows[k]) / cross_secs[k] / s.throats_denss[k])
                            for k in mass_flows)
        # print('velocities: ', s.velocities)

    def calc_coupling_params(s):
        s.equation.calc_throats_av_sats()
        s.equation.calc_throats_sats_grads()

        # print('sats:', s.equation.sats[s.equation.i_curr])
        s.av_sats = s.equation.throats_av_sats
        s.av_density = s.av_sats * s.paramsPnm['liq_dens'] + \
                       (1 - s.av_sats) * s.paramsPnm['b_gas_dens']
        s.av_viscosity = s.av_sats * s.paramsPnm['liq_visc'] + \
                         (1 - s.av_sats) * s.paramsPnm['gas_visc']
        s.throats_denss = s.av_density
        s.throats_viscs = s.av_viscosity

        coeff = -2. * abs(math.cos(s.contact_angle)) * s.ift
        coeff = coeff
        # ToDo: make more readable
        s.capillary_pressures = dict((k, (coeff / s.throats_widths[k]) +
                                      (coeff / s.throats_depths[k]))
                                     for k in s.throats_widths)
        s.capillary_pressures = list(s.capillary_pressures.values())
        s.capillary_pressures = np.array(s.capillary_pressures)
        # ###########################
        capillary_coeffs = copy.deepcopy(s.equation.throats_sats_grads)
        # threshold = 0.01
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

    def calc_rel_perms(s, rel_perms_water_array, rel_perms_gas_array, water_sats_array,
                       velocities_gas):

        water_av_sat = \
            np.sum(s.throats_volumes * s.equation.throats_av_sats) / np.sum(s.throats_volumes)

        relat_velocity = np.mean(
            np.array(list(s.velocities.values())) / np.array(list(velocities_gas.values())))

        visc_av = water_av_sat * s.paramsPnm['liq_visc'] + (1. - water_av_sat) * s.paramsPnm[
            'gas_visc']

        rel_perm_water = water_av_sat * s.paramsPnm['liq_visc'] * relat_velocity / \
                         s.paramsPnm['gas_visc']
        rel_perm_gas = (1. - water_av_sat) * relat_velocity

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
    velocities_gas = copy.deepcopy(cfd.velocities)

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

    rel_perms_water_array = []
    rel_perms_gas_array = []
    water_av_sats_array = []

    volume_already_in = copy.deepcopy(np.sum(cfd.throats_volumes * cfd.equation.throats_av_sats))
    flow_in_array = []
    flow_out_array = []
    volume_inside_array = []

    cfd.throats_values_to_cells(av_sats_array, cfd.equation.throats_av_sats)
    cfd.throats_values_to_cells(sats_grads_array, cfd.equation.throats_sats_grads)
    cfd.throats_values_to_cells(capillary_pressures_array, cfd.capillary_pressures)
    cfd.pores_values_to_cells(pressures_array, cfd.pnm.pressures)

    velocities = np.array(list(cfd.velocities.values()))
    cfd.throats_values_to_cells(velocities_array, velocities)

    time = [0]
    cour_number = np.empty([])
    time_curr = 0

    for cfd.time_step in cfd.local.time_steps:
        time_curr += cfd.time_step
        cfd.equation.cfd_procedure_one_step(cfd.velocities, cfd.time_step)

        cfd.calc_coupling_params()

        volume_inside = copy.deepcopy(np.sum(cfd.throats_volumes * cfd.equation.throats_av_sats))
        volume_inside_array.append(volume_inside)

        flow_in = 0
        for throat in cfd.inlet_throats:
            flow_in += cfd.velocities[throat] * cfd.netgrid.throats_Ss[throat]
        flow_in_array.append(abs(flow_in))

        flow_out = 0
        for throat in cfd.outlet_throats:
            last_cell = cfd.netgrid.throats_cells[throat][-1]
            second_last_cell = cfd.netgrid.throats_cells[throat][-2]
            velocity = cfd.velocities[throat]
            S = cfd.netgrid.throats_Ss[throat]
            sat = cfd.equation.sats[cfd.equation.i_curr][last_cell]
            flow_out += velocity * S * sat
        flow_out_array.append(flow_out)

        # cfd.calc_rel_perms(rel_perms_water_array, rel_perms_gas_array, water_av_sats_array,
        #                    velocities_gas)
        cfd.throats_values_to_cells(av_sats_array, cfd.equation.throats_av_sats)
        cfd.throats_values_to_cells(sats_grads_array, cfd.equation.throats_sats_grads)
        cfd.throats_values_to_cells(capillary_pressures_array, cfd.capillary_pressures)
        velocities = np.array(list(cfd.velocities.values()))
        cfd.throats_values_to_cells(velocities_array, velocities)
        cfd.sats_time.append(copy.deepcopy(cfd.equation.sats[cfd.equation.i_curr]))
        print('time_step: ', int(time_curr / cfd.time_step))
        time.append(time_curr)
        print('test1\n')
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

    # fig, ax1 = plt.subplots()
    # ax1.plot(water_av_sats_array, rel_perms_water_array, ls="", marker="o", markersize=2,
    #          color="b", label='water')
    # # ax2 = ax1.twinx()
    # ax1.plot(water_av_sats_array, rel_perms_gas_array, ls="", marker="o", markersize=2,
    #          color="y", label='gas')
    # ax1.set_xlabel('Sw')
    # ax1.set_ylabel('Krw')
    # # ax2.set_ylabel('Krg')
    # plt.legend()
    # plt.show()

    flow_in_accum = np.cumsum(np.array(flow_in_array) * cfd.time_step)
    flow_out_accum = np.cumsum(np.array(flow_out_array) * cfd.time_step)
    volume_inside_accum = np.array(volume_inside_array) - volume_already_in
    time_accum = np.cumsum(cfd.local.time_steps)

    fig, ax2 = plt.subplots()

    ax2.plot(time_accum, flow_in_accum - flow_out_accum, ls="", marker="o", markersize=2,
             color="b", label='flowrate_net')
    # ax2.plot(time_accum, flow_in_accum, ls="", marker="o", markersize=2, label='flowrate_in')
    # ax2.plot(time_accum, flow_out_accum, ls="", marker="o", markersize=2, label='flowrate_out')
    # ax2 = ax1.twinx()
    ax2.set_xlabel('time')
    ax2.set_ylabel('volume')
    ax2.plot(time_accum, volume_inside_accum, ls="", marker="o", markersize=2,
             color="y", label='volume_in')
    plt.legend()
    plt.show()

    #################
    # Paraview output
    #################
    os.system('rm -r inOut/*.vtu')
    os.system('rm -r inOut/*.pvd')
    sats_dict = dict()
    file_name = 'inOut/collection_refined.pvd'
    files_names = list()
    files_descriptions = list()
    for i in range(len(time)):
        cfd.netgrid.cells_arrays = {'sat': cfd.sats_time[i],
                                    'sat_av': av_sats_array[i],
                                    'sat_grad': sats_grads_array[i],
                                    'capillary_Ps': capillary_pressures_array[i],
                                    'pressures': pressures_array[i],
                                    'velocities': velocities_array[i]}
        files_names.append(str(i) + '_refined.vtu')
        files_descriptions.append(str(i))
        cfd.netgrid.save_cells('inOut/' + files_names[i])
    save_files_collection_to_file(file_name, files_names, files_descriptions)

    # for pores in cfd.netgrid.throats_pores.values():
    # print('press_grad:', cfd.pnm.pressures[pores[0]] - cfd.pnm.pressures[pores[1]])
    # print('pores: ', pores[0], pores[1])
