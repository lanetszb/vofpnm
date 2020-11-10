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

        s.pores_coordinates = {0: [1., 2.], 1: [1., -2.], 2: [5., 0.], 3: [7., 0.],
                               4: [9., 2.], 5: [9., -2.]}
        s.throats_pores = {0: [0, 2], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [3, 5]}
        s.throats_widths = {0: 0.1, 1: 0.1, 2: 0.25, 3: 0.15, 4: 0.15}
        s.throats_depths = {0: 0.35, 1: 0.35, 2: 0.6, 3: 0.25, 4: 0.25}
        s.delta_L = float(get('Properties_grid', 'delta_L'))
        s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))

        s.inlet_pores = {0, 1}
        s.outlet_pores = {4, 5}

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
        s.throats_denss = np.tile(s.paramsPnm['b_gas_dens'], s.pore_n)
        s.throats_viscs = np.tile(s.paramsPnm['gas_visc'], s.pore_n)
        s.capillary_pressures = np.tile(0., s.pore_n)
        s.newman_pores_flows = {}
        s.dirichlet_pores_pressures = {0: s.paramsPnm['pressure_in'],
                                       1: s.paramsPnm['pressure_in'],
                                       4: s.paramsPnm['pressure_out'],
                                       5: s.paramsPnm['pressure_out']}

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

        print(s.velocities)

    def calc_coupling_params(s):
        s.equation.calc_throats_av_sats()
        s.equation.calc_throats_sats_grads()

        s.av_sats = s.equation.throats_av_sats
        s.av_density = s.av_sats * s.paramsPnm['liq_dens'] + \
                       (1 - s.av_sats) * s.paramsPnm['a_gas_dens']
        s.av_viscosity = s.av_sats * s.paramsPnm['liq_visc'] + \
                         (1 - s.av_sats) * s.paramsPnm['gas_visc']

        # coeff = 2. * abs(math.cos(s.contact_angle)) * s.ift
        coeff = 0.
        # ToDo: make more readable
        s.capillary_pressures = dict((k, (coeff / s.throats_widths[k]) +
                                      (coeff / s.throats_depths[k]))
                                     for k in s.throats_widths)
        s.capillary_pressures = list(s.capillary_pressures.values())
        s.capillary_pressures = np.array(s.capillary_pressures)
        ###########################
        capillary_coeffs = copy.deepcopy(s.equation.throats_sats_grads)
        threshold = 0.001
        capillary_coeffs = np.where(capillary_coeffs > threshold, 1, capillary_coeffs)
        capillary_coeffs = np.where(capillary_coeffs <= threshold, 0, capillary_coeffs)
        capillary_coeffs = np.where(capillary_coeffs < -threshold, -1, capillary_coeffs)
        s.capillary_pressures = np.multiply(capillary_coeffs, s.capillary_pressures)

    def throats_values_to_cells(s, array_arrays, array):
        cells_values = np.full(s.netgrid.cells_N, 0, dtype=np.float64)
        s.netgrid.throats_values_to_cells(array, cells_values)
        array_arrays.append(copy.deepcopy(cells_values))

    def pores_values_to_cells(s, array_arrays, array):
        cells_values = np.full(s.netgrid.cells_N, 0, dtype=np.float64)
        s.netgrid.pores_values_to_cells(array, cells_values)
        array_arrays.append(copy.deepcopy(cells_values))


if __name__ == '__main__':
    cfd = Cfd(config_file=sys.argv[1])
    cfd.calc_coupling_params()

    cfd.newman_pores_flows = {0: 2.E-1, 1: 2.E-1}
    cfd.dirichlet_pores_pressures = {4: cfd.paramsPnm['pressure_out'],
                                     5: cfd.paramsPnm['pressure_out']}
    cfd.run_pnm()

    pressures_array = []
    av_sats_array = []
    sats_grads_array = []
    capillary_pressures_array = []

    cfd.throats_values_to_cells(av_sats_array, cfd.equation.throats_av_sats)
    cfd.throats_values_to_cells(sats_grads_array, cfd.equation.throats_sats_grads)
    cfd.throats_values_to_cells(capillary_pressures_array, cfd.capillary_pressures)
    cfd.pores_values_to_cells(pressures_array, cfd.pnm.pressures)

    time = [0]
    cour_number = np.empty([])
    time_curr = 0

    for cfd.time_step in cfd.local.time_steps:
        time_curr += cfd.time_step
        cfd.equation.cfd_procedure_one_step(cfd.velocities, cfd.time_step)

        cfd.calc_coupling_params()
        cfd.throats_values_to_cells(av_sats_array, cfd.equation.throats_av_sats)
        cfd.throats_values_to_cells(sats_grads_array, cfd.equation.throats_sats_grads)
        cfd.throats_values_to_cells(capillary_pressures_array, cfd.capillary_pressures)
        cfd.sats_time.append(copy.deepcopy(cfd.equation.sats[cfd.equation.i_curr]))
        time.append(time_curr)

        cfd.equation.print_cour_numbers(cfd.velocities, cfd.time_step)
        print(' time:', round((time_curr / cfd.time_period * 1000 * 0.1), 2), '%.')
        cfd.run_pnm()
        cfd.pores_values_to_cells(pressures_array, cfd.pnm.pressures)

        for pores in cfd.netgrid.throats_pores.values():
            print('press_grad:', cfd.pnm.pressures[pores[0]] - cfd.pnm.pressures[pores[1]])
        print('P_c:', cfd.capillary_pressures)
        print('velocities: ', cfd.velocities)
        print('mass_rates:', cfd.pnm.throats_flow_rates)
        print('pressures:', cfd.pnm.pressures)

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
                                    'pressures': pressures_array[i]}
        files_names.append(str(i) + '_refined.vtu')
        files_descriptions.append(str(i))
        cfd.netgrid.save_cells('inOut/' + files_names[i])
    save_files_collection_to_file(file_name, files_names, files_descriptions)
