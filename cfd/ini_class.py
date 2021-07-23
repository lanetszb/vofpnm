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

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))

from netgrid import Netgrid, save_files_collection_to_file
from vofpnm import Pnm
from vofpnm import Props, Boundary, Local, Convective, Equation


class Ini:
    def __init__(s, config_file):
        s.__config = configparser.ConfigParser()
        s.__config.read(config_file)
        get = s.__config.get

        ################################
        # creating grid with Netgrid
        #################################

        json_file_name = str(get('Properties_grid', 'case_name'))
        with open('inOut/' + json_file_name) as f:
            data = json.load(f)

        s.pores_coordinates = {int(key): value for key, value in data['pores_coordinates'].items()}
        s.throats_pores = {int(key): value for key, value in data['throats_pores'].items()}
        s.throats_widths = {int(key): value for key, value in data['throats_widths'].items()}
        s.throats_depths = {int(key): value for key, value in data['throats_depths'].items()}

        s.inlet_pores = set(data['boundary_pores']['inlet_pores'])
        s.outlet_pores = set(data['boundary_pores']['outlet_pores'])
        s.inlet_throats = set(data['boundary_throats']['inlet_throats'])
        s.outlet_throats = set(data['boundary_throats']['outlet_throats'])

        s.delta_V = float(get('Properties_grid', 'delta_V'))
        s.min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))

        s.netgrid = Netgrid(s.pores_coordinates, s.throats_pores,
                            s.throats_widths, s.throats_depths, s.delta_V, s.min_cells_N,
                            s.inlet_pores, s.outlet_pores)
        s.netgrid.save_cells('cells.vtu')

        #############
        # PNM
        #############

        s.paramsPnm = {'a_dens_fluid1': float(get('Properties_fluid1', 'a_dens_fluid1')),
                       'b_dens_fluid1': float(get('Properties_fluid1', 'b_dens_fluid1')),
                       'visc_1': float(get('Properties_fluid1', 'visc_1')),
                       'dens_0': float(get('Properties_fluid0', 'dens_0')),
                       'visc_0': float(get('Properties_fluid0', 'visc_0')),
                       'pressure_in': float(get('Properties_simulation', 'pressure_in')),
                       'pressure_out': float(get('Properties_simulation', 'pressure_out')),
                       'it_accuracy': float(get('Properties_simulation', 'it_accuracy')),
                       'solver_method': str(get('Properties_simulation', 'solver_method'))}

        s.pore_n = s.netgrid.pores_N
        s.throats_denss = np.tile(s.paramsPnm['b_dens_fluid1'], s.netgrid.throats_N)

        if s.paramsPnm['visc_1'] < s.paramsPnm['visc_0']:
            s.visc_ref = s.paramsPnm['visc_1']
        else:
            s.visc_ref = s.paramsPnm['visc_0']

        s.throats_viscs = np.tile(s.visc_ref, s.netgrid.throats_N)
        s.throats_capillary_pressures = np.tile(0., s.netgrid.throats_N)

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
        s.throats_velocities = None
        s.flow_0_ref = None
        s.flow_1_ref = None

        #############
        # VOF
        #############

        s.sat_ini = float(get('Properties_vof', 'sat_ini'))
        s.sat_inlet = float(get('Properties_vof', 'sat_inlet'))
        s.sat_outlet = float(get('Properties_vof', 'sat_outlet'))

        s.sats_curr = np.tile(s.sat_ini, s.netgrid.cells_N)
        # fill the first cell in the inlet throats
        for i in s.netgrid.types_cells['inlet']:
            s.sats_curr[i] = s.sat_inlet

        s.contact_angle = float(get('Properties_vof', 'contact_angle'))
        s.ift = float(get('Properties_vof', 'interfacial_tension'))
        s.power_coeff = float(get('Properties_vof', 'power_coeff'))

        coeff = 2. * abs(math.cos(math.radians(s.contact_angle))) * s.ift
        # coeff = 0.
        throats_capillary_pressures_max = dict(
            (thr, (coeff / s.throats_widths[thr])) for thr in s.throats_widths)
        # print(throats_capillary_pressures_max)
        # throats_capillary_pressures_max = dict(
        #     (thr, (coeff / s.throats_widths[thr]) + (coeff / s.throats_depths[thr]))
        #     for thr in s.throats_widths)
        s.throats_capillary_pressures_max = np.array(list(throats_capillary_pressures_max.values()))

        # fully fill inlet throats
        # s.sats_curr = np.tile(s.sat_ini, s.netgrid.cells_N)
        # for throat in s.inlet_throats:
        #     for cell in s.netgrid.throats_cells[throat]:
        #         s.sats_curr[cell] = s.sat_inlet

        # fully fill particular number of cells in inlet throats
        # mult = 3. / 4.
        # mult = 0
        # s.sats_curr = np.tile(s.sat_ini, s.netgrid.cells_N)
        # for throat in s.inlet_throats:
        #     cells = s.netgrid.throats_cells[throat]
        #     for i in range(int(mult * len(cells))):
        #         s.sats_curr[cells[i]] = s.sat_inlet

        s.sats_prev = copy.deepcopy(s.sats_curr)
        s.sats_arrays = {"sats_curr": s.sats_curr,
                         "sats_prev": s.sats_prev}
        s.netgrid.cells_arrays = s.sats_arrays

        s.time_period = float(get('Properties_vof', 'time_period'))  # sec
        s.const_time_step = float(get('Properties_vof', 'const_time_step'))  # sec
        s.time_step_type = str(get('Properties_vof', 'time_step_type'))  # sec
        s.tsm = float(get('Properties_vof', 'tsm'))
        s.round_output_time = float(get('Properties_vof', 'round_output_time'))
        s.output_time_step = float(get('Properties_vof', 'output_time_step'))
        s.sat_trim = float(get('Properties_vof', 'sat_trim'))
        s.params = {'time_period': s.time_period, 'const_time_step': s.const_time_step,
                    'tsm': s.tsm, 'sat_trim': s.sat_trim}

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
        s.throats_av_sats = None

        #############
        # Grid
        #############

        s.grid_volume = float(get('Properties_grid', 'delta_V'))

    def initialize_sats(s):
        s.sats_curr = np.tile(s.sat_ini, s.netgrid.cells_N)
        # fill the first cell in the inlet throats
        for i in s.netgrid.types_cells['inlet']:
            s.sats_curr[i] = s.sat_inlet

        s.sats_prev = copy.deepcopy(s.sats_curr)
        s.sats_arrays = {"sats_curr": s.sats_curr,
                         "sats_prev": s.sats_prev}
        s.netgrid.cells_arrays = s.sats_arrays

        s.equation.sats = [s.sats_curr, s.sats_prev]
        s.sats_init = copy.deepcopy(s.equation.sats[s.equation.i_curr])
        s.sats_time = [s.sats_init]

        s.throats_viscs = np.tile(s.visc_ref, s.netgrid.throats_N)
        s.throats_capillary_pressures = np.tile(0., s.netgrid.throats_N)
