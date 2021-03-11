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
sys.path.append(os.path.join(current_path, '../../'))

from netgrid import save_files_collection_to_file
from vofpnm.cfd.ini_class import Ini
from vofpnm.helpers import plot_rel_perms, plot_conesrvation_check, plot_viscs_vels, plot_av_sat


class Cfd:
    def __init__(s, ini):
        s.ini = ini

    def run_pnm(s):
        s.ini.pnm.cfd_procedure(s.ini.throats_denss, s.ini.throats_viscs,
                                s.ini.throats_capillary_pressures,
                                s.ini.newman_pores_flows, s.ini.dirichlet_pores_pressures)
        s.ini.pnm.calc_throats_flow_rates(s.ini.throats_capillary_pressures)

        mass_flows = s.ini.pnm.throats_mass_flows
        cross_secs = s.ini.netgrid.throats_Ss
        s.ini.throats_velocities = dict((thr, float(mass_flows[thr]) / cross_secs[thr])
                                        for thr in mass_flows)

    def calc_coupling_params(s):
        s.ini.equation.calc_throats_av_sats()
        s.ini.equation.calc_throats_sats_grads()

        throats_av_sats = s.ini.equation.throats_av_sats
        dens_0 = s.ini.paramsPnm['dens_0']
        dens1 = s.ini.paramsPnm['b_dens_fluid1']
        visc_0 = s.ini.paramsPnm['visc_0']
        visc_1 = s.ini.paramsPnm['visc_1']

        s.ini.throats_denss = throats_av_sats * dens_0 + (1. - throats_av_sats) * dens1
        s.ini.throats_viscs = throats_av_sats * visc_0 + (1. - throats_av_sats) * visc_1

        coeff = 2. * abs(math.cos(s.ini.contact_angle)) * s.ini.ift

        throats_widths = s.ini.throats_widths
        throats_depths = s.ini.throats_depths

        throats_capillary_pressures = dict(
            (thr, (coeff / throats_widths[thr]) + (coeff / throats_depths[thr]))
            for thr in throats_widths)
        # throats_capillary_pressures = dict(
        #     (thr, (coeff / throats_widths[thr])) for thr in throats_widths)
        throats_capillary_pressures = np.array(list(throats_capillary_pressures.values()))

        throats__coeffs = copy.deepcopy(s.ini.equation.throats_sats_grads ** 3)
        # throats__coeffs = copy.deepcopy(s.ini.equation.throats_sats_grads)
        # threshold = 0.0001
        # throats__coeffs = np.where(throats__coeffs > threshold, 1, throats__coeffs)
        # throats__coeffs = np.where(throats__coeffs <= threshold, 0, throats__coeffs)
        # throats__coeffs = np.where(throats__coeffs < -threshold, -1, throats__coeffs)
        s.ini.throats_capillary_pressures = np.multiply(throats__coeffs,
                                                        throats_capillary_pressures)

    def throats_values_to_cells(s, array):
        cells_values = np.full(s.ini.netgrid.cells_N, 0, dtype=np.float64)
        s.ini.netgrid.throats_values_to_cells(array, cells_values)
        return cells_values

    def pores_values_to_cells(s, array):
        cells_values = np.full(s.ini.netgrid.cells_N, 0, dtype=np.float64)
        s.ini.netgrid.pores_values_to_cells(array, cells_values)
        return cells_values

    def process_paraview_data(s):

        sats_to_cells = s.throats_values_to_cells(s.ini.equation.throats_av_sats)
        sats_grads_to_cells = s.throats_values_to_cells(s.ini.equation.throats_sats_grads)
        ca_pressures_to_cells = s.throats_values_to_cells(s.ini.throats_capillary_pressures)
        pressures_to_cells = s.pores_values_to_cells(s.ini.pnm.pressures)

        throats_idxs = np.arange(s.ini.netgrid.throats_N, dtype=float)
        idxs_to_cells = s.throats_values_to_cells(throats_idxs)
        velocities = np.array(list(s.ini.throats_velocities.values()))
        velocities_to_cells = s.throats_values_to_cells(velocities)
        densities_to_cells = s.throats_values_to_cells(s.ini.throats_denss)
        conductances = np.array(list(s.ini.pnm.conductances.values()))
        conductances_to_cells = s.throats_values_to_cells(conductances)

        delta_pressures = s.calc_delta_pressures()
        delta_pressures_to_cells = s.throats_values_to_cells(np.array(delta_pressures))

        pressures_to_cells = s.pores_values_to_cells(s.ini.pnm.pressures)
        sats = copy.deepcopy(s.ini.equation.sats[s.ini.equation.i_curr])
        output_1 = np.array(s.ini.local.output_1, dtype=float)
        output_2 = np.array(s.ini.local.output_2, dtype=float)

        cells_arrays = {
            'sat': sats,
            'sat_av': sats_to_cells,
            'sat_grad': sats_grads_to_cells,
            'output_1': output_1,
            'output_2': output_2,
            'capillary_Ps': ca_pressures_to_cells,
            'pressures': pressures_to_cells,
            'velocities': velocities_to_cells,
            'conductances': conductances_to_cells,
            'delta_P': delta_pressures_to_cells,
            'throats_idxs': idxs_to_cells}

        return cells_arrays

    def calc_delta_pressures(s):

        delta_pressures = []
        for pores in s.ini.netgrid.throats_pores.values():
            delta_pressures.append(
                s.ini.pnm.pressures[pores[0]] - s.ini.pnm.pressures[pores[1]])

        return delta_pressures

    def calc_flow_rates(s, mass_rates_in, mass_rates_out):

        mass_rate_in = 0
        vol_rate_in = 0
        for throat in s.ini.inlet_throats:
            first_cell = s.ini.netgrid.throats_cells[throat][0]
            velocity = s.ini.throats_velocities[throat]
            area = s.ini.netgrid.throats_Ss[throat]
            density = s.ini.paramsPnm['dens_0']
            sat = s.ini.equation.sats[s.ini.equation.i_curr][first_cell]
            mass_rate_in += velocity * area * sat * density
            vol_rate_in += velocity * area
        mass_rates_in.append(mass_rate_in)

        mass_rate_out = 0
        vol_rate_out = 0
        for throat in s.ini.outlet_throats:
            last_cell = s.ini.netgrid.throats_cells[throat][-1]
            velocity = s.ini.throats_velocities[throat]
            area = s.ini.netgrid.throats_Ss[throat]
            density = s.ini.paramsPnm['dens_0']
            sat = s.ini.equation.sats[s.ini.equation.i_curr][last_cell]
            mass_rate_out += velocity * area * sat * density
            vol_rate_out += velocity * area
        mass_rates_out.append(mass_rate_out)

        return vol_rate_in, vol_rate_out

    def calc_rel_flow_rate(s):

        flow_ref = 0
        for throat in s.ini.inlet_throats:
            velocity = s.ini.throats_velocities[throat]
            area = s.ini.netgrid.throats_Ss[throat]
            flow_ref += velocity * area

        return flow_ref

    def calc_rel_perms(s, rel_perms_0, rel_perms_1, ca_numbers, av_sats,
                       flow_0_ref, flow_1_ref, flow_curr, visc_ref):
        throats_volumes = s.ini.throats_volumes
        throats_av_sats = s.ini.equation.throats_av_sats

        throats_vol_fluxes = np.absolute(np.array(list(dict(
            (throat, float(s.ini.netgrid.throats_Ss[throat] * s.ini.throats_velocities[throat]))
            for throat in s.ini.netgrid.throats_Ss).values())))

        av_sat = np.sum(throats_volumes * throats_av_sats) / np.sum(throats_volumes)
        av_bl = np.sum(throats_vol_fluxes * throats_av_sats) / np.sum(throats_vol_fluxes)

        throats_viscs = s.ini.throats_viscs
        throats_velocities = np.array(list(s.ini.throats_velocities.values()))

        throats_ca_numbers = throats_viscs * np.absolute(throats_velocities) / s.ini.ift
        ca_number = np.sum(throats_volumes * throats_ca_numbers) / np.sum(throats_volumes)
        ca_numbers.append(ca_number)

        flow_rel_0 = flow_curr / flow_0_ref
        flow_rel_1 = flow_curr / flow_1_ref

        rel_perm_0 = av_bl * flow_rel_0
        rel_perm_1 = (1. - av_bl) * flow_rel_1

        rel_perms_0.append(rel_perm_0)
        rel_perms_1.append(rel_perm_1)
        av_sats.append(av_sat)

# if __name__ == '__main__':
