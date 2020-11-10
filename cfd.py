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

__config = configparser.ConfigParser()
__config.read(sys.argv[1])
get = __config.get

################################
# creating grid with Netgrid
#################################

pores_coordinates = {0: [1., 2.], 1: [0., -3.], 2: [5., 0.], 3: [7., 0.],
                     4: [9., 2.], 5: [9., -2.]}
throats_pores = {0: [0, 2], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [3, 5]}
throats_widths = {0: 0.1, 1: 0.15, 2: 0.25, 3: 0.15, 4: 0.15}
throats_depths = {0: 0.45, 1: 0.35, 2: 0.6, 3: 0.25, 4: 0.25}
delta_L = float(get('Properties_grid', 'delta_L'))
min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))

inlet_pores = {0, 1}
outlet_pores = {4, 5}

netgrid = Netgrid(pores_coordinates, throats_pores,
                  throats_widths, throats_depths, delta_L, min_cells_N,
                  inlet_pores, outlet_pores)

#############
# Testing PNM
#############

paramsPnm = {'a_gas_dens': float(get('Properties_gas', 'a_gas_dens')),
             'b_gas_dens': float(get('Properties_gas', 'b_gas_dens')),
             'gas_visc': float(get('Properties_gas', 'gas_visc')),
             'liq_dens': float(get('Properties_liquid', 'liq_dens')),
             'liq_visc': float(get('Properties_liquid', 'liq_visc')),
             'pressure_in': float(get('Properties_simulation', 'pressure_in')),
             'pressure_out': float(get('Properties_simulation', 'pressure_out')),
             'it_accuracy': float(get('Properties_simulation', 'it_accuracy')),
             'solver_method': str(get('Properties_simulation', 'solver_method'))}

pnm = Pnm(paramsPnm, netgrid)
pore_n = len(netgrid.pores_throats)

throats_denss = np.tile(paramsPnm['b_gas_dens'], pore_n)
throats_viscs = np.tile(paramsPnm['gas_visc'], pore_n)

newman_pores_flows = {0: 5.E+2, 1: 5.E+2}
dirichlet_pores_pressures = {4: paramsPnm['pressure_out'], 5: paramsPnm['pressure_out']}
# dirichlet_pores_pressures = {3: pressure_out}

# newman_pores_flows = {}
# dirichlet_pores_pressures = {0: paramsPnm['pressure_in'], 1: paramsPnm['pressure_in'],
#                              4: paramsPnm['pressure_out'], 5: paramsPnm['pressure_out']}
capillary_pressures = np.tile(0., len(netgrid.throats_Ss))

pnm.cfd_procedure(throats_denss, throats_viscs, capillary_pressures,
                  newman_pores_flows, dirichlet_pores_pressures)

pnm.calc_thrs_flow_rates()
pnm.calc_pores_flow_rates()

# Preparing PNM output for VoF
mass_flows = pnm.thrs_flow_rates
cross_secs = netgrid.throats_Ss
vol_flows = dict((k, float(mass_flows[k]) / cross_secs[k]) for k in mass_flows)
velocities = dict((k, float(vol_flows[k]) / paramsPnm['liq_dens']) for k in vol_flows)
print('velocities: ', velocities)
print('\n')

#############
# Testing VoF
#############
sat_ini = float(get('Properties_vof', 'sat_ini'))
sat_inlet = float(get('Properties_vof', 'sat_inlet'))
sat_outlet = float(get('Properties_vof', 'sat_outlet'))

sats_curr = np.tile(sat_ini, netgrid.cells_N)
for i in netgrid.types_cells['inlet']:
    sats_curr[i] = sat_inlet
sats_prev = copy.deepcopy(sats_curr)
sats_arrays = {"sats_curr": sats_curr,
               "sats_prev": sats_prev}
netgrid.cells_arrays = sats_arrays

time_period = float(get('Properties_vof', 'time_period'))  # sec
time_step = float(get('Properties_vof', 'time_step'))  # sec
params = {'time_period': time_period, 'time_step': time_step}
props = Props(params)

boundary = Boundary(props, netgrid)
boundary_faces_one = copy.deepcopy(netgrid.types_faces['inlet'])
boundary_faces_two = copy.deepcopy(netgrid.types_faces['outlet'])
# boundary_face_one = netgrid.types_faces[key_dirichlet_one][0]
# boundary_faces_one_axis = netgrid.faces_axes[boundary_face_one]
# boundary_face_two = netgrid.types_faces[key_dirichlet_two][0]
# boundary_faces_two_axis = netgrid.faces_axes[boundary_face_two]
boundary.shift_boundary_faces(boundary_faces_one)
boundary.shift_boundary_faces(boundary_faces_two)

local = Local(props, netgrid)
local.calc_time_steps()
convective = Convective(props, netgrid)
equation = Equation(props, netgrid, local, convective)

equation.bound_groups_dirich = ['inlet']
equation.sats_bound_dirich = {'inlet': sat_inlet}
equation.bound_groups_newman = ['outlet']

# equation.sats_bound_dirich = {inlet: sat_inlet,
#                               outlet: sat_outlet}

# equation.cfd_procedure(velocities)

#############################
# Getting params for coupling
#############################
av_sats = equation.throats_av_sats
av_density = av_sats * paramsPnm['liq_dens'] + (1 - av_sats) * paramsPnm['a_gas_dens']
av_viscosity = av_sats * paramsPnm['liq_visc'] + (1 - av_sats) * paramsPnm['gas_visc']

contact_angle = float(get('Properties_vof', 'contact_angle'))
ift = float(get('Properties_vof', 'interfacial_tension'))
coeff = -2. * math.cos(contact_angle) * ift
capillary_pressures = dict((k, (coeff / throats_widths[k]) + (coeff / throats_depths[k]))
                           for k in throats_widths)
capillary_pressures = list(capillary_pressures.values())
capillary_pressures = np.array(capillary_pressures)

#################
# Coupling itself
#################
equation.sats = [sats_curr, sats_prev]
local.calc_time_steps()
sats_init = copy.deepcopy(equation.sats[equation.i_curr])
sats_time = [sats_init]

# equation.sats_time = sats_time
# sats_time.append(sats_init)
# equation.sats_time = sats_time
time = [0]
cour_number = np.empty([])
time_curr = 0

cells_values = np.full(netgrid.cells_N, 0, dtype=np.float64)

av_sats_array = []
equation.calc_throats_av_sats()
netgrid.throats_values_to_cells(equation.throats_av_sats, cells_values)
av_sats_array.append(copy.deepcopy(cells_values))

sats_grads_array = []
equation.calc_throats_sats_grads()
netgrid.throats_values_to_cells(equation.throats_sats_grads, cells_values)
sats_grads_array.append(copy.deepcopy(cells_values))
print('\n')

capillary_pressures_array = []
capillary_coeffs = copy.deepcopy(equation.throats_sats_grads)
threshold = 0.001
capillary_coeffs = np.where(capillary_coeffs > threshold, 1, capillary_coeffs)
capillary_coeffs = np.where(capillary_coeffs <= threshold, 0, capillary_coeffs)
capillary_coeffs = np.where(capillary_coeffs < -threshold, -1, capillary_coeffs)
capillary_pressures_curr = np.multiply(capillary_coeffs, capillary_pressures)
netgrid.throats_values_to_cells(capillary_pressures_curr, cells_values)
capillary_pressures_array.append(copy.deepcopy(cells_values))

for time_step in local.time_steps:
    time_curr += time_step
    equation.cfd_procedure_one_step(velocities, time_step)
    equation.calc_throats_av_sats()
    equation.calc_throats_sats_grads()

    av_sats = equation.throats_av_sats
    netgrid.throats_values_to_cells(equation.throats_av_sats, cells_values)
    av_sats_array.append(copy.deepcopy(cells_values))
    av_density = av_sats * paramsPnm['liq_dens'] + (1 - av_sats) * paramsPnm['a_gas_dens']
    av_viscosity = av_sats * paramsPnm['liq_visc'] + (1 - av_sats) * paramsPnm['gas_visc']

    netgrid.throats_values_to_cells(equation.throats_sats_grads, cells_values)
    sats_grads_array.append(copy.deepcopy(cells_values))
    capillary_coeffs = copy.deepcopy(equation.throats_sats_grads)
    threshold = 0.001
    capillary_coeffs = np.where(capillary_coeffs > threshold, 1, capillary_coeffs)
    capillary_coeffs = np.where(capillary_coeffs <= threshold, 0, capillary_coeffs)
    capillary_coeffs = np.where(capillary_coeffs < -threshold, -1, capillary_coeffs)

    capillary_pressures_curr = np.multiply(capillary_coeffs, capillary_pressures)
    netgrid.throats_values_to_cells(capillary_pressures_curr, cells_values)
    capillary_pressures_array.append(copy.deepcopy(cells_values))

    sats_curr = copy.deepcopy(equation.sats[equation.i_curr])
    sats_time.append(sats_curr)
    time.append(time_curr)

    equation.print_cour_numbers(velocities, time_step)
    print(' time:', round((time_curr / time_period * 1000 * 0.1), 2), '%.')

    pnm.cfd_procedure(av_density, av_viscosity, capillary_pressures_curr,
                      newman_pores_flows, dirichlet_pores_pressures)
    print('capillary_pressures_curr: ', capillary_pressures_curr)
    print('velocities: ', velocities)
    print('\n')

    pnm.calc_thrs_flow_rates()
    pnm.calc_pores_flow_rates()

    # Preparing PNM output for VoF
    mass_flows = pnm.thrs_flow_rates
    cross_secs = netgrid.throats_Ss
    vol_flows = dict((k, float(mass_flows[k]) / cross_secs[k]) for k in mass_flows)
    velocities = dict((k, float(mass_flows[k]) / paramsPnm['liq_dens']) for k in mass_flows)

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
    netgrid.cells_arrays = {'sat': sats_time[i],
                            'sat_av': av_sats_array[i],
                            'sat_grad': sats_grads_array[i],
                            'capillary_Ps': capillary_pressures_array[i]}
    files_names.append(str(i) + '_refined.vtu')
    files_descriptions.append(str(i))
    netgrid.save_cells('inOut/' + files_names[i])

save_files_collection_to_file(file_name, files_names, files_descriptions)

# Output
# netgrid.cells_arrays = {'cells': np.arange(netgrid.cells_N, dtype=np.float64)}
# netgrid.faces_arrays = {'faces': np.arange(netgrid.faces_N, dtype=np.float64)}
#
# os.system('rm -r inOut/*.vtu')
# netgrid.save_cells('inOut/cells.vtu')
# netgrid.save_faces('inOut/faces.vtu')

# model geometry 3 thrs
# model geometry
# pores_coordinates = {0: [1., 2.], 1: [0., -3.], 2: [5., 0.], 3: [7., 0.]}
# throats_pores = {0: [0, 2], 1: [1, 2], 2: [2, 3]}
# throats_widths = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}
# throats_depths = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
# delta_L = 0.05
# min_cells_N = 10
#
# inlet_pores = {0, 1}
# outlet_pores = {3}

# model geometry 1D
# pores_coordinates = {0: [0., 0.], 1: [1., 0.], 2: [2., 0.], 3: [3., 0.]}
# throats_pores = {0: [0, 1], 1: [1, 2], 2: [2, 3]}
# throats_widths = {0: 0.1, 1: 0.1, 2: 0.1}
# throats_depths = {0: 0.25, 1: 0.25, 2: 0.25}
# delta_L = 0.05
# min_cells_N = 10
#
# inlet_pores = {0}
# outlet_pores = {3}
