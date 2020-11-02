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
import copy
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../'))

from netgrid import Netgrid, save_files_collection_to_file
from vofpnm import Pnm
from vofpnm import Props, Boundary, Local, Convective, Equation

# from vofpnm import Props, Boundary, Local, Convective, Equation

# model geometry
pores_coordinates = {0: [1., 2.], 1: [0., -3.], 2: [5., 0.], 3: [7., 0.],
                     4: [9., 2.], 5: [10., -3.]}
throats_pores = {0: [0, 2], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [3, 5]}
throats_widths = {0: 0.1, 1: 0.15, 2: 0.25, 3: 0.15, 4: 0.25}
throats_depths = {0: 0.45, 1: 0.35, 2: 0.6, 3: 0.35, 4: 0.6}
delta_L = 0.01
min_cells_N = 10

inlet_pores = {0, 1}
outlet_pores = {4, 5}

netgrid = Netgrid(pores_coordinates, throats_pores,
                  throats_widths, throats_depths, delta_L, min_cells_N,
                  inlet_pores, outlet_pores)

# simulation parameters for pnm
# a_dens (kg/m3) is a coefficient for equation_diff of gas density = a * P + b
a_gas_dens = 6.71079e-06
# b_dens (kg/m3/Pa) is b coefficient for equation_diff of gas density = a * P + b
b_gas_dens = -2.37253E-02
# gas_visc (Pa*s) is constant viscosity of gas
gas_visc = 1.99E-5
# liq_dens (kg/m3) is constant density of water
liq_dens = 997.
# water_visc (Pa*s) is constant viscosity of water
liq_visc = 0.001
# pressure_in (Pa) is a PN inlet pressure
pressure_in = 275000.
# pressure_out (Pa) is a PN outlet pressure
pressure_out = 225000.
# iterative_accuracy is the accuracy for iterative procedures
it_accuracy = 1.e-17
# eigen solver method (can be biCGSTAB, sparseLU or leastSqCG)
solver_method = 'sparseLU'
# boundary conditions
boundCond = 'dirichlet'

paramsPnm = {'a_gas_dens': a_gas_dens, 'b_gas_dens': b_gas_dens,
             'gas_visc': gas_visc, 'liq_dens': liq_dens,
             'pressure_in': pressure_in, 'pressure_out': pressure_out,
             'it_accuracy': it_accuracy, 'solver_method': solver_method,
             'boundCond': boundCond}

# # Testing Pnm
pnm = Pnm(paramsPnm, netgrid)
pore_n = len(netgrid.pores_throats)

throats_denss = np.tile(liq_dens, pore_n)
throats_viscs = np.tile(liq_visc, pore_n)
newman_pores_flows = {0: 1.E+5, 1: 3.5E+5}
dirichlet_pores_pressures = {4: pressure_out, 5: pressure_out}

# newman_pores_flows = {}
# dirichlet_pores_pressures = {0: 1.E-7, 1: 1.5E-7,
#                              4: pressure_out, 5: pressure_out}

pnm.cfd_procedure(throats_denss, throats_viscs,
                  newman_pores_flows, dirichlet_pores_pressures)
pnm.calc_thrs_flow_rates()
pnm.calc_pores_flow_rates()

mass_flows = pnm.thrs_flow_rates
cross_secs = netgrid.throats_Ss

vol_flows = dict((k, float(mass_flows[k]) / cross_secs[k]) for k in mass_flows)
velocities = dict((k, float(mass_flows[k]) / liq_dens) for k in mass_flows)
# # Testing VoF
conc_ini = float(0.0)
concs_array1 = np.tile(conc_ini, netgrid.cells_N)
concs_array2 = np.tile(conc_ini, netgrid.cells_N)
concs_arrays = {"concs_array1": concs_array1,
                "concs_array2": concs_array2}

netgrid.cells_arrays = concs_arrays

# computation time
time_period = float(1)  # sec
# numerical time step
time_step = float(0.01)  # sec

# diffusivity coeffs (specify only b coeff to make free diffusion constant)
d_coeff_a = float(0)  # m2/sec
d_coeff_b = float(15.E-3)  # m2/sec
# porosity of rock
poro = float(1)
params = {'time_period': time_period, 'time_step': time_step,
          'd_coeff_a': d_coeff_a, 'd_coeff_b': d_coeff_b,
          'poro': poro}

key_dirichlet_one = 'inlet'
key_dirichlet_two = 'outlet'

props = Props(params)
boundary = Boundary(props, netgrid)

boundary_faces_one = copy.deepcopy(netgrid.types_faces[key_dirichlet_one])
boundary_faces_two = copy.deepcopy(netgrid.types_faces[key_dirichlet_two])
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

equation.bound_groups_dirich = [key_dirichlet_one, key_dirichlet_two]
# concentration on dirichlet cells
conc_left = float(1.0)
conc_right = float(0.)
equation.concs_bound_dirich = {key_dirichlet_one: conc_left,
                               key_dirichlet_two: conc_right}

equation.cfd_procedure(velocities)

os.system('rm -r inOut/*.vtu')
os.system('rm -r inOut/*.pvd')
concs_dict = dict()
file_name = 'inOut/collection.pvd'
files_names = list()
files_descriptions = list()
for i in range(len(local.time_steps)):
    netgrid.cells_arrays = {'conc_i': equation.concs_time[i]}
    files_names.append(str(i) + '.vtu')
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
