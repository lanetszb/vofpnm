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
sys.path.append(os.path.join(current_path, '../../'))

from netgrid import Netgrid, save_files_collection_to_file
from vofpnm import Props, Boundary, Local, Convective, Equation

################################
# creating grid with Netgrid
#################################
# Model capillary rise
__config = configparser.ConfigParser()
__config.read(sys.argv[1])
get = __config.get

pores_coordinates = {0: [0.002, 0], 1: [0.002, 0.02]}
throats_pores = {0: [0, 1]}
throats_widths = {0: 1.e-3}
throats_depths = {0: 1.e-3}
delta_V = float(get('Properties_grid', 'delta_V'))
min_cells_N = np.uint16(get('Properties_grid', 'min_cells_N'))
#
inlet_pores = {0}
outlet_pores = {4}

netgrid = Netgrid(pores_coordinates, throats_pores,
                  throats_widths, throats_depths, delta_V, min_cells_N,
                  inlet_pores, outlet_pores)

# Obtaining velocity
# water_viscosity = 1.48e-5
# air_viscosity = 1.e-6
water_viscosity = 1.e-3
air_viscosity = 1.5e-5

pressure_in = 300.
pressuer_out = 100.
height = netgrid.throats_depths[0]
length = netgrid.throats_Ls[0]
area = netgrid.throats_Ss[0]

sigma = float(get('Properties_vof', 'contact_angle'))
ift = float(get('Properties_vof', 'interfacial_tension'))

capillary_pressure = 2. * abs(math.cos(sigma)) * ift / height


def calc_conductance(av_viscosity):
    return height ** 2 / 12 / av_viscosity / length


def calc_velocity(conductance):
    return conductance * (pressure_in - pressuer_out + capillary_pressure)


# VOF
# finding number of cells to fill by water
cells_to_fill = int(0.0111 / netgrid.throats_dLs[0])

sat_ini = float(get('Properties_vof', 'sat_ini'))
sat_inlet = float(get('Properties_vof', 'sat_inlet'))
sat_outlet = float(get('Properties_vof', 'sat_outlet'))
sats_curr = np.tile(sat_ini, netgrid.cells_N)

for cell in range(cells_to_fill):
    sats_curr[cell] = sat_inlet

sats_prev = copy.deepcopy(sats_curr)
sats_arrays = {"sats_curr": sats_curr,
               "sats_prev": sats_prev}
netgrid.cells_arrays = sats_arrays

const_time_step = float(get('Properties_vof', 'const_time_step'))  # sec
time_period = float(get('Properties_vof', 'time_period'))  # sec
time_step_type = str(get('Properties_vof', 'time_step_type'))  # sec
tsm = float(get('Properties_vof', 'tsm'))
sat_trim = float(get('Properties_vof', 'sat_trim'))
params = {'time_period': time_period, 'const_time_step': const_time_step,
          'tsm': tsm, 'sat_trim': sat_trim}

props = Props(params)
local = Local(props, netgrid)
convective = Convective(props, netgrid)
equation = Equation(props, netgrid, local, convective)

equation.bound_groups_dirich = ['inlet']
equation.sats_bound_dirich = {'inlet': sat_inlet}
equation.bound_groups_newman = ['outlet']
equation.sats = [sats_curr, sats_prev]
sats_init = copy.deepcopy(equation.sats[equation.i_curr])
sats_time = [sats_init]

equation.calc_throats_av_sats()
av_sat = equation.throats_av_sats
viscosity = av_sat * water_viscosity + (1 - av_sat) * air_viscosity
# viscosity = (water_viscosity * air_viscosity) / (
#         av_sat * air_viscosity + (1 - av_sat) * water_viscosity)

conductance = calc_conductance(viscosity)
velocity = calc_velocity(conductance)

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
time_output_freq = time_period / 100.
time_bound = time_output_freq
is_output_step = False
is_last_step = False
i = int(0)
while True:

    if time_step_type == 'const':
        time_step = const_time_step
    elif time_step_type == 'flow_variable':
        time_step = local.calc_flow_variable_time_step(velocity)
    elif time_step_type == 'div_variable':
        time_step = local.calc_div_variable_time_step(equation.sats[equation.i_curr], velocity)

    if time_curr + const_time_step >= time_bound:
        time_step = time_bound - time_curr
        time_bound += time_output_freq
        is_output_step = True

    if time_curr + const_time_step >= time_period:
        time_step = time_period - time_curr
        is_last_step = True

    time_steps.append(time_step)
    time_curr += time_step

    print('velocity:', velocity)
    equation.cfd_procedure_one_step({0: velocity}, time_step)
    sats_time.append(copy.deepcopy(equation.sats[equation.i_curr]))

    equation.calc_throats_av_sats()
    av_sat = equation.throats_av_sats
    viscosity = av_sat * water_viscosity + (1 - av_sat) * air_viscosity
    # viscosity = (water_viscosity * air_viscosity) / (
    #         av_sat * air_viscosity + (1 - av_sat) * water_viscosity)
    print('viscosity:', viscosity)
    print('vel_dp:', conductance * (pressure_in - pressuer_out))
    print('vel_pc:', conductance * capillary_pressure)

    conductance = calc_conductance(viscosity)
    velocity = calc_velocity(conductance)

    print('time_step: ', int(time_curr / time_step))
    time.append(time_curr)
    equation.print_cour_numbers({0: velocity}, time_step)
    print(' time:', round((time_curr / time_period * 1000 * 0.1), 2), '%.', '\n')

    sats_array = copy.deepcopy(equation.sats[equation.i_curr])

    if is_output_step:
        netgrid.cells_arrays = {'sat': sats_array}
        files_names.append(str(i) + '.vtu')
        files_descriptions.append(str(i))
        netgrid.save_cells('inOut/' + files_names[-1])
        save_files_collection_to_file(file_name, files_names, files_descriptions)
        i += 1
        is_output_step = False

    if is_last_step:
        break

