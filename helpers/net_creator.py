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
import json
import random

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))


def create_net(dims, length, width_range, width_step):
    dims = dims
    length = length

    x_coord_min = 0
    x_coord_max = 0

    pores_coordinates = dict()
    pore_n = 0
    for y in range(dims[1]):
        for x in range(dims[0]):
            pores_coordinates[pore_n] = list([int(x) * length, int(y) * length])
            if x < x_coord_min:
                x_coord_min = x * length
            if x > x_coord_max:
                x_coord_max = x * length
            pore_n += 1

    throats_pores = dict()
    throat_n = 0
    pore_n = 0

    for col in range(dims[1]):
        for row in range(dims[0] - 1):
            throats_pores[throat_n] = [int(pore_n + row), int(pore_n + row + 1)]
            throat_n += 1
        pore_n += dims[0]

    throat_n = len(throats_pores.keys())
    pore_n = int(0)
    for col in range(int(dims[1] - 1)):
        throats_skip_freq = random.randint(2, int(dims[1]))
        throats_skip_freq = 2 * 10**3
        for row in range(dims[0]):
            if pores_coordinates[int(pore_n + row)][0] != x_coord_min and \
                    pores_coordinates[int(pore_n + row)][0] != x_coord_max:
                if row % throats_skip_freq != 0:
                    throats_pores[throat_n] = [int(pore_n + row), int(pore_n + row + dims[0])]
                    throat_n += 1
        pore_n += dims[0]

    throats_widths = dict()
    throats_depths = dict()

    min_int = int(width_range[0] / width_step)
    max_int = int(width_range[1] / width_step)
    depth = width_step

    for throat in throats_pores.keys():
        throats_depths[throat] = depth
        if x_coord_min == pores_coordinates[throats_pores[throat][0]][0] or x_coord_min == \
                pores_coordinates[throats_pores[throat][1]][0]:
            throats_widths[throat] = width_step * int(max_int / 2) * 2
        else:
            throats_widths[throat] = width_step * random.randint(min_int / 2, max_int / 2) * 2

    inlet_pores = list()
    outlet_pores = list()

    x_coord_min = 0.
    x_coord_max = (dims[0] - 1) * length

    for pore in pores_coordinates.keys():
        if x_coord_min == pores_coordinates[pore][0]:
            inlet_pores.append(pore)
        if x_coord_max == pores_coordinates[pore][0]:
            outlet_pores.append(pore)

    for pore in pores_coordinates.keys():
        pores_coordinates[pore][1] = pores_coordinates[pore][1] + width_step * 6.

    inlet_throats = list()
    outlet_throats = list()

    for throat, pores in throats_pores.items():
        for pore in inlet_pores:
            if pore in pores:
                inlet_throats.append(throat)
                break

    for throat, pores in throats_pores.items():
        for pore in outlet_pores:
            if pore in pores:
                outlet_throats.append(throat)
                break

    boundary_pores = {'inlet_pores': inlet_pores, 'outlet_pores': outlet_pores}
    boundary_throats = {'inlet_throats': inlet_throats, 'outlet_throats': outlet_throats}

    network = {'pores_coordinates': pores_coordinates, 'throats_pores': throats_pores,
               'throats_widths': throats_widths, 'throats_depths': throats_depths,
               'boundary_pores': boundary_pores, 'boundary_throats': boundary_throats}

    return network
