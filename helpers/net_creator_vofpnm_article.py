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
import copy

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
    it_n = 0
    pore_n = 0

    # model 1
    # it_to_skip = []
    # model 2
    # it_to_skip = [22, 30, 37, 40]
    # model 3
    # it_to_skip = [7, 39, 46]
    # model 4
    # it_to_skip = [8, 17, 21]
    # model 5
    # it_to_skip = [30, 35, 41, 44, 46]
    # model 6
    it_to_skip = [23, 35, 37, 40, 45]
    # model 6
    # it_to_skip = [17, 34, 47]

    # model final_3_1 enlarged
    # it_to_skip = [3, 10, 16, 24, 29, 37]
    # model final_3_2 enlarged
    # it_to_skip = [10, 16, 24, 29]

    for col in range(dims[1]):
        for row in range(dims[0] - 1):
            if it_n not in it_to_skip:
                throats_pores[throat_n] = [int(pore_n + row), int(pore_n + row + 1)]
                throat_n += 1
                it_n += 1
            else:
                throat_n += 0
                it_n += 1
        pore_n += dims[0]
    throat_n_horiz = copy.deepcopy(throat_n)

    # model final_3_1 enlarged
    # it_to_skip = [36, 58]
    # model final_3_2 enlarged
    # it_to_skip = [38, 60]
    throat_n = len(throats_pores.keys())
    it_n = len(throats_pores.keys())
    pore_n = int(0)
    for col in range(int(dims[1] - 1)):
        for row in range(dims[0]):
            if pores_coordinates[int(pore_n + row)][0] != x_coord_min and \
                    pores_coordinates[int(pore_n + row)][0] != x_coord_max:
                if it_n not in it_to_skip:
                    throats_pores[throat_n] = [int(pore_n + row), int(pore_n + row + dims[0])]
                    throat_n += 1
                    it_n += 1
                else:
                    throat_n += 0
                    it_n += 1
        pore_n += dims[0]
    throat_n_vert = throat_n

    throats_widths = dict()
    throats_depths = dict()

    min_int = int(width_range[0] / width_step)
    max_int = int(width_range[1] / width_step)
    depth = width_step * 30

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

    pxy = dict()
    for key in throats_pores.keys():
        pore0 = throats_pores[key][0]
        pore1 = throats_pores[key][1]
        pxy[key] = [copy.deepcopy(pores_coordinates[pore0]),
                    copy.deepcopy(pores_coordinates[pore1])]
    print(pxy)

    for i in range(throat_n_horiz):
        pxy[i][0][1] = pxy[i][0][1] - throats_widths[i] / 2.
        pxy[i][1][1] = pxy[i][1][1] + throats_widths[i] / 2.
    for i in range(throat_n_horiz, throat_n_vert):
        pxy[i][0][0] = pxy[i][0][0] - throats_widths[i] / 2.
        pxy[i][1][0] = pxy[i][1][0] + throats_widths[i] / 2.

    print('pxy', pxy)
    print('width_step', width_step)

    boundary_pores = {'inlet_pores': inlet_pores, 'outlet_pores': outlet_pores}
    boundary_throats = {'inlet_throats': inlet_throats, 'outlet_throats': outlet_throats}

    network = {'pores_coordinates': pores_coordinates, 'throats_pores': throats_pores,
               'throats_widths': throats_widths, 'throats_depths': throats_depths,
               'boundary_pores': boundary_pores, 'boundary_throats': boundary_throats}

    return network

    # for throat, pores in throats_pores.items():
    #     for pore in inlet_pores:
    #         if pore in pores and pores_coordinates[throats_pores[throat][0]][0] != \
    #                 pores_coordinates[throats_pores[throat][1]][0]:
    #             inlet_throats.append(throat)
    #             break
    # for throat, pores in throats_pores.items():
    #     for pore in outlet_pores:
    #         if pore in pores and pores_coordinates[throats_pores[throat][0]][0] != \
    #                 pores_coordinates[throats_pores[throat][1]][0]:
    #             outlet_throats.append(throat)
    #             break
