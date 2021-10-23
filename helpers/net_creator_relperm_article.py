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
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))


def plot_line(ax, ob, zorder=1, linewidth=3, alpha=1):
    x, y = ob.xy
    ax.plot(x, y, linewidth=linewidth, solid_capstyle='round', zorder=zorder, alpha=alpha)
    plt.show()


def create_net(dims, length, width_range, width_step):
    dims = dims
    length = length
    depth = width_step * 16

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

    rand_move = width_range[0] * 0.5
    pore_n = 0
    for y in range(dims[1]):
        for x in range(dims[0]):
            sign = random.choice((-1, 1))
            if x_coord_min < pores_coordinates[pore_n][0] < x_coord_max:
                pores_coordinates[pore_n][0] = pores_coordinates[pore_n][0] + sign * rand_move
                pores_coordinates[pore_n][1] = pores_coordinates[pore_n][1] + sign * rand_move
            pore_n += 1

    throats_pores = dict()
    throat_n = 0
    pore_n = 0
    it_n = 0

    for col in range(dims[1]):
        it_skip_freq = random.randint(2, 10)
        for row in range(dims[0] - 1):
            if it_n % it_skip_freq != 0 or \
                    pores_coordinates[int(pore_n + row)][0] == x_coord_min or \
                    pores_coordinates[int(pore_n + row + 1)][0] == x_coord_max:
                throats_pores[throat_n] = [int(pore_n + row), int(pore_n + row + 1)]
                throat_n += 1
                it_n += 1
            else:
                it_n += 1
        pore_n += dims[0]
    throat_n_horiz = copy.deepcopy(throat_n)

    it_n = 0
    throat_n = len(throats_pores.keys())
    pore_n = int(0)
    for col in range(dims[1] - 1):
        it_skip_freq = random.randint(2, 10)
        for row in range(dims[0]):
            if pores_coordinates[int(pore_n + row)][0] != x_coord_min and \
                    pores_coordinates[int(pore_n + row)][0] != x_coord_max:
                if it_n % it_skip_freq != 0:
                    throats_pores[throat_n] = [int(pore_n + row), int(pore_n + row + dims[0])]
                    throat_n += 1
                    it_n += 1
                else:
                    it_n += 1
        pore_n += dims[0]
    throat_n_vert = throat_n

    throats_widths = dict()
    throats_depths = dict()

    min_int = int(width_range[0] / width_step)
    max_int = int(width_range[1] / width_step)

    for throat in range(throat_n_horiz):
        throats_depths[throat] = depth
        if x_coord_min == pores_coordinates[throats_pores[throat][0]][0] or x_coord_min == \
                pores_coordinates[throats_pores[throat][1]][0]:
            throats_widths[throat] = width_step * int(max_int / 2) * 2
        else:
            throats_widths[throat] = width_step * random.randint(min_int / 2, max_int / 2) * 2

    for throat in range(throat_n_horiz, throat_n_vert):
        throats_depths[throat] = depth
        throats_widths[throat] = width_step * random.randint(min_int / 2, max_int / 2) * 2

    thrs_to_thrs_horiz_neighbs = defaultdict(list)
    for throat in range(throat_n_horiz):
        pores_in_throat = throats_pores[throat]
        for neighb_throat in range(throat_n_horiz):
            for pore in throats_pores[neighb_throat]:
                if throat != neighb_throat and pore in pores_in_throat:
                    thrs_to_thrs_horiz_neighbs[throat].append(neighb_throat)

    thrs_to_thrs_vert_neighbs = defaultdict(list)
    for throat in range(throat_n_horiz, throat_n_vert):
        pores_in_throat = throats_pores[throat]
        for neighb_throat in range(throat_n_horiz, throat_n_vert):
            for pore in throats_pores[neighb_throat]:
                if throat != neighb_throat and pore in pores_in_throat:
                    thrs_to_thrs_vert_neighbs[throat].append(neighb_throat)

    print('thrs_to_thrs_horiz_neighbs', thrs_to_thrs_horiz_neighbs)
    print('thrs_to_thrs_vert_neighbs', thrs_to_thrs_vert_neighbs)
    # sys.exit(0)

    throats_polygons = {}
    fig, ax = plt.subplots()

    for i in range(throat_n_horiz):
        line = LineString([(pores_coordinates[throats_pores[i][0]][0],
                            pores_coordinates[throats_pores[i][0]][1]),
                           (pores_coordinates[throats_pores[i][1]][0],
                            pores_coordinates[throats_pores[i][1]][1])])

        offset = throats_widths[i] / 2.
        line0 = line.parallel_offset(offset, side='right')
        line1 = line.parallel_offset(offset, side='left')

        plot_line(ax, line)
        plot_line(ax, line0)
        plot_line(ax, line1)

        polygon0 = []
        polygon1 = []

        distance = offset
        points_interp = line.interpolate(distance)
        dx = np.abs(line.xy[0][0] - points_interp.xy[0][0])
        dy = np.abs(line.xy[1][0] - points_interp.xy[1][0])

        if pores_coordinates[throats_pores[i][0]][0] == x_coord_min:
            if line0.xy[0][0] < line0.xy[0][1]:
                if len(thrs_to_thrs_horiz_neighbs[i]) == 1:
                    if throats_widths[i] < throats_widths[thrs_to_thrs_horiz_neighbs[i][0]]:
                        polygon0 = [[x_coord_min, line0.xy[1][0], 0],
                                    [x_coord_min, line0.xy[1][0], depth],
                                    [line0.xy[0][1] + dx, line0.xy[1][1] + dy, 0],
                                    [line0.xy[0][1] + dx, line0.xy[1][1] + dy, depth]]
                    else:
                        polygon0 = [[x_coord_min, line0.xy[1][0], 0],
                                    [x_coord_min, line0.xy[1][0], depth],
                                    [line0.xy[0][1], line0.xy[1][1], 0],
                                    [line0.xy[0][1], line0.xy[1][1], depth]]

                else:
                    polygon0 = [[x_coord_min, line0.xy[1][0], 0],
                                [x_coord_min, line0.xy[1][0], depth],
                                [line0.xy[0][1], line0.xy[1][1], 0],
                                [line0.xy[0][1], line0.xy[1][1], depth]]

            else:
                if len(thrs_to_thrs_horiz_neighbs[i]) == 1:
                    if throats_widths[i] < throats_widths[thrs_to_thrs_horiz_neighbs[i][0]]:
                        polygon0 = [[line0.xy[0][0] + dx, line0.xy[1][0] + dy, 0],
                                    [line0.xy[0][0] + dx, line0.xy[1][0] + dy, depth],
                                    [x_coord_min, line0.xy[1][1], 0],
                                    [x_coord_min, line0.xy[1][1], depth]]
                    else:
                        polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                    [line0.xy[0][0], line0.xy[1][0], depth],
                                    [x_coord_min, line0.xy[1][1], 0],
                                    [x_coord_min, line0.xy[1][1], depth]]

                else:
                    polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                [line0.xy[0][0], line0.xy[1][0], depth],
                                [x_coord_min, line0.xy[1][1], 0],
                                [x_coord_min, line0.xy[1][1], depth]]

            if line1.xy[0][0] < line1.xy[0][1]:
                if len(thrs_to_thrs_horiz_neighbs[i]) == 1:
                    if throats_widths[i] < throats_widths[thrs_to_thrs_horiz_neighbs[i][0]]:
                        polygon1 = [[x_coord_min, line1.xy[1][0], 0],
                                    [x_coord_min, line1.xy[1][0], depth],
                                    [line1.xy[0][1] + dx, line1.xy[1][1] + dy, 0],
                                    [line1.xy[0][1] + dx, line1.xy[1][1] + dy, depth]]

                    else:
                        polygon1 = [[x_coord_min, line1.xy[1][0], 0],
                                    [x_coord_min, line1.xy[1][0], depth],
                                    [line1.xy[0][1], line1.xy[1][1], 0],
                                    [line1.xy[0][1], line1.xy[1][1], depth]]

                else:
                    polygon1 = [[x_coord_min, line1.xy[1][0], 0],
                                [x_coord_min, line1.xy[1][0], depth],
                                [line1.xy[0][1], line1.xy[1][1], 0],
                                [line1.xy[0][1], line1.xy[1][1], depth]]

            else:
                if len(thrs_to_thrs_horiz_neighbs[i]) == 1:
                    if throats_widths[i] < throats_widths[thrs_to_thrs_horiz_neighbs[i][0]]:
                        polygon1 = [[line1.xy[0][0] + dx, line1.xy[1][0] + dy, 0],
                                    [line1.xy[0][0] + dx, line1.xy[1][0] + dy, depth],
                                    [x_coord_min, line1.xy[1][1], 0],
                                    [x_coord_min, line1.xy[1][1], depth]]

                    else:
                        polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                    [line1.xy[0][0], line1.xy[1][0], depth],
                                    [x_coord_min, line1.xy[1][1], 0],
                                    [x_coord_min, line1.xy[1][1], depth]]

                else:
                    polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                [line1.xy[0][0], line1.xy[1][0], depth],
                                [x_coord_min, line1.xy[1][1], 0],
                                [x_coord_min, line1.xy[1][1], depth]]

        elif pores_coordinates[throats_pores[i][1]][0] == x_coord_max:
            if line0.xy[0][0] > line0.xy[0][1]:
                if len(thrs_to_thrs_horiz_neighbs[i]) == 1:
                    if throats_widths[i] < throats_widths[thrs_to_thrs_horiz_neighbs[i][0]]:
                        polygon0 = [[x_coord_max, line0.xy[1][0], 0],
                                    [x_coord_max, line0.xy[1][0], depth],
                                    [line0.xy[0][1] - dx, line0.xy[1][1] - dy, 0],
                                    [line0.xy[0][1] - dx, line0.xy[1][1] - dy, depth]]
                    else:
                        polygon0 = [[x_coord_max, line0.xy[1][0], 0],
                                    [x_coord_max, line0.xy[1][0], depth],
                                    [line0.xy[0][1], line0.xy[1][1], 0],
                                    [line0.xy[0][1], line0.xy[1][1], depth]]

                else:
                    polygon0 = [[x_coord_max, line0.xy[1][0], 0],
                                [x_coord_max, line0.xy[1][0], depth],
                                [line0.xy[0][1], line0.xy[1][1], 0],
                                [line0.xy[0][1], line0.xy[1][1], depth]]
            else:
                if len(thrs_to_thrs_horiz_neighbs[i]) == 1:
                    if throats_widths[i] < throats_widths[thrs_to_thrs_horiz_neighbs[i][0]]:
                        polygon0 = [[line0.xy[0][0] - dx, line0.xy[1][0] - dy, 0],
                                    [line0.xy[0][0] - dx, line0.xy[1][0] - dy, depth],
                                    [x_coord_max, line0.xy[1][1], 0],
                                    [x_coord_max, line0.xy[1][1], depth]]
                    else:
                        polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                    [line0.xy[0][0], line0.xy[1][0], depth],
                                    [x_coord_max, line0.xy[1][1], 0],
                                    [x_coord_max, line0.xy[1][1], depth]]

                else:
                    polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                [line0.xy[0][0], line0.xy[1][0], depth],
                                [x_coord_max, line0.xy[1][1], 0],
                                [x_coord_max, line0.xy[1][1], depth]]

            if line1.xy[0][0] > line1.xy[0][1]:
                if len(thrs_to_thrs_horiz_neighbs[i]) == 1:
                    if throats_widths[i] < throats_widths[thrs_to_thrs_horiz_neighbs[i][0]]:
                        polygon1 = [[x_coord_max - dx, line1.xy[1][0] - dy, 0],
                                    [x_coord_max - dx, line1.xy[1][0] - dy, depth],
                                    [line1.xy[0][1], line1.xy[1][1], 0],
                                    [line1.xy[0][1], line1.xy[1][1], depth]]
                    else:
                        polygon1 = [[x_coord_max, line1.xy[1][0], 0],
                                    [x_coord_max, line1.xy[1][0], depth],
                                    [line1.xy[0][1], line1.xy[1][1], 0],
                                    [line1.xy[0][1], line1.xy[1][1], depth]]

                else:
                    polygon1 = [[x_coord_max, line1.xy[1][0], 0],
                                [x_coord_max, line1.xy[1][0], depth],
                                [line1.xy[0][1], line1.xy[1][1], 0],
                                [line1.xy[0][1], line1.xy[1][1], depth]]
            else:
                if len(thrs_to_thrs_horiz_neighbs[i]) == 1:
                    if throats_widths[i] < throats_widths[thrs_to_thrs_horiz_neighbs[i][0]]:
                        polygon1 = [[line1.xy[0][0] - dx, line1.xy[1][0] - dy, 0],
                                    [line1.xy[0][0] - dx, line1.xy[1][0] - dy, depth],
                                    [x_coord_max, line1.xy[1][1], 0],
                                    [x_coord_max, line1.xy[1][1], depth]]

                    else:
                        polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                    [line1.xy[0][0], line1.xy[1][0], depth],
                                    [x_coord_max, line1.xy[1][1], 0],
                                    [x_coord_max, line1.xy[1][1], depth]]
                else:
                    polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                [line1.xy[0][0], line1.xy[1][0], depth],
                                [x_coord_max, line1.xy[1][1], 0],
                                [x_coord_max, line1.xy[1][1], depth]]

        else:
            if len(thrs_to_thrs_horiz_neighbs[i]) < 1:
                polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                            [line0.xy[0][0], line0.xy[1][0], depth],
                            [line0.xy[0][1], line0.xy[1][1], 0],
                            [line0.xy[0][1], line0.xy[1][1], depth]]
                polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                            [line1.xy[0][0], line1.xy[1][0], depth],
                            [line1.xy[0][1], line1.xy[1][1], 0],
                            [line1.xy[0][1], line1.xy[1][1], depth]]
            else:
                for neighb_throat in thrs_to_thrs_horiz_neighbs[i]:
                    if throats_widths[i] < throats_widths[neighb_throat]:
                        if i < neighb_throat:
                            if line0.xy[0][0] < line0.xy[0][1]:
                                polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                            [line0.xy[0][0], line0.xy[1][0], depth],
                                            [line0.xy[0][1] + dx, line0.xy[1][1] + dy, 0],
                                            [line0.xy[0][1] + dx, line0.xy[1][1] + dy, depth]]
                            else:
                                polygon0 = [[line0.xy[0][0] + dx, line0.xy[1][0] + dy, 0],
                                            [line0.xy[0][0] + dx, line0.xy[1][0] + dy, depth],
                                            [line0.xy[0][1], line0.xy[1][1], 0],
                                            [line0.xy[0][1], line0.xy[1][1], depth]]

                            if line1.xy[0][0] < line1.xy[0][1]:
                                polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                            [line1.xy[0][0], line1.xy[1][0], depth],
                                            [line1.xy[0][1] + dx, line1.xy[1][1] + dy, 0],
                                            [line1.xy[0][1] + dx, line1.xy[1][1] + dy, depth]]
                            else:
                                polygon1 = [[line1.xy[0][0] + dx, line1.xy[1][0] + dy, 0],
                                            [line1.xy[0][0] + dx, line1.xy[1][0] + dy, depth],
                                            [line1.xy[0][1], line1.xy[1][1], 0],
                                            [line1.xy[0][1], line1.xy[1][1], depth]]

                        elif i > neighb_throat:
                            print('i', i, 'neighb_throat', neighb_throat)
                            print('throats_widths[i]', throats_widths[i],
                                  'throats_widths[neighb_throat]', throats_widths[neighb_throat])
                            print('dx', dx, 'dy', dy)
                            print(line0.xy)
                            print()
                            if line0.xy[0][0] < line0.xy[0][1]:
                                polygon0 = [[line0.xy[0][0] - dx, line0.xy[1][0] - dy, 0],
                                            [line0.xy[0][0] - dx, line0.xy[1][0] - dy, depth],
                                            [line0.xy[0][1], line0.xy[1][1], 0],
                                            [line0.xy[0][1], line0.xy[1][1], depth]]

                            else:
                                polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                            [line0.xy[0][0], line0.xy[1][0], depth],
                                            [line0.xy[0][1] - dx, line0.xy[1][1] - dy, 0],
                                            [line0.xy[0][1] - dx, line0.xy[1][1] - dy, depth]]

                            if line1.xy[0][0] < line1.xy[0][1]:
                                polygon1 = [[line1.xy[0][0] - dx, line1.xy[1][0] - dy, 0],
                                            [line1.xy[0][0] - dx, line1.xy[1][0] - dy, depth],
                                            [line1.xy[0][1], line1.xy[1][1], 0],
                                            [line1.xy[0][1], line1.xy[1][1], depth]]
                            else:
                                polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                            [line1.xy[0][0], line1.xy[1][0], depth],
                                            [line1.xy[0][1] - dx, line1.xy[1][1] - dy, 0],
                                            [line1.xy[0][1] - dx, line1.xy[1][1] - dy, depth]]

                    else:
                        polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                    [line0.xy[0][0], line0.xy[1][0], depth],
                                    [line0.xy[0][1], line0.xy[1][1], 0],
                                    [line0.xy[0][1], line0.xy[1][1], depth]]
                        polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                    [line1.xy[0][0], line1.xy[1][0], depth],
                                    [line1.xy[0][1], line1.xy[1][1], 0],
                                    [line1.xy[0][1], line1.xy[1][1], depth]]

        polygons = [polygon0, polygon1]
        throats_polygons[i] = polygons

    for i in range(throat_n_horiz, throat_n_vert):
        line = LineString([(pores_coordinates[throats_pores[i][0]][0],
                            pores_coordinates[throats_pores[i][0]][1]),
                           (pores_coordinates[throats_pores[i][1]][0],
                            pores_coordinates[throats_pores[i][1]][1])])

        offset = throats_widths[i] / 2.
        line0 = line.parallel_offset(offset, side='right')
        line1 = line.parallel_offset(offset, side='left')
        plot_line(ax, line)
        plot_line(ax, line0)
        plot_line(ax, line1)

        distance = offset
        points_interp = line.interpolate(distance)
        dx = np.abs(line.xy[0][0] - points_interp.xy[0][0])
        dy = np.abs(line.xy[1][0] - points_interp.xy[1][0])

        if len(thrs_to_thrs_vert_neighbs[i]) < 1:
            if line0.xy[1][0] < line0.xy[1][1]:
                polygon0 = [[line0.xy[0][0] - dx, line0.xy[1][0] - dy, 0],
                            [line0.xy[0][0] - dx, line0.xy[1][0] - dy, depth],
                            [line0.xy[0][1] + dx, line0.xy[1][1] + dy, 0],
                            [line0.xy[0][1] + dx, line0.xy[1][1] + dy, depth]]

            else:
                polygon0 = [[line0.xy[0][0] + dx, line0.xy[1][0] + dy, 0],
                            [line0.xy[0][0] + dx, line0.xy[1][0] + dy, depth],
                            [line0.xy[0][1] - dx, line0.xy[1][1] - dy, 0],
                            [line0.xy[0][1] - dx, line0.xy[1][1] - dy, depth]]

            if line1.xy[1][0] < line1.xy[1][1]:
                polygon1 = [[line1.xy[0][0] - dx, line1.xy[1][0] - dy, 0],
                            [line1.xy[0][0] - dx, line1.xy[1][0] - dy, depth],
                            [line1.xy[0][1] + dx, line1.xy[1][1] + dy, 0],
                            [line1.xy[0][1] + dx, line1.xy[1][1] + dy, depth]]


            else:
                polygon1 = [[line1.xy[0][0] + dx, line1.xy[1][0] + dy, 0],
                            [line1.xy[0][0] + dx, line1.xy[1][0] + dy, depth],
                            [line1.xy[0][1] - dx, line1.xy[1][1] - dy, 0],
                            [line1.xy[0][1] - dx, line1.xy[1][1] - dy, depth]]
        else:
            for neighb_throat in thrs_to_thrs_vert_neighbs[i]:
                if throats_widths[i] <= throats_widths[neighb_throat]:
                    if i < neighb_throat:
                        if line0.xy[1][0] < line0.xy[1][1]:
                            polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                        [line0.xy[0][0], line0.xy[1][0], depth],
                                        [line0.xy[0][1] + dx, line0.xy[1][1] + dy, 0],
                                        [line0.xy[0][1] + dx, line0.xy[1][1] + dy, depth]]
                        else:
                            polygon0 = [[line0.xy[0][0] + dx, line0.xy[1][0] + dy, 0],
                                        [line0.xy[0][0] + dx, line0.xy[1][0] + dy, depth],
                                        [line0.xy[0][1], line0.xy[1][1], 0],
                                        [line0.xy[0][1], line0.xy[1][1], depth]]

                        if line1.xy[1][0] < line1.xy[1][1]:
                            polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                        [line1.xy[0][0], line1.xy[1][0], depth],
                                        [line1.xy[0][1] + dx, line1.xy[1][1] + dy, 0],
                                        [line1.xy[0][1] + dx, line1.xy[1][1] + dy, depth]]
                        else:
                            polygon1 = [[line1.xy[0][0] + dx, line1.xy[1][0] + dy, 0],
                                        [line1.xy[0][0] + dx, line1.xy[1][0] + dy, depth],
                                        [line1.xy[0][1], line1.xy[1][1], 0],
                                        [line1.xy[0][1], line1.xy[1][1], depth]]

                    else:
                        if line0.xy[1][0] < line0.xy[1][1]:
                            polygon0 = [[line0.xy[0][0] - dx, line0.xy[1][0] - dy, 0],
                                        [line0.xy[0][0] - dx, line0.xy[1][0] - dy, depth],
                                        [line0.xy[0][1], line0.xy[1][1], 0],
                                        [line0.xy[0][1], line0.xy[1][1], depth]]

                        else:
                            polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                        [line0.xy[0][0], line0.xy[1][0], depth],
                                        [line0.xy[0][1] - dx, line0.xy[1][1] - dy, 0],
                                        [line0.xy[0][1] - dx, line0.xy[1][1] - dy, depth]]

                        if line1.xy[1][0] < line1.xy[1][1]:
                            polygon1 = [[line1.xy[0][0] - dx, line1.xy[1][0] - dy, 0],
                                        [line1.xy[0][0] - dx, line1.xy[1][0] - dy, depth],
                                        [line1.xy[0][1], line1.xy[1][1], 0],
                                        [line1.xy[0][1], line1.xy[1][1], depth]]
                        else:
                            polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                        [line1.xy[0][0], line1.xy[1][0], depth],
                                        [line1.xy[0][1] - dx, line1.xy[1][1] - dy, 0],
                                        [line1.xy[0][1] - dx, line1.xy[1][1] - dy, depth]]

                else:
                    polygon0 = [[line0.xy[0][0], line0.xy[1][0], 0],
                                [line0.xy[0][0], line0.xy[1][0], depth],
                                [line0.xy[0][1], line0.xy[1][1], 0],
                                [line0.xy[0][1], line0.xy[1][1], depth]]
                    polygon1 = [[line1.xy[0][0], line1.xy[1][0], 0],
                                [line1.xy[0][0], line1.xy[1][0], depth],
                                [line1.xy[0][1], line1.xy[1][1], 0],
                                [line1.xy[0][1], line1.xy[1][1], depth]]

        polygons = [polygon0, polygon1]
        throats_polygons[i] = polygons

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

    json_file_name = '../inOut/validation/throats_polygons.json'
    with open(json_file_name, 'w') as f:
        json.dump({'polygons': throats_polygons,
                   'inlet_throats': inlet_throats,
                   'outlet_throats': outlet_throats},
                  f, sort_keys=True, indent=4 * ' ', ensure_ascii=False)

    boundary_pores = {'inlet_pores': inlet_pores, 'outlet_pores': outlet_pores}
    boundary_throats = {'inlet_throats': inlet_throats, 'outlet_throats': outlet_throats}

    network = {'pores_coordinates': pores_coordinates, 'throats_pores': throats_pores,
               'throats_widths': throats_widths, 'throats_depths': throats_depths,
               'boundary_pores': boundary_pores, 'boundary_throats': boundary_throats}

    return network
