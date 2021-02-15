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
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../'))

pores_coordinates = {0: [0., 0.], 1: [0., 4.], 2: [2., 4.], 3: [-2., 4.],
                     4: [0., 8.], 5: [2., 8.], 6: [-2., 8.]}
throats_pores = {0: [0, 1], 1: [1, 2], 2: [1, 3], 3: [1, 4], 4: [2, 5], 5: [3, 6]}
throats_widths = {0: 0.13, 1: 0.08, 2: 0.4, 3: 0.13, 4: 0.08, 5: 0.4}
throats_depths = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1}

delta_V = 13.
min_cells_N = 5

boundary_pores = {'inlet_pores': [0], 'outlet_pores': [4, 5]}
boundary_throats = {'inlet_throats': [0], 'outlet_throats': [3, 4]}

json_file_name = 'inOut/model_fork.txt'
with open(json_file_name, 'w') as f:
    json.dump({'pores_coordinates': pores_coordinates, 'throats_pores': throats_pores,
               'throats_widths': throats_widths, 'throats_depths': throats_depths,
               'boundary_pores': boundary_pores, 'boundary_throats': boundary_throats},
              f, sort_keys=True, indent=4 * ' ')

with open(json_file_name) as f:
    data = json.load(f)

pores_coordinates = {int(key): value for key, value in data['pores_coordinates'].items()}
throats_pores = {int(key): value for key, value in data['throats_pores'].items()}
throats_widths = {int(key): value for key, value in data['throats_widths'].items()}
throats_depths = {int(key): value for key, value in data['throats_depths'].items()}

inlet_pores = set(data['boundary_pores']['inlet_pores'])
outlet_pores = set(data['boundary_pores']['outlet_pores'])
inlet_throats = set(data['boundary_throats']['inlet_throats'])
outlet_throats = set(data['boundary_throats']['outlet_throats'])

