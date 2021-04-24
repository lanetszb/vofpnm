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

from net_creator import create_net

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))

dims = [int(10), int(10)]
length = 1.
depth = 0.1
width_range = [0.1, 0.3]
throats_skip_freq = 3

network_dict = create_net(dims, length, depth, width_range)

# be cautious, do not rewrite existing models
json_file_name = '../inOut/models/model_chess.txt'

with open(json_file_name, 'w') as f:
    json.dump(network_dict, f, sort_keys=True, indent=4 * ' ', ensure_ascii=False)

# pores_coordinates = {0: [0.1, 0.2], 1: [1.2, 0.3], 2: [2.3, 0.4], 3: [3.4, 0.1], 4: [4.1, 0.3], 5: [5.5, 0.4],
#                      6: [2.1, 2.1], 7: [0.3, 2.2], 8: [1.4, 2.4], 9: [3.2, 2.1], 10: [4.4, 2.3]}
# throats_pores = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [2, 6],
#                  6: [7, 8], 7: [8, 6], 8: [6, 9], 9: [9, 10]}
# throats_widths = {0: 0.15, 1: 0.2, 2: 0.3, 3: 0.12, 4: 0.13, 5: 0.18, 6: 0.11, 7: 0.21, 8: 0.19, 9: 0.23}
# throats_depths = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}
#
# boundary_pores = {'inlet_pores': inlet_pores, 'outlet_pores': outlet_pores}
# boundary_throats = {'inlet_throats': inlet_throats, 'outlet_throats': outlet_throats}
#
# # be cautious, do not rewrite existing models
# json_file_name = 'inOut/model_linear.txt'
# with open(json_file_name, 'w') as f:
#     json.dump({'pores_coordinates': pores_coordinates, 'throats_pores': throats_pores,
#                'throats_widths': throats_widths, 'throats_depths': throats_depths,
#                'boundary_pores': boundary_pores, 'boundary_throats': boundary_throats},
#               f, sort_keys=True, indent=4 * ' ', ensure_ascii=False)

# json_file_name = 'inOut/lbm_validation_fork.txt'
# with open(json_file_name) as f:
#     data = json.load(f)
#
# pores_coordinates = {int(key): value for key, value in data['pores_coordinates'].items()}
# throats_pores = {int(key): value for key, value in data['throats_pores'].items()}
# throats_widths = {int(key): value for key, value in data['throats_widths'].items()}
# throats_depths = {int(key): value for key, value in data['throats_depths'].items()}
#
# inlet_pores = set(data['boundary_pores']['inlet_pores'])
# outlet_pores = set(data['boundary_pores']['outlet_pores'])
# inlet_throats = set(data['boundary_throats']['inlet_throats'])
# outlet_throats = set(data['boundary_throats']['outlet_throats'])
