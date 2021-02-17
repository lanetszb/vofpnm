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

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, '../../'))

pores_coordinates = {0: [1., 2.], 1: [1., -2.], 2: [5., 0.], 3: [7., 0.],
                     4: [9., 2.], 5: [9., -2.]}
throats_pores = {0: [0, 2], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [3, 5]}
throats_widths = {0: 0.1, 1: 0.2, 2: 0.25, 3: 0.15, 4: 0.15}
throats_depths = {0: 0.15, 1: 0.35, 2: 0.6, 3: 0.25, 4: 0.25}

boundary_pores = {'inlet_pores': [0, 1], 'outlet_pores': [4, 5]}
boundary_throats = {'inlet_throats': [0, 1], 'outlet_throats': [3, 4]}

# be cautious, do not rewrite existing models
json_file_name = 'inOut/model_test.txt'
with open(json_file_name, 'w') as f:
    json.dump({'pores_coordinates': pores_coordinates, 'throats_pores': throats_pores,
               'throats_widths': throats_widths, 'throats_depths': throats_depths,
               'boundary_pores': boundary_pores, 'boundary_throats': boundary_throats},
              f, sort_keys=True, indent=4 * ' ', ensure_ascii=False)

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
