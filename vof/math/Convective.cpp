/* MIT License
 *
 * Copyright (c) 2020 Aleksandr Zhuravlyov and Zakhar Lanets
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "Convective.h"
#include "funcs.h"
#include <algorithm>
#include <iterator>

Convective::Convective(std::shared_ptr<Props> props,
                       std::shared_ptr<Netgrid> netgrid) :
        _props(props),
        _netgrid(netgrid),
        _betas(_netgrid->_facesN) {}

void Convective::calcBetas(std::map<uint32_t, double> &thrsVelocities) {

    for (auto &[throat, faces]: _netgrid->_throatsFaces) {
        auto &velocity = thrsVelocities[throat];
        auto bCoeff = calcBFunc(velocity);
        uint32_t faceCurr;
        for (auto &face : faces) {
            faceCurr = face;
            _betas[face] = bCoeff * _netgrid->_throatsSs[throat];
        }
        _betas[faceCurr] *= -1;
    }

}
