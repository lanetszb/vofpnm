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

double Convective::weighing(const std::string &method, const double &value0,
                            const double &value1) {

    if (method == "meanAverage")
        return (value0 + value1) / 2;
    else if (method == "meanHarmonic")
        return 2. * value0 * value1 / (value0 + value1);
    else if (method == "upWind")
        return std::max(value0, value1);
    else exit(0);
}

void Convective::calcBetas(std::map<uint32_t, double> &thrsVelocities) {

    for (auto &[throat, faces]: _netgrid->_throatsFaces) {
        auto &velocity = thrsVelocities[throat];
        auto bCoeff = calcBFunc(velocity);
        for (auto &face : faces)
            _betas[face] = bCoeff * _netgrid->_throatsSs[throat] / 2.;
    }

}
