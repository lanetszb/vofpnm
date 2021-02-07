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

#include "Local.h"
#include "funcs.h"
#include <algorithm>

Local::Local(std::shared_ptr<Props> props, std::shared_ptr<Netgrid> netgrid) :
        _props(props),
        _netgrid(netgrid),
        _alphas(_netgrid->_cellsN, 0) {}

void Local::calcTimeSteps() {

    auto &time = std::get<double>(_props->_params["time_period"]);
    auto &timeStep = std::get<double>(_props->_params["time_step"]);
    double division = time / timeStep;
    double fullStepsN;
    auto lastStep = std::modf(division, &fullStepsN);

    _timeSteps.clear();
    _timeSteps = std::vector<double>(fullStepsN, timeStep);
    if (lastStep > 0)
        _timeSteps.push_back(lastStep * timeStep);
}

double Local::calcFlowVariableTimeStep(std::map<uint32_t, double> &thrsVelocities) {

    auto &maxCourant = std::get<double>(_props->_params["max_courant"]);

    std::vector<double> values;
    for (auto &[throat, velocity]: thrsVelocities)
        values.push_back(_netgrid->_throatsDLs[throat] / fabs(velocity));
    auto &min = *min_element(values.begin(), values.end());

    double timeStep = maxCourant * min;

    return timeStep;
}

double Local::calcDivVariableTimeStep(Eigen::Ref<Eigen::VectorXd> sats,
                                      std::map<uint32_t, double> &thrsVelocities) {

    auto &maxCourant = std::get<double>(_props->_params["max_courant"]);

    std::vector<double> flows(0, _netgrid->_facesN);
    for (auto &[throat, faces]: _netgrid->_throatsFaces) {
        auto &velocity = thrsVelocities[throat];
        uint32_t faceCurr;
        for (auto &face : faces) {
            faceCurr = face;
            flows[face] = velocity * _netgrid->_throatsSs[throat];
        }
        flows[faceCurr] *= -1;
    }

    std::vector<double> satFlows(0, _netgrid->_facesN);
    for (uint32_t face = 0; face < _netgrid->_facesN; face++) {

        auto &cells = _netgrid->_neighborsCells[face];
        auto &facesAss = _netgrid->_neighborsCellsFaces[face];
        auto &normals = _netgrid->_normalsNeighborsCells[face];

        std::vector<uint32_t> upwindCellsIdxs;
        if (cells.size() == 2 and facesAss[0] == facesAss[1]) {
            for (int i = 0; i < cells.size(); i++)
                if (normals[i] * flows[face] > 0)
                    upwindCellsIdxs.push_back(i);
        } else if (cells.size() == 1 and normals[0] * flows[face] > 0)
            upwindCellsIdxs.push_back(0);
        else
            for (int i = 0; i < cells.size(); i++)
                if (flows[facesAss[i]] < 0)
                    upwindCellsIdxs.push_back(i);


        std::vector<double> upwindFlows(cells.size(), 0);
        double sumUpwindFlows = 0;
        for (auto &i : upwindCellsIdxs) {
            upwindFlows[i] = fabs(flows[facesAss[i]]);
            sumUpwindFlows += upwindFlows[i];
        }

        for (auto &i : upwindCellsIdxs)
            satFlows[face] = sats[cells[i]] * upwindFlows[i] / sumUpwindFlows;

    }

    std::vector<double> satDivs(0, _netgrid->_cellsN);

    for (uint32_t cell = 0; cell < _netgrid->_cellsN; cell++) {
        auto &faces = _netgrid->_neighborsFaces[cell];
        auto &normals = _netgrid->_normalsNeighborsFaces[cell];

        for (int i = 0; i < 2; i++)
            satDivs[cell] += normals[i] * satFlows[faces[i]];
    }
    // todo: calc time step
    return 0;
}

void Local::calcAlphas(const double &timeStep) {

    for (auto &[throat, cells] : _netgrid->_throatsCells)
        for (auto &cell : cells) {
            auto aCoeff = calcAFunc();
            _alphas[cell] = aCoeff * _netgrid->_throatsDVs[throat] / timeStep;
        }

}