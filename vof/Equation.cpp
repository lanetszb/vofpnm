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

#include "Equation.h"
#include "eigenSetGet.h"
#include <time.h>
#include <algorithm>
#include <iterator>
#include <numeric>

Equation::Equation(std::shared_ptr<Props> props,
                   std::shared_ptr<Netgrid> netgrid,
                   std::shared_ptr<Local> local,
                   std::shared_ptr<Convective> convective) :

        _props(props),
        _netgrid(netgrid),
        _local(local),
        _convective(convective),
        dim(_netgrid->_cellsN),
        iCurr(0), iPrev(1),
        _satsIni(new double[dim], dim),
        _throatsAvSats(new double[_netgrid->_throatsN], _netgrid->_throatsN),
        _throatsSatsGrads(new double[_netgrid->_throatsN], _netgrid->_throatsN),
        matrix(dim, dim),
        freeVector(new double[dim], dim) {

    std::vector<Triplet> triplets;
    triplets.reserve(3 * dim - 4);

    for (int i = 0; i < dim; i++)
        triplets.emplace_back(i, i);

    for (auto &nonDirichCell: findNonDirichCells(_boundGroupsDirich))
        for (auto &face: _netgrid->_neighborsFaces[nonDirichCell])
            for (auto &cell: _netgrid->_neighborsCells[face])
                triplets.emplace_back(nonDirichCell, cell);

    matrix.setFromTriplets(triplets.begin(), triplets.end());
}

void Equation::processFaces(const std::set<uint32_t> &faces) {


    for (auto &face : faces) {

        auto &cells = _netgrid->_neighborsCells[face];
        auto &facesAss = _netgrid->_neighborsCellsFaces[face];
        auto &normals = _netgrid->_normalsNeighborsCells[face];

        // be very fucking careful
        std::vector<uint32_t> upwindCellsIdxs;

        if (cells.size() == 2 and facesAss[0] == facesAss[1]) {
            for (int i = 0; i < cells.size(); i++)
                if (normals[i] * _convective->_betas[face] > 0)
                    upwindCellsIdxs.push_back(i);
        } else if (normals[0] * _convective->_betas[face] > 0)
            upwindCellsIdxs.push_back(0);
        else if (cells.size() == 1 and normals[0] * _convective->_betas[face] < 0) {
            // upwindCellsIdxs.push_back(0);
            // _sats[iPrev][cells[0]] = 0.;
        } else
            for (int i = 0; i < cells.size(); i++)
                if (face != facesAss[i])
                    if (_convective->_betas[facesAss[i]] < 0)
                        upwindCellsIdxs.push_back(i);


        std::vector<double> upwindBetas(cells.size(), 0);
        double sumUpwindBetas = 0;
        for (auto &i : upwindCellsIdxs) {
            upwindBetas[i] = _convective->_betas[facesAss[i]];
            sumUpwindBetas += upwindBetas[i];
        }

        for (auto &i : upwindCellsIdxs) {
            _matrixFacesCells[face][cells[i]] =
                    _convective->_betas[face] * upwindBetas[i] / sumUpwindBetas;
            _freeFacesCells[face][cells[i]] = 0;
        }

    }

}

std::set<uint32_t> Equation::groupCellsByTypes(const std::vector<std::string> &groups) {

    std::set<uint32_t> groupedCells;
    for (auto &bound : groups) {
        auto cells = _netgrid->_typesCells.at(bound);
        groupedCells.insert(cells.begin(), cells.end());
    }

    return groupedCells;
}

std::set<uint32_t> Equation::findNonDirichCells
        (const std::vector<std::string> &boundGroupsDirich) {

    auto allCells = groupCellsByTypes({"inlet", "nonbound", "outlet", "deadend"});
    auto dirichCells = groupCellsByTypes(boundGroupsDirich);

    std::set<uint32_t> nonDirichCells;
    std::set_difference(std::begin(allCells),
                        std::end(allCells),
                        std::begin(dirichCells), std::end(dirichCells),
                        std::inserter(nonDirichCells, nonDirichCells.begin()));

    return nonDirichCells;
}

void Equation::processDirichCells(std::vector<std::string> &boundGroups,
                                  std::map<std::string, double> &satsBound) {

    for (auto &bound : boundGroups) {
        auto &conc = satsBound[bound];
        auto dirichCells = _netgrid->_typesCells.at(bound);

        for (auto &cell : dirichCells)
            freeVector[cell] = conc * _local->_alphas[cell];
    }
}

void Equation::fillMatrix() {

    for (int i = 0; i < dim; ++i)
        for (MatrixIterator it(matrix, i); it; ++it)
            it.valueRef() = 0;

    for (uint64_t i = 0; i < _netgrid->_cellsN; i++) {
        matrix.coeffRef(i, i) = _local->_alphas[i];
        freeVector[i] = _local->_alphas[i] * _sats[iPrev][i];
    }

    for (auto &nonDirichCell: findNonDirichCells(_boundGroupsDirich)) {

        auto &faces = _netgrid->_neighborsFaces[nonDirichCell];
        auto &normals = _netgrid->_normalsNeighborsFaces[nonDirichCell];
        for (int32_t i = 0; i < faces.size(); i++) {
            auto face = faces[i];
            auto normal = normals[i];
            for (auto &[cell, cellCoeff] : _matrixFacesCells[face])
                matrix.coeffRef(nonDirichCell, cell) += normal * cellCoeff;
        }

    }

    // std::cout << matrix << std::endl;

    // std::cout << Eigen::VectorXd::Ones(matrix.cols()) << std::endl;

}

void Equation::calcSatsImplicit() {

    //   BiCGSTAB biCGSTAB;
    //   biCGSTAB.compute(matrix);
    //   _sats[iCurr] = biCGSTAB.solveWithGuess(freeVector, _sats[iPrev]);

    SparseLU sparseLU;
    sparseLU.compute(matrix);
    _sats[iCurr] = sparseLU.solve(freeVector);

}

void Equation::printCourNumbers(std::map<uint32_t, double> &thrsVelocities,
                                const double &timeStep) {

    std::vector<double> values;

    for (auto &[throat, velocity]: thrsVelocities)
        values.push_back(fabs(velocity) * timeStep / _netgrid->_throatsDLs[throat]);

    auto &max = *max_element(values.begin(), values.end());
    auto &min = *min_element(values.begin(), values.end());
    auto mean = accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.);
    double stdev = std::sqrt(sq_sum / values.size() - mean * mean);


    std::cout << "min: " << min << "; max: " << max << "; mean: " << mean << "; stdev: " << stdev;
    std::cout << std::endl;
    std::cout << fabs(thrsVelocities[0]) * timeStep / _netgrid->_throatsDLs[0];

}

void Equation::calcThroatsAvSats() {

    for (auto &[throat, cells] : _netgrid->_throatsCells) {
        double cum_sat = 0;
        for (auto &cell: cells)
            cum_sat += _sats[iCurr][cell];
        _throatsAvSats[throat] = cum_sat / cells.size();
    }

}

void Equation::calcThroatsSatsGrads() {

    for (auto &[throat, faces]: _netgrid->_throatsFaces) {
        double cum_sats_grad = 0;
        for (int i = 1; i < faces.size() - 1; i++) {
            auto &face = faces[i];
            for (int j = 0; j < _netgrid->_neighborsCells[face].size(); j++) {
                auto &cell = _netgrid->_neighborsCells[face][j];
                auto &normal = _netgrid->_normalsNeighborsCells[face][j];
                cum_sats_grad += normal * _sats[iCurr][cell];
            }
        }
        _throatsSatsGrads[throat] = cum_sats_grad;
    }

}

void Equation::cfdProcedureOneStep(std::map<uint32_t, double> &thrsVelocities,
                                   const double &timeStep) {

    std::swap(iCurr, iPrev);

    _convective->calcBetas(thrsVelocities);
    _local->calcAlphas(timeStep);

    processFaces(_netgrid->_typesFaces.at("nonbound"));
    for (auto &faces: _boundGroupsNewman) {
        processFaces(_netgrid->_typesFaces.at(faces));
    }

    fillMatrix();
    processDirichCells(_boundGroupsDirich, _satsBoundDirich);
    calcSatsImplicit();

}

void Equation::cfdProcedure(std::map<uint32_t, double> &thrsVelocities) {


    _sats.emplace_back(_netgrid->_cellsArrays.at("sats_curr"));
    _sats.emplace_back(_netgrid->_cellsArrays.at("sats_prev"));

    _local->calcTimeSteps();

    Eigen::Map<Eigen::VectorXd> satsInit(new double[dim], dim);
    satsInit = _sats[iCurr];
    _satsTime.push_back(satsInit);
    _time.push_back(0);

    std::vector<double> courNumber;
    double timeCurr = 0;
    auto timePeriod = std::get<double>(_props->_params["time_period"]);
    for (auto &timeStep : _local->_timeSteps) {

        timeCurr += timeStep;
        cfdProcedureOneStep(thrsVelocities, timeStep);

        Eigen::Map<Eigen::VectorXd> satsCurr(new double[dim], dim);
        satsCurr = _sats[iCurr];
        _satsTime.push_back(satsCurr);
        _time.push_back(timeCurr);

        printCourNumbers(thrsVelocities, timeStep);
        std::cout << " time: " << round(timeCurr / timePeriod * 1000) * 0.1 << "%."
                  << std::endl;

    }

}

std::vector<Eigen::Ref<Eigen::VectorXd>> Equation::getSats() {
    return Eigen::vectorGetter<Eigen::VectorXd>(_sats);
}

void Equation::setSats(std::vector<Eigen::Ref<Eigen::VectorXd>> &sats) {
    Eigen::vectorSetter<Eigen::VectorXd>(sats, _sats);
}

std::vector<Eigen::Ref<Eigen::VectorXd>> Equation::getSatsTime() {
    return Eigen::vectorGetter<Eigen::VectorXd>(_satsTime);
}

void Equation::setSatsTime(
        std::vector<Eigen::Ref<Eigen::VectorXd>> &satsTime) {
    Eigen::vectorSetter<Eigen::VectorXd>(satsTime, _satsTime);
}

Eigen::Ref<Eigen::VectorXd> Equation::getSatsIni() {
    return _satsIni;
}

void Equation::setSatsIni(Eigen::Ref<Eigen::VectorXd> satsIni) {
    if (_satsIni.data() != satsIni.data())
        delete _satsIni.data();
    new(&_satsIni) Eigen::Map<Eigen::VectorXd>(satsIni.data(),
                                               satsIni.size());
}

Eigen::Ref<Eigen::VectorXd> Equation::getThroatsAvSats() {
    return _throatsAvSats;
}

void Equation::setThroatsAvSats(Eigen::Ref<Eigen::VectorXd> throatsAvSats) {
    if (_throatsAvSats.data() != throatsAvSats.data())
        delete _throatsAvSats.data();
    new(&_throatsAvSats) Eigen::Map<Eigen::VectorXd>(throatsAvSats.data(),
                                                     throatsAvSats.size());
}

Eigen::Ref<Eigen::VectorXd> Equation::getThroatsSatsGrads() {
    return _throatsSatsGrads;
}

void Equation::setThroatsSatsGrads(Eigen::Ref<Eigen::VectorXd> throatsSatsGrads) {
    if (_throatsSatsGrads.data() != throatsSatsGrads.data())
        delete _throatsSatsGrads.data();
    new(&_throatsSatsGrads) Eigen::Map<Eigen::VectorXd>(throatsSatsGrads.data(),
                                                        throatsSatsGrads.size());
}