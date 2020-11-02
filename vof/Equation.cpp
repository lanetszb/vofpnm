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
        _concsIni(new double[dim], dim),
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


void Equation::processNewmanFaces(const double &flowNewman,
                                  const std::set<uint32_t> &faces) {

    for (auto &face : faces)
        for (auto &cell : _netgrid->_neighborsCells[face]) {
            _matrixFacesCells[face][cell] = 0;
            _freeFacesCells[face][cell] = flowNewman;
        }
}

void Equation::processNonBoundFaces(const std::set<uint32_t> &faces) {
    // ToDo: think how to update the method, implement upwind
    for (auto &face : faces) {
        auto &cells = _netgrid->_neighborsCells[face];
        auto &normals = _netgrid->_normalsNeighborsCells[face];
        for (int i = 0; i < cells.size(); i++) {
            auto &cell = cells[i];
            auto &normal = normals[i];
            _matrixFacesCells[face][cell] = normal * _convective->_betas[face];
            _freeFacesCells[face][cell] = 0;
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

std::set<uint32_t> Equation::findNonDirichCells(const std::vector<std::string> &boundGroupsDirich) {

    auto allCells = groupCellsByTypes({"inlet", "nonbound", "outlet"});
    auto dirichCells = groupCellsByTypes(boundGroupsDirich);

    std::set<uint32_t> nonDirichCells;
    std::set_difference(std::begin(allCells),
                        std::end(allCells),
                        std::begin(dirichCells), std::end(dirichCells),
                        std::inserter(nonDirichCells, nonDirichCells.begin()));

    return nonDirichCells;
}

void Equation::processDirichCells(std::vector<std::string> &boundGroups,
                                  std::map<std::string, double> &concsBound) {

    auto allBoundCells = groupCellsByTypes({"inlet", "outlet"});

    for (auto &bound : boundGroups) {
        auto &conc = concsBound[bound];
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
        freeVector[i] = _local->_alphas[i] * _concs[iPrev][i];
    }

    for (auto &nonDirichCell: findNonDirichCells(_boundGroupsDirich)) {

        auto &faces = _netgrid->_neighborsFaces[nonDirichCell];
        auto &normals = _netgrid->_normalsNeighborsFaces[nonDirichCell];
        for (int32_t i = 0; i < faces.size(); i++) {
            auto face = faces[i];
            auto normal = normals[i];
            for (auto &[cell, cellCoeff] : _matrixFacesCells[face])
                if (nonDirichCell != cell) {
                    matrix.coeffRef(nonDirichCell, cell) += normal * cellCoeff;
                    matrix.coeffRef(nonDirichCell, nonDirichCell) +=
                            normal * _matrixFacesCells[face][nonDirichCell];
                }
        }
    }

}

void Equation::calcConcsImplicit() {

    BiCGSTAB biCGSTAB;

    biCGSTAB.compute(matrix);

    _concs[iCurr] = biCGSTAB.solveWithGuess(freeVector, _concs[iPrev]);

}

void Equation::cfdProcedureOneStep(std::map<uint32_t, double> &thrsVelocities,
                                   const double &timeStep) {

    std::swap(iCurr, iPrev);

    _convective->calcBetas(thrsVelocities);
    _local->calcAlphas(timeStep);

    processNonBoundFaces(_netgrid->_typesFaces.at("nonbound"));
    fillMatrix();
    processDirichCells(_boundGroupsDirich, _concsBoundDirich);

    calcConcsImplicit();

}

void Equation::cfdProcedure(std::map<uint32_t, double> &thrsVelocities) {

    _concs.emplace_back(_netgrid->_cellsArrays.at("concs_array1"));
    _concs.emplace_back(_netgrid->_cellsArrays.at("concs_array2"));

    _local->calcTimeSteps();

    for (auto &timeStep : _local->_timeSteps) {
        cfdProcedureOneStep(thrsVelocities, timeStep);
        Eigen::Map<Eigen::VectorXd> concCurr(new double[dim], dim);
        concCurr = _concs[iCurr];
        _concsTime.push_back(concCurr);
    }
}

//
// double Equation::calcFacesFlowRate(Eigen::Ref<Eigen::VectorXui64> faces) {
//
//     auto &poro = std::get<double>(_props->_params["poro"]);
//
//     double totalFlowRate = 0;
//     for (uint64_t i = 0; i < faces.size(); i++) {
//         auto &face = faces[i];
//
//         auto &neighborsCells = _netgrid->_neighborsCells[face];
//         auto &conc_prev0 = _concs[iPrev](neighborsCells[0]);
//         auto &conc_prev1 = _concs[iPrev](neighborsCells[1]);
//
//         auto &normalsNeighborsCells = _netgrid->_normalsNeighborsCells[face];
//         auto &norm0 = normalsNeighborsCells[0];
//         auto &norm1 = normalsNeighborsCells[1];
//
//         auto diffusivity0 = _props->calcD(conc_prev0);
//         auto diffusivity1 = _props->calcD(conc_prev1);
//
//         // ToDo: axis not needed
//         auto &axis = _netgrid->_facesAxes[face];
//         // ToDo: consider more general case
//         auto diffusivity = _convective->weighing("meanAverage",
//                                                  diffusivity0, diffusivity1);
//
//         auto &conc_curr0 = _concs[iCurr](neighborsCells[0]);
//         auto &conc_curr1 = _concs[iCurr](neighborsCells[1]);
//         auto &dS = _netgrid->_facesSs[axis];
//         auto &dL = _netgrid->_spacing[axis];
//
//         totalFlowRate -=
//                 poro * diffusivity *
//                 (norm0 * conc_curr0 + norm1 * conc_curr1) * dS / dL;
//     }
//
//     return totalFlowRate;
// }
//
std::vector<Eigen::Ref<Eigen::VectorXd>> Equation::getConcs() {
    return Eigen::vectorGetter<Eigen::VectorXd>(_concs);
}

void Equation::setConcs(std::vector<Eigen::Ref<Eigen::VectorXd>> &concs) {
    Eigen::vectorSetter<Eigen::VectorXd>(concs, _concs);
}

std::vector<Eigen::Ref<Eigen::VectorXd>> Equation::getConcsTime() {
    return Eigen::vectorGetter<Eigen::VectorXd>(_concsTime);
}

void Equation::setConcsTime(
        std::vector<Eigen::Ref<Eigen::VectorXd>> &concsTime) {
    Eigen::vectorSetter<Eigen::VectorXd>(concsTime, _concsTime);
}

Eigen::Ref<Eigen::VectorXd> Equation::getConcsIni() {
    return _concsIni;
}

void Equation::setConcsIni(Eigen::Ref<Eigen::VectorXd> concsIni) {
    if (_concsIni.data() != concsIni.data())
        delete _concsIni.data();
    new(&_concsIni) Eigen::Map<Eigen::VectorXd>(concsIni.data(),
                                                concsIni.size());
}