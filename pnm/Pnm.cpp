#include "Pnm.h"

#include <iostream>

#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> Matrix;
typedef Matrix::InnerIterator MatrixIterator;
typedef Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> BiCGSTAB;
typedef Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>>
        LeastSqCG;
typedef Eigen::SparseLU<Eigen::SparseMatrix<double>> SparseLU;

Pnm::Pnm(const std::map<std::string, std::variant<int, double, std::string>> &paramsPnm,
         std::shared_ptr<Netgrid> netgrid) :
        _paramsPnm(paramsPnm),
        _netgrid(netgrid),
        _dim(netgrid->_poresPores.size()),
        _matrix(_dim, _dim),
        _freeVector(new double[_dim], _dim),
        _pressures(new double[_dim], _dim) {}

void Pnm::printParamsPnm() {

    for (auto const &[name, param] : _paramsPnm) {
        std::cout << name << ": ";
        if (std::get_if<int>(&param))
            std::cout << std::get<int>(param) << std::endl;
        else if (std::get_if<double>(&param))
            std::cout << std::get<double>(param) << std::endl;
        else if (std::get_if<std::string>(&param))
            std::cout << std::get<std::string>(param) << std::endl;
    }

}

void Pnm::calcConductances(const std::vector<double> &densities,
                           const std::vector<double> &viscosities) {

    for (auto &[throat, pores] : _netgrid->_throatsPores) {

        auto &height = _netgrid->_throatsDepths[throat];
        auto &width = _netgrid->_throatsWidths[throat];
        auto &length = _netgrid->_throatsLs[throat];
        auto &density = densities[throat];
        auto &viscosity = viscosities[throat];

        double resistance = 12. * viscosity * length / height / height / height / width;

        _conductances[throat] = density / resistance;
    }

}

void Pnm::processNonboundPores() {

    for (auto &pore: _netgrid->_inletPores)
        _freeCoeffs[pore] = 0;
    for (auto &pore: _netgrid->_outletPores)
        _freeCoeffs[pore] = 0;
    for (auto &pore: _netgrid->_nonboundPores)
        _freeCoeffs[pore] = 0;

    for (auto &pore: _netgrid->_nonboundPores) {
        auto &poresPores = _netgrid->_poresPores[pore];
        auto &poresThroats = _netgrid->_poresThroats[pore];
        auto &normalsPoresThroats = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < poresPores.size(); i++)
            _matrixCoeffs[pore][poresPores[i]] =
                    normalsPoresThroats[i] * _conductances[poresThroats[i]];
    }

}

void Pnm::processNewmanPores(std::map<uint32_t, double> &poresFlows) {

    for (auto &[pore, flow]: poresFlows) {
        _freeCoeffs[pore] = flow;
        auto poresPores = _netgrid->_poresPores[pore];
        auto poresThroats = _netgrid->_poresThroats[pore];
        auto &normalsPoresThroats = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < poresPores.size(); i++)
            _matrixCoeffs[pore][poresPores[i]] =
                    normalsPoresThroats[i] * _conductances[poresThroats[i]];
    }

}

void Pnm::processDirichPores(std::map<uint32_t, double> &poresPressures) {

    for (auto &[pore, pressure]: poresPressures) {
        _matrixCoeffs[pore][pore] = 1.;
        _freeCoeffs[pore] = pressure;
    }

}

void Pnm::fillMatrix(std::map<uint32_t, double> &poresFlows,
                     std::map<uint32_t, double> &poresPressures) {

    for (int i = 0; i < _dim; ++i)
        for (MatrixIterator it(_matrix, i); it; ++it)
            it.valueRef() = 0;

    for (auto &currPore: _netgrid->_nonboundPores) {
        _freeVector[currPore] = _freeCoeffs[currPore];
        for (auto &[neigbPore, conductance]: _matrixCoeffs[currPore]) {
            _matrix.coeffRef(currPore, neigbPore) = conductance;
            _matrix.coeffRef(currPore, currPore) -= conductance;
        }
    }

    for (auto &[currPore, flow]: poresFlows) {
        _freeVector[currPore] = _freeCoeffs[currPore];
        for (auto &[neigbPore, conductance]: _matrixCoeffs[currPore]) {
            _matrix.coeffRef(currPore, neigbPore) = conductance;
            _matrix.coeffRef(currPore, currPore) -= conductance;
        }
    }

    for (auto &[currPore, pressure]: poresPressures) {
        _matrix.coeffRef(currPore, currPore) = _matrixCoeffs[currPore][currPore];
        _freeVector[currPore] = _freeCoeffs[currPore];
    }

    std::cout << _matrix << std::endl;
}

void Pnm::processCapillaryPressures(const std::vector<double> &capillaryPressures) {

    for (auto &pore: _netgrid->_nonboundPores)
        for (auto &throat: _netgrid->_poresThroats[pore])
            _freeCoeffs[pore] -= _netgrid->_normalsPoresThroats[pore][throat] *
                                 _conductances[throat] * capillaryPressures[throat];

    for (auto &pore: _netgrid->_inletPores)
        for (auto &throat: _netgrid->_poresThroats[pore])
            _freeCoeffs[pore] -= _netgrid->_normalsPoresThroats[pore][throat] *
                                 _conductances[throat] * capillaryPressures[throat];

    // ToDo:remove zatychka
    for (auto &currPore: _netgrid->_nonboundPores)
        _freeVector[currPore] = _freeCoeffs[currPore];
    for (auto &currPore: _netgrid->_inletPores)
        _freeVector[currPore] = _freeCoeffs[currPore];

}

void Pnm::calculatePress() {

    auto &itAccuracy = std::get<double>(_paramsPnm["it_accuracy"]);
    auto &solverMethod = std::get<std::string>(_paramsPnm["solver_method"]);

    if (solverMethod == "sparseLU") {

        SparseLU sparseLU;
        sparseLU.compute(_matrix);
        _pressures = sparseLU.solve(_freeVector);

    } else if (solverMethod == "biCGSTAB") {

        BiCGSTAB biCGSTAB;
        biCGSTAB.compute(_matrix);
        biCGSTAB.setTolerance(itAccuracy);
        _pressures = biCGSTAB.solveWithGuess(_freeVector, _pressures);

    } else if (solverMethod == "leastSqCG") {

        LeastSqCG leastSqCG;
        leastSqCG.compute(_matrix);
        leastSqCG.setTolerance(itAccuracy);
        _pressures = leastSqCG.solveWithGuess(_freeVector, _pressures);

    }

}

void Pnm::cfdProcedure(const std::vector<double> &densities,
                       const std::vector<double> &viscosities,
                       const std::vector<double> &capillaryPressures,
                       std::map<uint32_t, double> &poresFlows,
                       std::map<uint32_t, double> &poresPressures) {

    calcConductances(densities, viscosities);
    processNonboundPores();
    processNewmanPores(poresFlows);
    processDirichPores(poresPressures);
    fillMatrix(poresFlows, poresPressures);
    processCapillaryPressures(capillaryPressures);

    calculatePress();

}

void Pnm::calcThroatsFlowRates(const std::vector<double> &capillaryPressures) {

    for (auto &[throat, conductance] : _conductances) {
        auto &pore0 = _netgrid->_throatsPores[throat].front();
        auto &pore1 = _netgrid->_throatsPores[throat].back();
        _throatsFlowRates[throat] = conductance * (_pressures[pore0] - _pressures[pore1] +
                                                   capillaryPressures[throat]);
    }

}

void Pnm::calcPoresFlowRates() {

    for (auto &pore: _netgrid->_nonboundPores) {
        auto &poresThroats = _netgrid->_poresThroats[pore];
        auto &normals = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < poresThroats.size(); i++)
            _poresFlowRates[pore] += normals[i] * _throatsFlowRates[poresThroats[i]];
    }

    for (auto &pore: _netgrid->_outletPores) {
        auto &poresThroats = _netgrid->_poresThroats[pore];
        auto &normals = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < poresThroats.size(); i++)
            _poresFlowRates[pore] += normals[i] * _throatsFlowRates[poresThroats[i]];
    }

    for (auto &pore: _netgrid->_inletPores) {
        auto &poresThroats = _netgrid->_poresThroats[pore];
        auto &normals = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < poresThroats.size(); i++)
            _poresFlowRates[pore] += -normals[i] * _throatsFlowRates[poresThroats[i]];
    }

}

void Pnm::calcTotalFlowRate(const std::set<uint32_t> &pores) {

    _totFlowRate = 0;

    for (auto &pore: pores)
        _totFlowRate += _poresFlowRates[pore];
}

void Pnm::setPressures(Eigen::Ref<Eigen::VectorXd> pressures) {
    if (_pressures.data() != pressures.data())
        delete _pressures.data();
    new(&_pressures) Eigen::Map<Eigen::VectorXd>(pressures.data(),
                                                 pressures.size());
}

Eigen::Ref<Eigen::VectorXd> Pnm::getPressures() {
    return _pressures;
}