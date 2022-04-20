#include "Pnm.h"

#include <iostream>

#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> Matrix;
typedef Matrix::InnerIterator MatrixIterator;
typedef Eigen::BiCGSTAB <Eigen::SparseMatrix<double>> BiCGSTAB;
typedef Eigen::LeastSquaresConjugateGradient <Eigen::SparseMatrix<double>>
        LeastSqCG;
typedef Eigen::SparseLU <Eigen::SparseMatrix<double>> SparseLU;

Pnm::Pnm(const std::map <std::string, std::variant<int, double, std::string>> &paramsPnm,
         std::shared_ptr <Netgrid> netgrid) :
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
        auto corrFactor = 1.;
        double resistance = corrFactor * 12. * viscosity * length / width / width / width / height;
        _conductances[throat] = 1. / resistance;
    }

}

void Pnm::processThroats() {
    for (auto&[throat, pores] : _netgrid->_throatsPores) {
        auto &normals = _netgrid->_normalsThroatsPores[throat];
        for (uint32_t i = 0; i < pores.size(); i++)
            _matrixCoeffs[throat][pores[i]] = normals[i] * _conductances[throat];
    }
}

void Pnm::processPores() {

    for (auto &pore: _netgrid->_inletPores)
        _freeCoeffs[pore] = 0;
    for (auto &pore: _netgrid->_outletPores)
        _freeCoeffs[pore] = 0;
    for (auto &pore: _netgrid->_deadendPores)
        _freeCoeffs[pore] = 0;
    for (auto &pore: _netgrid->_nonboundPores)
        _freeCoeffs[pore] = 0;
}

void Pnm::processNewmanPores(std::map<uint32_t, double> &poresFlows) {
    // ToDo: make sure it is ok with flow sign
    for (auto &[pore, flow]: poresFlows)
        _freeCoeffs[pore] = flow;
}

void Pnm::processDirichPores(std::map<uint32_t, double> &poresPressures) {
    for (auto &[pore, pressure]: poresPressures)
        _freeCoeffs[pore] = pressure;
}

void Pnm::fillMatrix(std::map<uint32_t, double> &poresFlows,
                     std::map<uint32_t, double> &poresPressures,
                     const std::vector<double> &capillaryPressures) {

    for (int i = 0; i < _dim; ++i)
        for (MatrixIterator it(_matrix, i); it; ++it)
            it.valueRef() = 0;

    std::set <uint32_t> groupedPores;
    groupedPores.insert(_netgrid->_nonboundPores.begin(), _netgrid->_nonboundPores.end());
    groupedPores.insert(_netgrid->_deadendPores.begin(), _netgrid->_deadendPores.end());

    for (auto &pore: groupedPores) {
        _freeVector[pore] = _freeCoeffs[pore];
        auto &throats = _netgrid->_poresThroats[pore];
        auto &normals = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < throats.size(); i++) {
            _freeVector[pore] += normals[i] * _conductances[throats[i]] *
                                 capillaryPressures[throats[i]];
            for (auto &poreCurr : _netgrid->_throatsPores[throats[i]])
                _matrix.coeffRef(pore, poreCurr) +=
                        normals[i] * _matrixCoeffs[throats[i]][poreCurr];
        }
    }

    for (auto &[pore, flow]: poresFlows) {
        _freeVector[pore] = _freeCoeffs[pore];
        auto &throats = _netgrid->_poresThroats[pore];
        auto &normals = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < throats.size(); i++) {
            _freeVector[pore] += normals[i] * _conductances[throats[i]] *
                                 capillaryPressures[throats[i]];
            for (auto &poreCurr : _netgrid->_throatsPores[throats[i]])
                _matrix.coeffRef(pore, poreCurr) +=
                        normals[i] * _matrixCoeffs[throats[i]][poreCurr];
        }
    }

    for (auto &[pore, pressure]: poresPressures) {
        _freeVector[pore] = _freeCoeffs[pore];
        _matrix.coeffRef(pore, pore) = 1;
    }

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
    processThroats();
    processPores();
    processNewmanPores(poresFlows);
    processDirichPores(poresPressures);
    fillMatrix(poresFlows, poresPressures, capillaryPressures);
    // std::cout << _matrix << std::endl;

    calculatePress();

}

void Pnm::calcThroatsVolFlows(const std::vector<double> &capillaryPressures) {

    for (auto &[throat, conductance] : _conductances) {
        auto &pores = _netgrid->_throatsPores[throat];
        auto &normals = _netgrid->_normalsThroatsPores[throat];
        _throatsVolFlows[throat] = conductance * capillaryPressures[throat];
        for (uint32_t i = 0; i < pores.size(); i++)
            _throatsVolFlows[throat] -= normals[i] * conductance * _pressures[pores[i]];
    }

}

void Pnm::calcPoresFlowRates() {

    std::set <uint32_t> groupedPores;
    groupedPores.insert(_netgrid->_nonboundPores.begin(), _netgrid->_nonboundPores.end());
    groupedPores.insert(_netgrid->_deadendPores.begin(), _netgrid->_deadendPores.end());

    for (auto &pore: groupedPores) {
        auto &poresThroats = _netgrid->_poresThroats[pore];
        auto &normals = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < poresThroats.size(); i++)
            _poresFlowRates[pore] += normals[i] * _throatsVolFlows[poresThroats[i]];
    }

    for (auto &pore: _netgrid->_outletPores) {
        auto &poresThroats = _netgrid->_poresThroats[pore];
        auto &normals = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < poresThroats.size(); i++)
            _poresFlowRates[pore] += normals[i] * _throatsVolFlows[poresThroats[i]];
    }

    for (auto &pore: _netgrid->_inletPores) {
        auto &poresThroats = _netgrid->_poresThroats[pore];
        auto &normals = _netgrid->_normalsPoresThroats[pore];
        for (uint32_t i = 0; i < poresThroats.size(); i++)
            _poresFlowRates[pore] += -normals[i] * _throatsVolFlows[poresThroats[i]];
    }

}

void Pnm::calcTotalFlowRate(const std::set <uint32_t> &pores) {

    _totFlowRate = 0;

    for (auto &pore: pores)
        _totFlowRate += _poresFlowRates[pore];
}

void Pnm::setPressures(Eigen::Ref <Eigen::VectorXd> pressures) {
    if (_pressures.data() != pressures.data())
        delete _pressures.data();
    new(&_pressures) Eigen::Map<Eigen::VectorXd>(pressures.data(),
                                                 pressures.size());
}

Eigen::Ref <Eigen::VectorXd> Pnm::getPressures() {
    return _pressures;
}