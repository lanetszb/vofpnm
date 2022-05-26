#ifndef PNM_H
#define PNM_H

#include <vector>
#include <string>
#include <variant>
#include <map>

#include <Eigen/Sparse>
#include <Netgrid.h>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> Matrix;
typedef Matrix::InnerIterator MatrixIterator;

class Pnm {

public:

    explicit Pnm(const std::map<std::string, std::variant<int, double, std::string>> &paramsPnm,
                 std::shared_ptr<Netgrid> netgrid);

    virtual ~Pnm() = default;

    void printParamsPnm();

    Eigen::Ref<Eigen::VectorXd> getPressures();

    void setPressures(Eigen::Ref<Eigen::VectorXd> pressures);

    void calcConductances(const std::vector<double> &densities,
                          const std::vector<double> &viscosities);

    void processThroats();

    void processPores();

    void processNewmanPores(std::map<uint32_t, double> &poresFlows);

    void processDirichPores(std::map<uint32_t, double> &poresPressures);

    void fillMatrix(std::map<uint32_t, double> &poresFlows,
                    std::map<uint32_t, double> &poresPressures,
                    const std::vector<double> &capillaryPressures);

    void calculatePress();

    void cfdProcedure(const std::vector<double> &densities,
                      const std::vector<double> &viscosities,
                      const std::vector<double> &capillaryPressures,
                      std::map<uint32_t, double> &poresFlows,
                      std::map<uint32_t, double> &poresPressures);

    void calcThroatsVolFlows(const std::vector<double> &capillaryPressures);

    void calcPoresFlowRates();

    void calcTotalFlowRate(const std::set<uint32_t> &pores);

    std::map<std::string, std::variant<int, double, std::string>> _paramsPnm;
    std::shared_ptr<Netgrid> _netgrid;

    int _dim;
    Matrix _matrix;
    Eigen::Map<Eigen::VectorXd> _freeVector;
    Eigen::Map<Eigen::VectorXd> _pressures;

    std::map<uint32_t, double> _conductances;
    std::map<uint32_t, double> _freeCoeffs;
    std::map<uint32_t, std::map<uint32_t, double>> _matrixCoeffs;

    std::map<uint32_t, double> _throatsVolFlows;
    std::map<uint32_t, double> _poresFlowRates;

    double _totFlowRate;

};


#endif
