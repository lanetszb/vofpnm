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

#ifndef EQUATION_H
#define EQUATION_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "math/Props.h"
#include "math/Local.h"
#include "math/Convective.h"
#include <netgrid/Netgrid.h>

typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> Matrix;
typedef Matrix::InnerIterator MatrixIterator;
typedef Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> BiCGSTAB;

class Equation {

public:

    explicit Equation(std::shared_ptr<Props> props,
                      std::shared_ptr<Netgrid> netgrid,
                      std::shared_ptr<Local> local,
                      std::shared_ptr<Convective> convective);

    virtual ~Equation() {}

    std::vector<Eigen::Ref<Eigen::VectorXd>> getConcs();

    void setConcs(std::vector<Eigen::Ref<Eigen::VectorXd>> &concs);

    std::vector<Eigen::Ref<Eigen::VectorXd>> getConcsTime();

    void setConcsTime(std::vector<Eigen::Ref<Eigen::VectorXd>> &concsTime);

    Eigen::Ref<Eigen::VectorXd> getConcsIni();

    void setConcsIni(Eigen::Ref<Eigen::VectorXd> concsIni);

    void processNewmanFaces(const double &flowNewman,
                            const std::set<uint32_t> &faces);

    void processDirichCells(std::vector<std::string> &boundGroups,
                            std::map<std::string, double> &concsBound);

    std::set<uint32_t> groupCellsByTypes(const std::vector<std::string> &groups);

    std::set<uint32_t> findNonDirichCells(const std::vector<std::string> &boundGroupsDirich);

    void processNonBoundFaces(const std::set<uint32_t> &faces);

    void fillMatrix();

    // void calcConcsIni();

     void calcConcsImplicit();

    void cfdProcedureOneStep(const double &timeStep);

    void cfdProcedure();

    // double calcFacesFlowRate(Eigen::Ref<Eigen::VectorXui64> faces);

    std::shared_ptr<Props> _props;
    std::shared_ptr<Netgrid> _netgrid;
    std::shared_ptr<Local> _local;
    std::shared_ptr<Convective> _convective;

    int dim;
    int iCurr;
    int iPrev;

    std::vector<std::string> _boundGroupsDirich;
    std::vector<std::string> _boundGroupsNewman;
    std::map<std::string, double> _concsBoundDirich;

    std::vector<Eigen::Map<Eigen::VectorXd>> _concs;
    std::vector<Eigen::Map<Eigen::VectorXd>> _concsTime;
    Eigen::Map<Eigen::VectorXd> _concsIni;

    std::map<uint32_t, std::map<uint32_t, double>> _matrixFacesCells;
    std::map<uint32_t, std::map<uint32_t, double>> _freeFacesCells;

    Matrix matrix;
    Eigen::Map<Eigen::VectorXd> freeVector;


};

#endif // EQUATION_H
