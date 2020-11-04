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


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "math/Props.h"
#include "math/Boundary.h"
#include "math/Local.h"
#include "math/Convective.h"
#include "math/funcs.h"
#include "Equation.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(vof_bind, m) {

    m.def("calc_a_func", calcAFunc);
    m.def("calc_b_func", calcBFunc, "velocity"_a);

    py::class_<Props, std::shared_ptr<Props>>(m, "Props")
            .def(py::init<const std::map<std::string, std::variant<int, double, std::string>> &>(),
                 "params"_a)

            .def_readwrite("params", &Props::_params)
            .def("print_params", &Props::printParams);

    py::class_<Boundary, std::shared_ptr<Boundary>>(m, "Boundary")
            .def(py::init<std::shared_ptr<Props>, std::shared_ptr<Netgrid>>(),
                 "props"_a, "netgrid"_a)

            .def("shift_boundary_faces", &Boundary::shiftBoundaryFaces,
                 "faces"_a);

    py::class_<Local, std::shared_ptr<Local>>(m, "Local")
            .def(py::init<std::shared_ptr<Props>, std::shared_ptr<Netgrid>>(),
                 "props"_a, "netgrid"_a)

            .def("calc_time_steps", &Local::calcTimeSteps)
            .def("calc_alphas", &Local::calcAlphas, "time_step"_a)
            .def_readwrite("time_steps", &Local::_timeSteps)
            .def_readwrite("alphas", &Local::_alphas);


    py::class_<Convective, std::shared_ptr<Convective>>(m, "Convective")
            .def(py::init<std::shared_ptr<Props>, std::shared_ptr<Netgrid>>(),
                 "props"_a, "netgrid"_a)

            .def("weigh_conc", &Convective::weighing, "method"_a,
                 "conc_first"_a, "conc_second"_a)
            .def("calc_betas", &Convective::calcBetas,
                 "velocities"_a)
            .def_readwrite("betas", &Convective::_betas);
    py::class_<Equation, std::shared_ptr<Equation>>(m, "Equation")
            .def(py::init<std::shared_ptr<Props>, std::shared_ptr<Netgrid>,
                         std::shared_ptr<Local>, std::shared_ptr<Convective>>(),
                 "props"_a, "netgrid"_a, "local"_a, "convective"_a)

            .def("fill_matrix", &Equation::fillMatrix)
            .def("calc_concs_implicit", &Equation::calcSatsImplicit)
            .def("cfd_procedure_one_step", &Equation::cfdProcedureOneStep,
                 "thrs_velocities"_a, "timeStep"_a)
            .def("cfd_procedure", &Equation::cfdProcedure, "thrs_velocities"_a)
                    // .def("calc_faces_flow_rate", &Equation::calcFacesFlowRate,
                    //      "faces"_a)

            .def("group_cells_by_types", &Equation::groupCellsByTypes,
                 "cells_groups"_a)
            .def("find_non_dirich_cells", &Equation::findNonDirichCells,
                 "cells_groups"_a)
            .def_readwrite("free_faces_cells", &Equation::_freeFacesCells)
            .def_readwrite("matrix_faces_cells", &Equation::_matrixFacesCells)
            .def_readwrite("dim", &Equation::dim)
            .def_readwrite("i_curr", &Equation::iCurr)
            .def_readwrite("i_prev", &Equation::iPrev)
            .def_readwrite("bound_groups_dirich", &Equation::_boundGroupsDirich)
            .def_readwrite("concs_bound_dirich", &Equation::_satsBoundDirich)
            .def_property("concs_ini",
                          &Equation::getSatsIni, &Equation::setSatsIni)
            .def_property("concs",
                          &Equation::getSats, &Equation::setSats)
            .def_property("concs_time",
                          &Equation::getSatsTime, &Equation::setSatsTime);


}