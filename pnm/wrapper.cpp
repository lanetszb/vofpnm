#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <string>

#include "Pnm.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(pnm_bind, m) {
    // Wrapper for PNM
    py::class_<Pnm>(m, "Pnm")
            .def(py::init<const std::map<std::string,
                    std::variant<int, double, std::string>> &,
                    std::shared_ptr<Netgrid>>(), "params_pnm"_a, "netgrid"_a)

            .def("calc_conductances", &Pnm::calcConductances,
                 "densities"_a,
                 "viscosities"_a)
            .def("process_nonbound_pores", &Pnm::processNonboundPores)
            .def("process_newman_pores", &Pnm::processNewmanPores,
                 "pores_flows"_a)
            .def("process_dirich_pores", &Pnm::processDirichPores,
                 "pores_pressures"_a)
            .def("fill_matrix", &Pnm::fillMatrix,
                 "pores_flows"_a,
                 "pores_pressures"_a,
                 "capillary_pressures"_a)
            .def("print_params_pnm", &Pnm::printParamsPnm)
            .def("cfd_procedure", &Pnm::cfdProcedure,
                 "densities"_a, "viscosities"_a, "capillary_pressures"_a,
                 "poresFlows"_a, "poresPressures"_a)
            .def("calc_throats_flow_rates", &Pnm::calcThroatsMassFlows,
                 "capillary_pressures"_a)
            .def("calc_pores_flow_rates", &Pnm::calcPoresFlowRates)
            .def("calc_total_flow_rate", &Pnm::calcTotalFlowRate)

            .def_readwrite("params_pnm", &Pnm::_paramsPnm)
            .def_readwrite("dim", &Pnm::_dim)
            .def_readwrite("conductances", &Pnm::_conductances)
            .def_readwrite("matrix_coeffs", &Pnm::_matrixCoeffs)
            .def_readwrite("free_coeffs", &Pnm::_freeCoeffs)
            .def_readwrite("throats_flow_rates", &Pnm::_throatsMassFlows)
            .def_readwrite("pores_flow_rates", &Pnm::_poresFlowRates)
            .def_readwrite("total_flow_rate", &Pnm::_totFlowRate)

            .def_property("pressures", &Pnm::getPressures, &Pnm::setPressures);

}



