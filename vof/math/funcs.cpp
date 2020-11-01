#include "funcs.h"

double calcAFunc(const double &conc, const double &poro) {

    return poro;
}

double
calcBFunc(const double &conc, const double &diffusivity, const double &poro) {
    return poro * diffusivity;
}

