#ifndef FUNCS_H
#define FUNCS_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>

double calcAFunc(const double &conc, const double &poro);

double
calcBFunc(const double &conc, const double &diffusion, const double &poro);


#endif // FUNCS_H
