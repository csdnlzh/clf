#ifndef LOGISTIC_H_INCLUDED
#define LOGISTIC_H_INCLUDED
#include <vector>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;

typedef Eigen::SparseMatrix<double,Eigen::RowMajorBit> SpMat;
typedef Eigen::Triplet<double> T;


class Hessian
{
public:

    Hessian(){X_t=NULL; X=NULL; D=NULL;};
    SpMat* X_t;
    SpMat* X;
    SpMat* D;
};
#endif // LOGISTIC_H_INCLUDED
