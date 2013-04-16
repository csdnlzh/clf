#ifndef LOGISTIC_CLASSIFIER_H_INCLUDED
#define LOGISTIC_CLASSIFIER_H_INCLUDED

#include "logistic.h"
#include "config.h"

using namespace Eigen;
class Logistic_Reg
{
public:
    Logistic_Reg()
    {
	    config = Config::GetInstance();
    }

    void Fit(SpMat& X, vector<int>& Y);

    double Prediction(SparseVector<double>& beta,SparseVector<double>& x, int y);
    SparseVector<double> Gradient(SparseVector<double>& beta);
    void Trust_region_newton();
    SparseVector<double> Conjugate_gradient(SparseVector<double>& g,Hessian& hessian,double delta);
    void WriteModel();
    vector<int> Predict(SpMat& X);
    void Soft_thresholding(SparseVector<double>& beta);

    Eigen::SparseVector<double> beta;
    vector<double> prediction_value_cache;
    int sample_size;
    int feature_size;
    SpMat X;
    vector<int> Y;
    Config* config;
private:
    void Init();
    void Hessian_D(SparseVector<double>& beta,SpMat& D);
    void Compute_Hessian(SparseVector<double>& beta, Hessian& hessian);
    double Rho(SparseVector<double>& beta,SparseVector<double>& g, Hessian& hessian,SparseVector<double>& s);
    double Function_value(SparseVector<double>& beta);
    double Quadratic_value(SparseVector<double>& g, Hessian& hessian, SparseVector<double>& s);
    SparseVector<double> Matrix_vector_multi(Hessian& hessian,SparseVector<double>& v);

    double Tau(SparseVector<double>&s,SparseVector<double>&d,double delta);

    void Update_beta(SparseVector<double>& s);
};

#endif // LOGISTIC_CLASSIFIER_H_INCLUDED
