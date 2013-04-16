#include <iostream>
#include <fstream>
#include "logistic_classifier.h"
#include <time.h>

using namespace std;
using namespace Eigen;



vector<int> Logistic_Reg::Predict(SpMat& mat)
{
	vector<int> label;
	for(int i=0;i<mat.rows();i++)
	{
		SparseVector<double> x = mat.row(i);
		if(this->Prediction(beta,x,1) > this->Prediction(beta,x,-1))
		{
			label.push_back(1);
		}
		else
		{
			label.push_back(-1);
		}
	}

	return label;
}
void Logistic_Reg::WriteModel()
{
	ofstream modelfile("model.txt");
	for (int i=0;i<beta.rows();i++)
	{
		modelfile<< beta.coeff(i,0)<<endl;
	}
}
void Logistic_Reg::Fit(SpMat& X,vector<int>& Y)
{
    this->X=X;
    this->Y=Y;

    //init the beta
    Init();

    this->Trust_region_newton();
}


void Logistic_Reg::Init()
{
    this->sample_size = this->X.rows();
    this->feature_size = this->X.cols();

    srand(time(NULL));
    Eigen::SparseVector<double>* tmp =new Eigen::SparseVector<double>(feature_size,1);
    for(int i=0;i<feature_size/1000;i++)
    {
        int random_idx =rand()% feature_size;
        (*tmp).coeffRef(random_idx) = rand()/(double)RAND_MAX;
    }
    beta=*tmp;

    this->prediction_value_cache.reserve(this->sample_size);
    for(int i=0;i<sample_size;i++)
    {
        SparseVector<double> x= X.row(i);
        double p= Prediction(beta,x,Y[i]);
        this->prediction_value_cache.push_back(p);
    }
}

double Logistic_Reg::Prediction(SparseVector<double>& beta,SparseVector<double>& x, int y)
{
    return 1.0/(1.0+exp(beta.dot(x)*(-y)));
}

SparseVector<double> Logistic_Reg::Gradient(SparseVector<double>& beta)
{
    Eigen::SparseVector<double> gradient(feature_size,1);
    for (int i=0;i<sample_size;i++)
    {
        double scale= (this->prediction_value_cache[i]-1)*(Y[i]);
        SparseVector<double> x = X.row(i);
        gradient+=x*scale;
    }
    return gradient;
}

void Logistic_Reg::Trust_region_newton()
{
    double eta_0=1e-4;
    double eta_1=0.25;
    double eta_2=0.75;
    double sigma_1=0.25;
    double sigma_2=0.5;
    double sigma_3=4.0;

    SparseVector<double> g= Gradient(beta);
    cout<< "get the gradient with the default beta"<<endl;

    double delta=g.norm();

    Hessian hessian;
    hessian.X_t = new SpMat(X.transpose());
    hessian.X = &X;
    int iteration=1;

    int it = config->iteration;
    while(iteration<it)
    {
	double beta_norm = beta.norm();
        cout<<"iteration: "<<iteration<<"  beta norm: "<<beta_norm<<endl;
        if (beta_norm<1e-2)
        {
            cout<<"converged!"<<endl;
            break;
        }

        //step 1: find an approximate solution s of the sub-problem
        g=this->Gradient(beta);
        this->Compute_Hessian(beta,hessian);
        SparseVector<double> s= this->Conjugate_gradient(g,hessian,delta);

        double snorm=s.norm();
        if(iteration==1)
        {
            delta=min(delta,snorm);
        }

        //step 2: compute rho via (8)
        //double rho=this->Rho(beta,g,hessian,s);
        double f = this->Function_value(beta);
        cout<<"current function value: "<<f<<endl;
        SparseVector<double> beta_s(beta+s);
        double fnew = this->Function_value(beta_s);
        double rho= (fnew-f)/this->Quadratic_value(g,hessian,s);

        double gs=g.dot(s);
        double alpha=0;

        if(fnew-f-gs<=0)
        {
            alpha=sigma_3;
        }
        else
        {
            alpha=max(sigma_1,-0.5*(gs/(fnew-f-gs)));
        }
        //step 3: update beta according to (9)
        if(rho>eta_0)
        {
            this->Update_beta(s);
            cout<<"updated beta !!"<<endl;
        }

        //step 4: obtain delta according to (10)
        if(rho<eta_0)
        {
            delta=min(max(alpha,sigma_1)*snorm, sigma_2*delta);
        }
        else if(rho<eta_1)
        {
            delta=max(sigma_1*delta,min(alpha*snorm,sigma_2*delta));
        }
        else if(rho<eta_2)
        {
            delta=max(sigma_1*delta,min(alpha*snorm,sigma_3*delta));
        }
        else
        {
            delta=max(delta,min(alpha*snorm,sigma_3*delta));
        }

        iteration++;
    }

}


void Logistic_Reg::Hessian_D(SparseVector<double>& beta,SpMat& D)
{
    vector<T> triplet_list;
    triplet_list.reserve(sample_size);
    for(int i=0;i<sample_size;i++)
    {
        double d= this->prediction_value_cache[i];
        triplet_list.push_back(T(i,i,d*(1-d)));
    }
    D.resize(sample_size,sample_size);
    D.setFromTriplets(triplet_list.begin(),triplet_list.end());
}

void Logistic_Reg::Compute_Hessian(SparseVector<double>& beta,Hessian& hessian)
{
    SpMat* D = new SpMat();
    this->Hessian_D(beta,*D);
    hessian.D = D;
}

double Logistic_Reg::Rho(SparseVector<double>& beta,SparseVector<double>& g, Hessian& hessian, SparseVector<double>& s)
{
    double current_function_value = this->Function_value(beta);
    cout<<"current function value: "<<current_function_value<<endl;
    SparseVector<double> beta_s(beta+s);
    double next_function_value = this->Function_value(beta_s);
    return (next_function_value-current_function_value)/this->Quadratic_value(g,hessian,s);
}

double Logistic_Reg::Function_value(SparseVector<double>& beta)
{
    double fun_value=0;
    for(int i=0;i<sample_size;i++)
    {
        SparseVector<double> x = X.row(i);
        double p=(beta.dot(x))*(-Y[i]);
        fun_value+= log(1.0+exp(p));
    }
    return fun_value;
}

double Logistic_Reg::Quadratic_value(SparseVector<double>& g,Hessian& hessian,SparseVector<double>& s)
{
    double g_s = g.dot(s);
    double s_hessian_s = s.dot(this->Matrix_vector_multi(hessian,s));
    return g_s+0.5*s_hessian_s;
}

SparseVector<double> Logistic_Reg::Matrix_vector_multi(Hessian& hessian,SparseVector<double>& v)
{
    return ((*(hessian.X_t))*((*(hessian.D))*((*(hessian.X))*v)));
}

SparseVector<double> Logistic_Reg::Conjugate_gradient(SparseVector<double>&g,Hessian& hessian,double delta)
{
    //step 1: init some parameters;
    double epsilon=0.1;
    SparseVector<double> s(feature_size,1);
    SparseVector<double> r= -1.0*g;
    SparseVector<double> d=r;
    SparseVector<double> news(feature_size,1);

    //step 2: inner iteration;
    int it_count=0;
    while(true)
    {
        if (r.norm()<= epsilon*g.norm())
        {
	   
	    cout<<"conjugate gradient iteration: "<<it_count<<endl;
            return s;
        }
        else
        {
            SparseVector<double> hessain_d=this->Matrix_vector_multi(hessian,d);
            double r_square_norm=r.squaredNorm();
            double a = r_square_norm/ d.dot(hessain_d);

            news = s + a*d;
            if(news.norm()>=delta)
            {
                //compute tau;
                double tau = this->Tau(s,d,delta);

                // output s=s+tau*d and stop
                SparseVector<double> result=(s+tau*d);
		cout<<"conjugate gradient iteration: "<<it_count<<endl;
                return result;
            }

            r = r-a*hessain_d;
            double newr_square_norm=r.squaredNorm();
            double b = newr_square_norm/r_square_norm;
            d=r+b*d;
            s=news;
        }

	it_count++;

    }
}

double Logistic_Reg::Tau(SparseVector<double>& s, SparseVector<double>& d, double delta)
{
    //trop.cpp
    double tau=0;

    double sd=s.dot(d);
    double ss=s.squaredNorm();
    double dd=d.squaredNorm();
    double dsq=delta*delta;
    double rad=sqrt(sd*sd+dd*(dsq-ss));
    if (sd>=0)
        tau=(dsq-ss)/(sd+rad);
    else
    {
        tau=(rad-sd)/dd;
    }
    return tau;
}


void Logistic_Reg::Update_beta(SparseVector<double>& s)
{
    this->beta+=s;

    if(config->type==Reg_type::L1)
    {
    	this->Soft_thresholding(beta);
    }

    //update the prediction cache
    for(int i=0;i<sample_size;i++)
    {
        SparseVector<double> x= X.row(i);
        double p= Prediction(beta,x,Y[i]);
        this->prediction_value_cache[i]=p;
    }
}



void Logistic_Reg::Soft_thresholding(SparseVector<double>& beta)
{

    //for L1 regularization
    const int t=1.0;
    for(int i=0;i<beta.rows();i++)
    {
	    double coeff = beta.coeff(i,0);
	    if (coeff>=t)
	    {
		    beta.coeffRef(i)= coeff-t; 
	    }
	    else if (coeff>-t && coeff<t)
	    {
		    beta.coeffRef(i)=0;
	    }
	    else if(coeff<=-t)
	    {
		    beta.coeffRef(i)=coeff+t;
	    }
    }

}
