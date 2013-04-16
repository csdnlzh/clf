#include <time.h>
#include <iostream>
#include "logistic.h"
#include "io.h"
#include "logistic_classifier.h"
#include "config.h"
using namespace std;
using namespace Eigen;


int main(int argc,char ** argv)
{
    time_t start_time(time(NULL));
    string train_filename=argv[1];
    int it =atoi(argv[2]);
    string reg_type(argv[3]);

    Config* config= Config::GetInstance();
    config-> iteration = it;
    if(reg_type =="L1")
    {
	    config->type=Reg_type::L1;
    }
    else
    {
	    config->type=Reg_type::L2;
    }

    IO io;
    SpMat X;
    vector<int> Y;
    io.GetData(train_filename, X,Y);
    cout<<"sample size: "<<X.rows()<<endl;
    cout<<"feature size: "<<X.cols()<<endl; 

    Logistic_Reg clf;
    clf.Fit(X,Y);
    clf.Soft_thresholding(clf.beta);
    clf.WriteModel();
    vector<int> pre =clf.Predict(X);

    int correct_count=0;
    for(int i=0;i<pre.size();i++)
    {
	    if (pre[i]==Y[i])
	    {
		    correct_count++;
	    }

    }

    double precision=(double)correct_count/pre.size();
    cout<<"precision score: "<<precision<<endl;

    time_t end_time(time(NULL));

    cout<<"total time used: "<<end_time-start_time<<endl;
	

}
