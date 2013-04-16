#ifndef IO_H_INCLUDED
#define IO_H_INCLUDED
#include <string>
#include "logistic.h"

using namespace std;

class IO
{
public:
    IO(){}
    void GetData(string filename, SpMat& X,vector<int>& Y);
};

#endif // IO_H_INCLUDED
