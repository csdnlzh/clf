#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include "io.h"


void IO::GetData(string filename, SpMat& X,vector<int> & Y)
{
    ifstream infile(filename.c_str());
    string linestr;
    int sample_size=0;
    int feature_size=0;

    vector<T> triplet_list;
    triplet_list.reserve(1000000);
    while(!infile.eof())
    {
        getline(infile,linestr);
        vector<string> v;
        boost::algorithm::split(v,linestr,boost::algorithm::is_any_of(" "));
        if (v.size()<=1)
        {
            continue;
        }
        Y.push_back(atoi(v[0].c_str())); //get the label;

        for (int i=1;i<v.size();i++)
        {

            vector<string> vv;
            boost::algorithm::split(vv,v[i],boost::algorithm::is_any_of(":"));
            if(vv.size()!=2)
            {
                continue;
            }
            int idx=atoi(vv[0].c_str());
            if(idx>feature_size)
            {
                feature_size=idx;
            }
            double value=atof(vv[1].c_str());
            triplet_list.push_back(T(sample_size,idx,value));
        }
        sample_size++;
    }
    X.resize(sample_size,feature_size+1);
    X.setFromTriplets(triplet_list.begin(),triplet_list.end());
}
