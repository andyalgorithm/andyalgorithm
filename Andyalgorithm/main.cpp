#include "andyalgorithm.h"

#include <QMap>
#include <cstdlib>
#include <iostream>
#include <QDir>


using namespace std;


int main(int argc,char* argv[])
{
    string  S = "ABBACABCD";
    string T = "BCD";
    cout << minWindow(S,T)<<endl;

    return 0;
}

