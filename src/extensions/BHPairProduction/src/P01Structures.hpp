// #include <stdlib.h>
#include <math.h>

#define SMALLD			1.0e-40
#define VSMALLD			1.0e-200
#define P01StructuresFlag	0

class P01Structures
{
    public:
    P01Structures();
    ~P01Structures();
    int Init();
//
    double pi,alpha,r0;
    double mp,me;
    double k,T;					//k= Boltzmann constant
    private:
};
