#include <stdio.h>

#include "P01Structures.hpp"

P01Structures::P01Structures()
{
    Init();
}

P01Structures::~P01Structures()
{}

int P01Structures::Init()
{
    pi= 3.141592654;
    alpha= 7.2973525664e-3;
    r0= 2.8179380e-13;				//[cm]
    mp= 9.382796e8;				//[eV]
    me= 5.110034e5;				//electron mass [eV]
    k= 8.6173324e-5;				//[eV/K]
    T= 2.725;					//for Planckian [K]
    //T= 27.25;					//[K]
    //T= (1.0+0.25)*2.725;					//[K]
    //T= 2.0*2.725;					//[K]
//
    if (P01StructuresFlag>0)
	    printf("mp= %13.6e [eV] me= %13.6e [eV]\n",mp,me);
    return(0);
}
