#include <stdio.h>
#include "gamma_gamma_interaction_rate.c"


void gamma_gamma_interaction_rate(char *photon_file, double E_min, double E_max)
{
    gamma_gamma_interaction_rate_internal(photon_file, E_min, E_max);
}

int main()
{
    return 0;
}
