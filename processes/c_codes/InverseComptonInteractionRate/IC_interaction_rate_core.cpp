#include <stdio.h>
#include "IC_interaction_rate.c"


void IC_interaction_rate(char *photon_file, double E_min, double E_max, double E_thr)
{
    inverse_compton_interaction_rate_internal(photon_file, E_min, E_max, E_thr);
}

int main()
{
    return 0;
}
