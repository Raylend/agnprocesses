#include "src/P01Structures.cpp"
#include "src/P01Pair.cpp"
#include <stdio.h>

P01Pair p;

void bh_pair_production(char* file_path, double energy_proton_min, double energy_proton_max, double p_p, double E_cut)
{
    p.Process(file_path, energy_proton_min, energy_proton_max, p_p, E_cut);
}

int main()
{
  return(0);
}
