#include "src/B01SSC.cpp"

#include <stdio.h>
#include <string>


void photohadron(
    const char* data_dir_path, const char* input_file_path, const char* output_dir_path,
    double energy_proton_min, double energy_proton_max,
    double p_p, double E_cut
)
{
    B01SSC *p = new B01SSC(
        std::string(data_dir_path), std::string(input_file_path), std::string(output_dir_path)
    );
    p->Process(energy_proton_min, energy_proton_max, p_p, E_cut);
}
