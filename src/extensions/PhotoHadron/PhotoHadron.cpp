#include "src/B01SSC.cpp"

#include <stdio.h>
#include <string>


void photohadron(
    const char* input_file_path, const char* output_dir_path,
    double energy_proton_min, double energy_proton_max,
    double p_p, double E_cut
)
{
    std::string data_file_test_dir = "/home/njvh/Documents/Science/gamma/agnprocesses/src/agnprocesses/data/PhotoHadronData/";
    B01SSC *p = new B01SSC(data_file_test_dir, std::string(input_file_path), std::string(output_dir_path));

    p->Process(energy_proton_min, energy_proton_max, p_p, E_cut);
}

int main()
{
    const char* filepath = "test";
    const char* output = "test";
    photohadron(filepath, output, 1.0, 2.0, 3.0, 4.0);
    return 0;
}
