#include "src/B01Structures.cpp"
#include "src/B01PhotoHadronG.cpp"
#include "src/B01PhotoHadronP.cpp"
#include "src/B01PhotoHadronE.cpp"
#include "src/B01PhotoHadronNuMu.cpp"
#include "src/B01PhotoHadronAntiNuMu.cpp"
#include "src/B01PhotoHadronNuE.cpp"
#include "src/B01PhotoHadronAntiNuE.cpp"
#include "src/B01PlanckianCMB.cpp"
#include "src/B01Planckian.cpp"
#include "src/B01SSC.cpp"
#include <stdio.h>

// B01PhotoHadronG phg;
// B01PhotoHadronP php;
// B01PhotoHadronE phe;
// B01PhotoHadronAntiNuMu phanm;
// B01PhotoHadronAntiNuE phane;
// B01PhotoHadronNuMu phnm;
// B01PhotoHadronNuE phne;
// B01PlanckianCMB plcmb;
// B01Planckian pl;
B01SSC p;

void photohadron(char* file_path, double energy_proton_min, double energy_proton_max, double p_p, double E_cut)
{
    // phg.Test();	//gamma
    // php.Test();	//positron
    // phe.Test();	//electron
    // phanm.Test();//anti-nu_mu
    // phane.Test();//anti-nu_e
    // phnm.Test();	//nu_mu
    // phne.Test();	//nu_e
    printf("%le\n", mp);
    p.Process(file_path, energy_proton_min, energy_proton_max, p_p, E_cut);
}

int main()
{
    // photohadron();
    return 0;
}
