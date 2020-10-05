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

//B01PhotoHadronG phg;
//B01PhotoHadronP php;
//B01PhotoHadronE phe;
//B01PhotoHadronAntiNuMu phanm;
//B01PhotoHadronAntiNuE phane;
//B01PhotoHadronNuMu phnm;
//B01PhotoHadronNuE phne;
// B01PlanckianCMB pl;//cmb;
//B01Planckian pl;
B01SSC pl;

int main()
{
    //    phg.Test();					//gamma
    //    php.Test();					//positron
    //    phe.Test();					//electron
    //    phanm.Test();				//anti-nu_mu
    //    phane.Test();				//anti-nu_e
    //    phnm.Test();				//nu_mu
    //    phne.Test();				//nu_e
    ////    plcmb.Process();
    pl.Process();
    return(0);
}
