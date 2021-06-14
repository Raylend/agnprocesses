#include <string>

#ifndef ELEM_PROCESSES_INCLUDED
#define ELEM_PROCESSES_INCLUDED
#include "./constants.c"
#include "./B01PhotoHadronAntiNuE.cpp"
#include "./B01PhotoHadronAntiNuMu.cpp"
#include "./B01PhotoHadronG.cpp"
#include "./B01PhotoHadronP.cpp"
#include "./B01PhotoHadronE.cpp"
#include "./B01PhotoHadronNuMu.cpp"
#include "./B01PhotoHadronNuE.cpp"
#endif


#define B01PlanckianCMBFlag	0


class B01PlanckianCMB
{
    public:
    B01PlanckianCMB(std::string data_dir_path);
    ~B01PlanckianCMB();
    int Process();
    int Integrate(double T,double Ep,double deps);
    //
    FILE *fpa;
    FILE *fpn;
    //
    B01PhotoHadronG *phg;
    B01PhotoHadronP *php;
    B01PhotoHadronE *phe;
    B01PhotoHadronNuMu *phnm;
    B01PhotoHadronAntiNuMu *phanm;
    B01PhotoHadronNuE *phne;
    B01PhotoHadronAntiNuE *phane;
private:
    std::string data_dir;
};

B01PlanckianCMB::B01PlanckianCMB(std::string data_dir_path)
{
    data_dir = data_dir_path;
    phg = new B01PhotoHadronG(data_dir);
    php = new B01PhotoHadronP(data_dir);
    phe = new B01PhotoHadronE(data_dir);
    phnm = new B01PhotoHadronNuMu(data_dir);
    phanm = new B01PhotoHadronAntiNuMu(data_dir);
    phne = new B01PhotoHadronNuE(data_dir);
    phane = new B01PhotoHadronAntiNuE(data_dir);
}

B01PlanckianCMB::~B01PlanckianCMB()
{}

int B01PlanckianCMB::Process()
{
    double T,Ep,deps;
    //
    T= 2.725*8.627330350e-5;			//[eV]
    Ep= 1.0e20/1.0e9;				//[eV]->[GeV]
    //Ep= 1.0e21/1.0e9;				//[eV]->[GeV]
    deps= 1.0e-5;				//[eV]
    //
    fpa= fopen("processes/c_codes/PhotoHadron/Data/Active","w");
    if (fpa == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    fpn= fopen("processes/c_codes/PhotoHadron/Data/Neutrino","w");
    if (fpn == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    fprintf(fpa,"%13.6e\n",T);
    fprintf(fpn,"%13.6e\n",T);
    Integrate(T,Ep,deps);
    fclose(fpn);
    fclose(fpa);
    //
    return(0);
}

int B01PlanckianCMB::Integrate(double T,double Ep,double deps)
{
    int i,j,flag1,flag2, antiflag1, antiflag2;
    double pi,GeV;
    double eta0,eta;
    double eps0,eps;
    double x,f,F;
    double sg,sp,se;
    double snm,sanm,sne,sane;
    //
    pi = 3.141592654;
    GeV= 1.0e-9;				//[eV]
    //
    eta0= phg->eta0;
    eps0= 1.0e9*(eta0*mp*mp)/(4.0*Ep);		//[eV]
    //
    if (B01PlanckianCMBFlag>0)
    printf("T= %13.6e [eV] eta0= %13.6e eps0= %13.6e [eV]\n",T,eta0,eps0);
    //
    for (j=0; j<1000; j++)
    {
        sg  = 0.0;
        sp  = 0.0;
        se  = 0.0;
        snm = 0.0;
        sanm= 0.0;
        sne = 0.0;
        sane= 0.0;
        x= 1.0e-4*pow(10.0,4.0e-3*j);
        for (i=0; i<1000; i++)
        {
            eps= eps0+deps*i;
            f= (1.0/(pi*pi))*(eps*eps)/(exp(eps/T)-1.0);
            eta= (4.0*GeV*eps*Ep)/(mp*mp);
            //gamma
            phg->Prepare(eta);
            phg->FindParameters(eta);
            F= phg->CalculateG(eta,x);
            sg+= f*F*deps;
            //positron
            php->Prepare(eta);
            php->FindParameters(eta);
            F= php->CalculateP(eta,x);
            sp+= f*F*deps;
            //electron
            phe->Prepare(eta);
            phe->FindParameters(eta);
            F= phe->CalculateE(eta,x);
            se+= f*F*deps;
            //nu_mu
            flag1 = phnm->Prepare(eta);
            if (flag1 == 0)
            {
                phnm->FindParameters(eta);
                F= phnm->CalculateNuMu(eta,x);
                snm+= f*F*deps;
            }
            //anti-nu_mu
            antiflag1 = phanm->Prepare(eta);
            if (antiflag1 == 0)
            {
                phanm->FindParameters(eta);
                F= phanm->CalculateAntiNuMu(eta,x);
                sanm+= f*F*deps;
            }
            //nu_e
            flag2 = phne->Prepare(eta);
            if (flag2 == 0)
            {
                phne->FindParameters(eta);
                F= phne->CalculateNuE(eta,x);
                sne+= f*F*deps;
            }
            //anti-nu_e
            antiflag2 = phane->Prepare(eta);
            if (antiflag2 == 0)
            {
                phane->FindParameters(eta);
                F= phane->CalculateAntiNuE(eta,x);
                sane+= f*F*deps;
            }
            //
            if (B01PlanckianCMBFlag>0)
            {
                printf("n= %8d eps= %13.6e f= %13.6e eta= %13.6e x= %13.6e F= %13.6e\n",
                i,eps,f,eta,x,F);
            }
        }
        // fprintf(fpa,"%13.6e  %13.6e %13.6e %13.6e\n",x,1.0e30*x*sg,1.0e30*x*sp,1.0e30*x*se);
        fprintf(fpa,"%13.6e  %13.6e\n", x, 1.0e30*x*sp);
        fprintf(fpn,"%13.6e  %13.6e %13.6e %13.6e %13.6e\n",x,1.0e30*x*snm,1.0e30*x*sanm,
        1.0e30*x*sne,1.0e30*x*sane);
    }
    return(0);
}
