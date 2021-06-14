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


#define B01PlanckianFlag	0


class B01Planckian
{
public:
    B01Planckian(std::string data_dir_path);
    ~B01Planckian();
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

B01Planckian::B01Planckian(std::string data_dir_path)
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

B01Planckian::~B01Planckian()
{}

int B01Planckian::Process()
{
    int i;
    double T,Ep,deps;
    //
    fpa= fopen((data_dir + "Active").c_str(),"w");
    if (fpa == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    fpn= fopen((data_dir + "Neutrino").c_str(),"w");
    if (fpn == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    T= 2.725; //10.0;					//blackbody temperature [eV]
    fprintf(fpa,"%13.6e\n",T);
    fprintf(fpn,"%13.6e\n",T);
    /*for (i=0; i<100; i++)
    { //100 different values of proton energy
    Ep= (1.0e14/1.0e9)*pow(10.0,0.04*i);	//primary proton energy [eV]->[GeV]
    deps= 0.1;				//integration step (on photon field energy) [eV]
    //probably we will need to change deps
    fprintf(fpa,"%13.6e\n",Ep*1.0e9);	//[eV]
    fprintf(fpn,"%13.6e\n",Ep*1.0e9);	//[eV]
    Integrate(T,Ep,deps);
    }*/
    //
    Ep= 1.0e16/1.0e9;				//primary proton energy [eV]->[GeV]
    deps= 0.1;					//integration step (on photon field energy) [eV]
    //probably we will need to change deps
    fprintf(fpa,"%13.6e\n",Ep*1.0e9);	//[eV]
    fprintf(fpn,"%13.6e\n",Ep*1.0e9);	//[eV]
    Integrate(T,Ep,deps);
    //
    fclose(fpn);
    fclose(fpa);
//
return(0);
}

int B01Planckian::Integrate(double T,double Ep,double deps)
{
    if (fpa == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    if (fpn == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    int i, j, flag1, flag2;
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
    eps0= 1.0e9*(eta0*mp*mp)/(4.0*Ep);		//energy threshold [eV]
    //
    //    if (B01PlanckianFlag>0)
    printf("T= %13.6e [eV] eta0= %13.6e eps0= %13.6e [eV]\n",T,eta0,eps0);
    //
    for (j=0; j<1000; j++)
    { //1000 different values of x
        sg  = 0.0;
        sp  = 0.0;
        se  = 0.0;
        snm = 0.0;
        sanm= 0.0;
        sne = 0.0;
        sane= 0.0;
        x= 1.0e-4*pow(10.0,4.0e-3*j);		//x= (E_{gamma}/E_{p}); (E_e/E_{p}), (...)
        for (i=0; i<1000; i++)
        { //integrate on photon field energy eps: 1000 different values of eps
            eps= eps0+deps*i;			//photon energy [eV]
            f= (1.0/(pi*pi))*(eps*eps)/(exp(eps/T)-1.0); //number density of photon field [1/(cm^{3}*eV)]
            //insert new f (number density) here
            eta= (4.0*GeV*eps*Ep)/(mp*mp);		 //approximation parameter: Phi(eta,x) -> here F(eta,x)
            //gamma
            phg->Prepare(eta);
            phg->FindParameters(eta);
            F= phg->CalculateG(eta,x);
            // dN/dx= Integral(f*Phi*deps)
            sg+= f*F*deps;			//[dN/(dx*dt*dV)] [1/(s*cm^{3})] -> proportional to [dN/dE]
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
            flag1= phnm->Prepare(eta);
            if (flag1==0)
            {
                phnm->FindParameters(eta);
                F= phnm->CalculateNuMu(eta,x);
                snm+= f*F*deps;
            }
            //anti-nu_mu
            phanm->Prepare(eta);
            phanm->FindParameters(eta);
            F= phanm->CalculateAntiNuMu(eta,x);
            sanm+= f*F*deps;
            //nu_e
            flag2 = phne->Prepare(eta);
            if (flag2==0)
            {
                phne->FindParameters(eta);
                F= phne->CalculateNuE(eta,x);
                sne+= f*F*deps;
            }
            //anti-nu_e
            phane->Prepare(eta);
            phane->FindParameters(eta);
            F= phane->CalculateAntiNuE(eta,x);
            sane+= f*F*deps;
            //
            if (B01PlanckianFlag>0)
            {
                printf("n= %8d eps= %13.6e f= %13.6e eta= %13.6e x= %13.6e F= %13.6e\n",
                i,eps,f,eta,x,F);
            }
        }
        fprintf(fpa,"%13.6e  %13.6e %13.6e %13.6e\n",x,1.0e30*x*sg,1.0e30*x*sp,1.0e30*x*se); //[x*dN/dE] -> proportional to [E*dN/dE]
        fprintf(fpn,"%13.6e  %13.6e %13.6e %13.6e %13.6e\n",x,1.0e30*x*snm,1.0e30*x*sanm,1.0e30*x*sne,1.0e30*x*sane);
        //	fprintf(fpa,"%13.6e  %13.6e\n",x,1.0e30*x*sg); //[x*dN/dE] -> proportional to [E*dN/dE]
        //  printf(fpn,"%13.6e  %13.6e\n",x,1.0e30*x*snm); //[x*dN/dE] -> proportional to [E*dN/dE]
    }
    fprintf(fpa,"\n");
    fprintf(fpn,"\n");
    return(0);
}
