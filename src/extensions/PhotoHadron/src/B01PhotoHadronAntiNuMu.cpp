#include <string>

#define B01PhotoHadronAntiNuMuFlag	0

class B01PhotoHadronAntiNuMu
{
public:
    B01PhotoHadronAntiNuMu(std::string data_dir_path);
    ~B01PhotoHadronAntiNuMu();
    int Test();
    int Init();
    int ReadTable();
    int Prepare(double eta);
    int FindParameters(double eta);
    double CalculateAntiNuMu(double eta,double x);
    //
    double eta0;
private:
    std::string data_dir;
    //constants
    double mpi,M,r,R;
    //variables
    double eta,xp,xm,xps,xms;
    //parameters
    double Bt,st,dt;
    //tables
    //anti-nu_mu
    int nanm;
    double etaanm[NH],sanm[NH],danm[NH],Banm[NH];
};

B01PhotoHadronAntiNuMu::B01PhotoHadronAntiNuMu(std::string data_dir_path)
{
    data_dir = data_dir_path;
    mpi= mpip;
    M= mp;
    r= mpi/mp;
    R= M/mp;
    eta0= 2.0*r+r*r;
    if (B01PhotoHadronAntiNuMuFlag>0)
    {
        printf("mpi= %8.6e mp= %8.6e M= %8.6e r= %8.6e R= %8.6e eta0= %8.6e\n",
        mpi,mp,M,r,R,eta0);
    }
    ReadTable();
}

B01PhotoHadronAntiNuMu::~B01PhotoHadronAntiNuMu()
{}

int B01PhotoHadronAntiNuMu::Test()
{
    int i;
    double x,F;
    FILE *fp;
    //
    //eta= 1.5*eta0;
    //eta= 1.55*eta0;
    //eta= 1.6*eta0;
    eta= 30.0*eta0;
    Prepare(eta);
    FindParameters(eta);
    //
    //fp= fopen("PhotoHadron-AntiNuMu-1.5","w");
    fp= fopen((data_dir + "/PhotoHadron-AntiNuMu-30").c_str(), "w");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    for (i=0; i<10000; i++)
    {
        x= 1.0e-4*(i+1.0);
        F= CalculateAntiNuMu(eta,x);
        fprintf(fp,"%8.6e %8.6e\n",x,x*F);
    }
    fclose(fp);
    return(0);
}

int B01PhotoHadronAntiNuMu::ReadTable()
{
    int i;
    double rd;
    FILE *fp;
    fp = fopen((data_dir + "/AntiNuMu").c_str(), "r");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    fscanf(fp,"%d",&nanm);
    for (i=0; i<nanm; i++)
    {
        fscanf(fp,"%lf",&rd); etaanm[i]= rd;
        fscanf(fp,"%lf",&rd); sanm[i] = rd;
        fscanf(fp,"%lf",&rd); danm[i] = rd;
        fscanf(fp,"%lf",&rd); Banm[i] = rd;
    }
    fclose(fp);
    for (i=0; i<nanm; i++)
    etaanm[i]= etaanm[i]*eta0;
    if (B01PhotoHadronAntiNuMuFlag>0)
    {
        for (i=0; i<nanm; i++)
        {
            printf("%8.6e %8.6e %8.6e %8.6e\n",
            etaanm[i],sanm[i],danm[i],Banm[i]);
        }
    }
    return(0);
}

int B01PhotoHadronAntiNuMu::Prepare(double eta)
{
    double xt1,xt2,xt3,xt31,xt32;
    int antif = 0;
    xt1 = 2.0*(1.0+eta);
    xt2 = eta+r*r;
    xt31= eta-r*r-2.0*r;
    xt32= eta-r*r+2.0*r;
    xt3 = xt31*xt32;
    xp= 0.0;
    xm= 0.0;
    if (xt3>SMALLD)
    {
        xp= (1.0/xt1)*(xt2+sqrt(xt3));
        xm= (1.0/xt1)*(xt2-sqrt(xt3));
    }
    else
    antif = 1;
    xps= xp;
    xms= xm/4.0;
    //
    if (B01PhotoHadronAntiNuMuFlag>0)
    {
        printf("B01PhotoHadronAntiNuMu::Prepare\n");
        printf("xt1= %8.6e xt2= %8.6e xt3= %8.6e\n",xt1,xt2,xt3);
        printf("xp= %8.6e xm= %8.6e\n",xp,xm);
    }
    return(antif);
}

int B01PhotoHadronAntiNuMu::FindParameters(double eta)
{
    int i;
    double a,delta;
    //
    if (eta<=etaanm[0])
    {
        st= sanm[0];
        dt= danm[0];
        Bt= Banm[0];
        return(0);
    }
    if (eta>=etaanm[nanm-1])
    {
        st= sanm[nanm-1];
        dt= danm[nanm-1];
        Bt= Banm[nanm-1];
        return(0);
    }
    //
    delta= 1.0e40;
    for (i=0; i<nanm-1; i++)
    {
        if ((fabs(etaanm[i]-eta)<delta)&&(etaanm[i]<=eta))
        {
            a= (eta-etaanm[i])/(etaanm[i+1]-etaanm[i]);
            st= (1.0-a)*sanm[i]+a*sanm[i+1];
            dt= (1.0-a)*danm[i]+a*danm[i+1];
            Bt= (1.0-a)*Banm[i]+a*Banm[i+1];
            delta= fabs(etaanm[i]-eta);
        }
    }
    if (B01PhotoHadronAntiNuMuFlag>0)
    {
        printf("parameters: %8.6e %8.6e %8.6e %8.6e\n",
        eta/eta0,st,dt,Bt);
    }
    return(0);
}

double B01PhotoHadronAntiNuMu::CalculateAntiNuMu(double eta,double x)
{
    double t1,t2,t11,t21,p;
    double ys,F;
    //
    p= 2.5+1.4*log(eta/eta0);
    if (x<=xms)
    F= Bt*pow(log(2.0),p);
    else if ((x>xms)&&(x<xps))
    {
        ys= (x-xms)/(xps-xms);
        t11= log(x/xms);
        t1= exp(-st*pow(t11,dt));
        t21= log(2.0/(1.0+ys*ys));
        t2= pow(t21,p);
        F= Bt*t1*t2;
        if (B01PhotoHadronAntiNuMuFlag>0)
        printf("x= %8.6e ys= %8.6e t11= %8.6e t1= %8.6e\n",x,ys,t11,t1);
    }
    else if (x>=xps)
    F= 0.0;
    return(F);
}
