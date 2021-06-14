#include <string>

#define B01PhotoHadronNuMuFlag	0

class B01PhotoHadronNuMu
{
public:
    B01PhotoHadronNuMu(std::string data_dir_path);
    ~B01PhotoHadronNuMu();
    int Test();
    int Init();
    int ReadTable();
    int Prepare(double eta);
    int FindParameters(double eta);
    double CalculateNuMu(double eta,double x);
    //
    double eta0;
private:
    std::string data_dir;
    //constants
    double mpi,M,r;
    //variables
    double eta,xp,xm,xps,xms;
    //parameters
    double Bt,st,dt;
    //tables
    //nu_mu
    int nnm;
    double etanm[NH],snm[NH],dnm[NH],Bnm[NH];
};

B01PhotoHadronNuMu::B01PhotoHadronNuMu(std::string data_dir_path)
{
    data_dir = data_dir_path;
    Init();
    ReadTable();
}

B01PhotoHadronNuMu::~B01PhotoHadronNuMu()
{}

int B01PhotoHadronNuMu::Test()
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
    //fp= fopen("PhotoHadron-NuMu-1.5","w");
    fp= fopen((data_dir + "PhotoHadron-NuMu-30").c_str(), "w");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    for (i=0; i<10000; i++)
    {
        x= 1.0e-4*(i+1.0);
        F= CalculateNuMu(eta,x);
        fprintf(fp,"%8.6e %8.6e\n",x,x*F);
    }
    fclose(fp);
    return(0);
}

int B01PhotoHadronNuMu::Init()
{
    mpi= mpip;
    M= mp;
    r= mpi/mp;
    eta0= 2.0*r+r*r;
    if (B01PhotoHadronNuMuFlag>0)
    {
        printf("mpi= %8.6e mp= %8.6e M= %8.6e r= %8.6e eta0= %8.6e\n",
        mpi,mp,M,r,eta0);
    }
    return(0);
}

int B01PhotoHadronNuMu::ReadTable()
{
    int i;
    double rd;
    FILE *fp;
    //
    fp = fopen((data_dir + "NuMu").c_str(), "r");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    assert(fp!=NULL);
    fscanf(fp,"%d",&nnm);
    assert(nnm<NH);
    for (i=0; i<nnm; i++)
    {
        fscanf(fp,"%lf",&rd); etanm[i]= rd;
        fscanf(fp,"%lf",&rd); snm[i] = rd;
        fscanf(fp,"%lf",&rd); dnm[i] = rd;
        fscanf(fp,"%lf",&rd); Bnm[i] = rd;
    }
    fclose(fp);
    for (i=0; i<nnm; i++)
    etanm[i]= etanm[i]*eta0;
    if (B01PhotoHadronNuMuFlag>0)
    {
        for (i=0; i<nnm; i++)
        {
            printf("%8.6e %8.6e %8.6e %8.6e\n",
            etanm[i],snm[i],dnm[i],Bnm[i]);
        }
    }
    return(0);
}

int B01PhotoHadronNuMu::Prepare(double eta)
{
    int f;
    double xt1,xt2,xt3,xt31,xt32;
    double rho;
    //
    xt1 = 2.0*(1.0+eta);
    xt2 = eta+r*r;
    xt31= eta-r*r-2.0*r;
    xt32= eta-r*r+2.0*r;
    xt3 = xt31*xt32;
    //
    assert(fabs(xt1)>SMALLD);
    //    assert(xt3>0.0);
    //
    f= 0;
    if (xt3<0.0)
    {
        //xp= (1.0/xt1)*xt2;
        //xm= (1.0/xt1)*xt2;
        f= 1;
    }
    if (xt3>0.0)
    {
        xp= (1.0/xt1)*(xt2+sqrt(xt3));
        xm= (1.0/xt1)*(xt2-sqrt(xt3));
    }
    //
    rho= eta/eta0;
    if (rho<2.14)
    xps= 0.427*xp;
    else if (rho<10.0)
    xps= (0.427+0.0729*(rho-2.14))*xp;
    else
    xps= xp;
    xms= 0.427*xm;
    //
    if (B01PhotoHadronNuMuFlag>0)
    {
        printf("xt1= %13.6e xt2= %13.6e xt3= %13.6e\n",xt1,xt2,xt3);
        printf("xp= %13.6e xm= %13.6e\n",xp,xm);
        printf("rho= %13.6e xps= %13.6e xms= %13.6e\n",rho,xps,xms);
    }
    return(f);
}

int B01PhotoHadronNuMu::FindParameters(double eta)
{
    int i;
    double a,delta;
    //
    if (eta<=etanm[0])
    {
        st= snm[0];
        dt= dnm[0];
        Bt= Bnm[0];
        return(0);
    }
    if (eta>=etanm[nnm-1])
    {
        st= snm[nnm-1];
        dt= dnm[nnm-1];
        Bt= Bnm[nnm-1];
        return(0);
    }
    //
    delta= 1.0e40;
    for (i=0; i<nnm-1; i++)
    {
        if ((fabs(etanm[i]-eta)<delta)&&(etanm[i]<=eta))
        {
            a= (eta-etanm[i])/(etanm[i+1]-etanm[i]);
            st= (1.0-a)*snm[i]+a*snm[i+1];
            dt= (1.0-a)*dnm[i]+a*dnm[i+1];
            Bt= (1.0-a)*Bnm[i]+a*Bnm[i+1];
            delta= fabs(etanm[i]-eta);
        }
    }
    if (B01PhotoHadronNuMuFlag>0)
    {
        printf("parameters: %8.6e %8.6e %8.6e %8.6e\n",
        eta/eta0,st,dt,Bt);
    }
    return(0);
}

double B01PhotoHadronNuMu::CalculateNuMu(double eta,double x)
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
        if (B01PhotoHadronNuMuFlag>0)
        printf("x= %8.6e ys= %8.6e t11= %8.6e t1= %8.6e\n",x,ys,t11,t1);
    }
    else if (x>=xps)
    F= 0.0;
    return(F);
}
