#define B01PhotoHadronAntiNuEFlag	0

class B01PhotoHadronAntiNuE
{
public:
    B01PhotoHadronAntiNuE();
    ~B01PhotoHadronAntiNuE();
    int Test();
    int Init();
    int ReadTable();
    int Prepare(double eta);
    int FindParameters(double eta);
    double CalculateAntiNuE(double eta,double x);
    //
    double eta0;
private:
    //constants
    double mpi,M,r,R;
    //variables
    double eta,xp,xm,xps,xms;
    //parameters
    double Bt,st,dt;
    //tables
    //electron
    int nane;
    double etaane[NH],sane[NH],dane[NH],Bane[NH];
};

B01PhotoHadronAntiNuE::B01PhotoHadronAntiNuE()
{
    Init();
    ReadTable();
}

B01PhotoHadronAntiNuE::~B01PhotoHadronAntiNuE()
{}

int B01PhotoHadronAntiNuE::Test()
{
    int i;
    double x,F;
    FILE *fp;
    //
    //eta= 3.0*eta0;
    //eta= 5.0*eta0;
    eta= 30.0*eta0;
    Prepare(eta);
    FindParameters(eta);
    //
    //fp= fopen("PhotoHadron-AntiNuE-3.0","w");
    //fp= fopen("PhotoHadron-AntiNuE-5.0","w");
    fp= fopen("processes/c_codes/PhotoHadron/Data/PhotoHadron-AntiNuE-30","w");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    for (i=0; i<10000; i++)
    {
        x= 1.0e-4*(i+1.0);
        F= CalculateAntiNuE(eta,x);
        fprintf(fp,"%8.6e %8.6e\n",x,x*F);
    }
    fclose(fp);
    return(0);
}

int B01PhotoHadronAntiNuE::Init()
{
    mpi= mpip;
    M= mp;
    r= mpi/mp;
    R= M/mp;
    eta0= 2.0*r+r*r;
    if (B01PhotoHadronAntiNuEFlag>0)
    {
        printf("mpi= %8.6e mp= %8.6e M= %8.6e r= %8.6e R= %8.6e eta0= %8.6e\n",
        mpi,mp,M,r,R,eta0);
    }
    return(0);
}

int B01PhotoHadronAntiNuE::ReadTable()
{
    int i;
    double rd;
    FILE *fp;
    fp= fopen("processes/c_codes/PhotoHadron/Data/AntiNuE","r");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    fscanf(fp,"%d",&nane);
    for (i=0; i<nane; i++)
    {
        fscanf(fp,"%lf",&rd); etaane[i]= rd;
        fscanf(fp,"%lf",&rd); sane[i] = rd;
        fscanf(fp,"%lf",&rd); dane[i] = rd;
        fscanf(fp,"%lf",&rd); Bane[i] = rd;
    }
    fclose(fp);
    for (i=0; i<nane; i++)
    etaane[i]= etaane[i]*eta0;
    if (B01PhotoHadronAntiNuEFlag>0)
    {
        for (i=0; i<nane; i++)
        {
            printf("%8.6e %8.6e %8.6e %8.6e\n",
            etaane[i],sane[i],dane[i],Bane[i]);
        }
    }
    return(0);
}

int B01PhotoHadronAntiNuE::Prepare(double eta)
{
    double xt1,xt2,xt3;
    double xmax,xmin;
    int antif = 0;
    xt1 = 2.0*(1.0+eta);
    xt2 = eta-2.0*r;
    xt3= eta*(eta-4.0*r*(1.0+r));
    if (xt3 > 0)
    {
        xmax= (1.0/xt1)*(xt2+sqrt(xt3));
        xmin= (1.0/xt1)*(xt2-sqrt(xt3));
    }
    else
    antif = 1;
    xps= xmax;
    xms= xmin/2.0;
    //
    if (B01PhotoHadronAntiNuEFlag>0)
    {
        printf("xt1= %8.6e xt2= %8.6e xt3= %8.6e\n",xt1,xt2,xt3);
        printf("xp= %8.6e xm= %8.6e\n",xp,xm);
    }
    return(antif);
}

int B01PhotoHadronAntiNuE::FindParameters(double eta)
{
    int i;
    double a,delta,rho,rho0;
    //
    if (eta<=etaane[0])
    {
        st= sane[0];
        dt= dane[0];
        rho= eta/eta0;
        rho0= etaane[0]/eta0;
        Bt= Bane[0]*(rho-2.14)/(rho0-2.14);
        return(0);
    }
    if (eta>=etaane[nane-1])
    {
        st= sane[nane-1];
        dt= dane[nane-1];
        Bt= Bane[nane-1];
        return(0);
    }
    //
    delta= 1.0e40;
    for (i=0; i<nane-1; i++)
    {
        if ((fabs(etaane[i]-eta)<delta)&&(etaane[i]<=eta))
        {
            a= (eta-etaane[i])/(etaane[i+1]-etaane[i]);
            st= (1.0-a)*sane[i]+a*sane[i+1];
            dt= (1.0-a)*dane[i]+a*dane[i+1];
            Bt= (1.0-a)*Bane[i]+a*Bane[i+1];
            delta= fabs(etaane[i]-eta);
        }
    }
    if (B01PhotoHadronAntiNuEFlag>0)
    {
        printf("parameters: %8.6e %8.6e %8.6e %8.6e\n",
        eta/eta0,st,dt,Bt);
    }
    return(0);
}

double B01PhotoHadronAntiNuE::CalculateAntiNuE(double eta,double x)
{
    double t1,t2,t11,t21,p;
    double ys,F,rho;
    //
    rho= eta/eta0;
    p= 6.0*(1.0-exp(1.5*(4.0-rho)));
    if (rho<4.0) p= 0.0;
    //
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
        if (B01PhotoHadronAntiNuEFlag>0)
        printf("x= %8.6e ys= %8.6e t11= %8.6e t1= %8.6e\n",x,ys,t11,t1);
    }
    else if (x>=xps)
    F= 0.0;
    if (rho<2.14) F= 0.0;
    return(F);
}
