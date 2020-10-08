#define B01PhotoHadronNuEFlag	0

class B01PhotoHadronNuE
{
public:
    B01PhotoHadronNuE();
    ~B01PhotoHadronNuE();
    int Test();
    int Init();
    int ReadTable();
    int Prepare(double eta);
    int FindParameters(double eta);
    double CalculateNuE(double eta,double x);
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
    //nu_e
    int nne;
    double etane[NH],sne[NH],dne[NH],Bne[NH];
};

B01PhotoHadronNuE::B01PhotoHadronNuE()
{
    Init();
    ReadTable();
}

B01PhotoHadronNuE::~B01PhotoHadronNuE()
{}

int B01PhotoHadronNuE::Test()
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
    //fp= fopen("PhotoHadron-NuE-1.5","w");
    fp= fopen("processes/c_codes/PhotoHadron/Data/PhotoHadron-NuE-30","w");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    for (i=0; i<10000; i++)
    {
        x= 1.0e-4*(i+1.0);
        F= CalculateNuE(eta,x);
        fprintf(fp,"%8.6e %8.6e\n",x,x*F);
    }
    fclose(fp);
    return(0);
}

int B01PhotoHadronNuE::Init()
{
    mpi= mpip;
    M= mp;
    r= mpi/mp;
    R= M/mp;
    eta0= 2.0*r+r*r;
    if (B01PhotoHadronNuEFlag>0)
    {
        printf("mpi= %8.6e mp= %8.6e M= %8.6e r= %8.6e R= %8.6e eta0= %8.6e\n",
        mpi,mp,M,r,R,eta0);
    }
    return(0);
}

int B01PhotoHadronNuE::ReadTable()
{
    int i;
    double rd;
    FILE *fp;
    fp= fopen("processes/c_codes/PhotoHadron/Data/NuE","r");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    fscanf(fp,"%d",&nne);
    for (i=0; i<nne; i++)
    {
        fscanf(fp,"%lf",&rd); etane[i]= rd;
        fscanf(fp,"%lf",&rd); sne[i] = rd;
        fscanf(fp,"%lf",&rd); dne[i] = rd;
        fscanf(fp,"%lf",&rd); Bne[i] = rd;
    }
    fclose(fp);
    for (i=0; i<nne; i++)
    etane[i]= etane[i]*eta0;
    if (B01PhotoHadronNuEFlag>0)
    {
        for (i=0; i<nne; i++)
        {
            printf("%8.6e %8.6e %8.6e %8.6e\n",
            etane[i],sne[i],dne[i],Bne[i]);
        }
    }
    return(0);
}

int B01PhotoHadronNuE::Prepare(double eta)
{
    double xt1,xt2,xt3,xt31,xt32;
    int f = 0;
    xt1 = 2.0*(1.0+eta);
    xt2 = eta+r*r;
    xt31= eta-r*r-2.0*r;
    xt32= eta-r*r+2.0*r;
    xt3 = xt31*xt32;
    if (xt3>0)
    {
        xp= (1.0/xt1)*(xt2+sqrt(xt3));
        xm= (1.0/xt1)*(xt2-sqrt(xt3));
    }
    else
    f = 1;
    xps= xp;
    xms= xm/4.0;
    //
    if (B01PhotoHadronNuEFlag>0)
    {
        printf("xt1= %8.6e xt2= %8.6e xt3= %8.6e\n",xt1,xt2,xt3);
        printf("xp= %8.6e xm= %8.6e\n",xp,xm);
    }
    return(f);
}

int B01PhotoHadronNuE::FindParameters(double eta)
{
    int i;
    double a,delta;
    //
    if (eta<=etane[0])
    {
        st= sne[0];
        dt= dne[0];
        Bt= Bne[0];
        return(0);
    }
    if (eta>=etane[nne-1])
    {
        st= sne[nne-1];
        dt= dne[nne-1];
        Bt= Bne[nne-1];
        return(0);
    }
    //
    delta= 1.0e40;
    for (i=0; i<nne-1; i++)
    {
        if ((fabs(etane[i]-eta)<delta)&&(etane[i]<=eta))
        {
            a= (eta-etane[i])/(etane[i+1]-etane[i]);
            st= (1.0-a)*sne[i]+a*sne[i+1];
            dt= (1.0-a)*dne[i]+a*dne[i+1];
            Bt= (1.0-a)*Bne[i]+a*Bne[i+1];
            delta= fabs(etane[i]-eta);
        }
    }
    if (B01PhotoHadronNuEFlag>0)
    {
        printf("parameters: %8.6e %8.6e %8.6e %8.6e\n",
        eta/eta0,st,dt,Bt);
    }
    return(0);
}

double B01PhotoHadronNuE::CalculateNuE(double eta,double x)
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
        if (B01PhotoHadronNuEFlag>0)
        printf("x= %8.6e ys= %8.6e t11= %8.6e t1= %8.6e\n",x,ys,t11,t1);
    }
    else if (x>=xps)
    F= 0.0;
    return(F);
}
