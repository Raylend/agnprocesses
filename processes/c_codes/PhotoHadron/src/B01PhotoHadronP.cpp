#define B01PhotoHadronPFlag	0
//positrons
class B01PhotoHadronP
{
public:
    B01PhotoHadronP();
    ~B01PhotoHadronP();
    int Test();
    int Init();
    int ReadTable();
    int Prepare(double eta);
    int FindParameters(double eta);
    double CalculateP(double eta,double x);
    //
    double eta0;
private:
    //constants
    double mpi,M,r,R;
    //variables
    double eta,xp,xm,xps,xms;
    //parameters
    double Bt,st,dt;
    //tables: positron
    int np;
    double etap[NH],sp[NH],dp[NH],Bp[NH];
};

B01PhotoHadronP::B01PhotoHadronP()
{
    Init();
    ReadTable();
}

B01PhotoHadronP::~B01PhotoHadronP()
{}

int B01PhotoHadronP::Test()
{
    int i;
    double x,F;
    FILE *fp;
    //
    //eta= 1.5*eta0;
    //eta= 1.55*eta0;
    //eta= 1.6*eta0;
    //eta= 3.0*eta0;
    //eta= 10.0*eta0;
    eta= 30.0*eta0;
    //eta= 100.0*eta0;
    Prepare(eta);
    FindParameters(eta);
    //
    //fp= fopen("PhotoHadron-Positron-1.5","w");
    //fp= fopen("PhotoHadron-Positron-3.0","w");
    //fp= fopen("PhotoHadron-Positron-10","w");
    fp= fopen("processes/c_codes/PhotoHadron/Data/PhotoHadron-Positron-30","w");
    //fp= fopen("PhotoHadron-Positron-100","w");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    for (i=0; i<10000; i++)
    {
        x= 1.0e-4*(i+1.0);
        F= CalculateP(eta,x);
        fprintf(fp,"%8.6e %8.6e\n",x,x*F);
    }
    fclose(fp);
    return(0);
}

int B01PhotoHadronP::Init()
{
    mpi= mpip;
    M= mp;
    r= mpi/mp;
    R= M/mp;
    eta0= 2.0*r+r*r;
    if (B01PhotoHadronPFlag>0)
    {
        printf("mpi= %8.6e mp= %8.6e M= %8.6e r= %8.6e R= %8.6e eta0= %8.6e\n", mpi,mp,M,r,R,eta0);
    }
    return(0);
}

int B01PhotoHadronP::ReadTable()
{
    int i;
    double rd;
    FILE *fp;
    fp= fopen("processes/c_codes/PhotoHadron/Data/Positron","r");
    if (fp == NULL)
    {
        printf("Couldn't create or read the file!\n");
        exit(1);
    }
    fscanf(fp,"%d",&np);
    for (i=0; i<np; i++)
    {
        fscanf(fp,"%lf",&rd); etap[i]= rd;
        fscanf(fp,"%lf",&rd); sp[i] = rd;
        fscanf(fp,"%lf",&rd); dp[i] = rd;
        fscanf(fp,"%lf",&rd); Bp[i] = rd;
    }
    fclose(fp);
    for (i=0; i<np; i++)
    etap[i]= etap[i]*eta0;
    if (B01PhotoHadronPFlag>0)
    {
        for (i=0; i<np; i++)
        {
            printf("%8.6e %8.6e %8.6e %8.6e\n",
            etap[i],sp[i],dp[i],Bp[i]);
        }
    }
    return(0);
}

int B01PhotoHadronP::Prepare(double eta)
{
    double xt1,xt2,xt3,xt31,xt32;
    xt1 = 2.0*(1.0+eta);
    xt2 = eta+r*r;
    xt31= eta-r*r-2.0*r;
    xt32= eta-r*r+2.0*r;
    xt3 = xt31*xt32;
    xp= 0.0;
    xm= 0.0;
    int f = 0;
    if (xt3>SMALLD)
    {
        xp= (1.0/xt1)*(xt2+sqrt(xt3));
        xm= (1.0/xt1)*(xt2-sqrt(xt3));
    }
    else
    f = 1;
    xps= xp;
    xms= xm/4.0;
    //
    if (B01PhotoHadronPFlag>0)
    {
        printf("B01PhotoHadronP::Prepare\n");
        printf("xt1= %8.6e xt2= %8.6e xt3= %8.6e\n",xt1,xt2,xt3);
        printf("xp= %8.6e xm= %8.6e\n",xp,xm);
    }
    return(f);
}

int B01PhotoHadronP::FindParameters(double eta)
{
    int i;
    double a,delta;
    //
    if (eta<=etap[0])
    {
        st= sp[0];
        dt= dp[0];
        Bt= Bp[0];
        return(0);
    }
    if (eta>=etap[np-1])
    {
        st= sp[np-1];
        dt= dp[np-1];
        Bt= Bp[np-1];
        return(0);
    }
    //
    delta= 1.0e40;
    for (i=0; i<np-1; i++)
    {
        if ((fabs(etap[i]-eta)<delta)&&(etap[i]<=eta))
        {
            a= (eta-etap[i])/(etap[i+1]-etap[i]);
            st= (1.0-a)*sp[i]+a*sp[i+1];
            dt= (1.0-a)*dp[i]+a*dp[i+1];
            Bt= (1.0-a)*Bp[i]+a*Bp[i+1];
            delta= fabs(etap[i]-eta);
        }
    }
    if (B01PhotoHadronPFlag>0)
    {
        printf("parameters: %8.6e %8.6e %8.6e %8.6e\n",
        eta/eta0,st,dt,Bt);
    }
    return(0);
}

double B01PhotoHadronP::CalculateP(double eta,double x)
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
        if (B01PhotoHadronPFlag>0)
        printf("x= %8.6e ys= %8.6e t11= %8.6e t1= %8.6e\n",x,ys,t11,t1);
    }
    else if (x>=xps)
    F= 0.0;
    return(F);
}
