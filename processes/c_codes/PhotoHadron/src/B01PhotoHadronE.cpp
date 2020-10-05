#define B01PhotoHadronEFlag	0
//electrons
class B01PhotoHadronE
{
public:
    B01PhotoHadronE();
    ~B01PhotoHadronE();
    int Test();
    int Init();
    int ReadTable();
    int Prepare(double eta);
    int FindParameters(double eta);
    double CalculateE(double eta,double x);
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
    int ne;
    double etae[NH],se[NH],de[NH],Be[NH];
};

B01PhotoHadronE::B01PhotoHadronE()
{
    Init();
    ReadTable();
}

B01PhotoHadronE::~B01PhotoHadronE()
{}

int B01PhotoHadronE::Test()
{
    int i;
    double x,F;
    FILE *fp;
    //
    //eta= 3.0*eta0;
    //eta= 5.0*eta0;
    //eta= 10.0*eta0;
    eta= 30.0*eta0;
    //eta= 100.0*eta0;
    Prepare(eta);
    FindParameters(eta);
    //
    //fp= fopen("PhotoHadron-Electron-3.0","w");
    //fp= fopen("PhotoHadron-Electron-5.0","w");
    //fp= fopen("PhotoHadron-Electron-10","w");
    fp= fopen("PhotoHadron-Electron-30","w");
    //fp= fopen("PhotoHadron-Electron-100","w");
    for (i=0; i<10000; i++)
    {
        x= 1.0e-4*(i+1.0);
        F= CalculateE(eta,x);
        fprintf(fp,"%8.6e %8.6e\n",x,x*F);
    }
    fclose(fp);
    return(0);
}

int B01PhotoHadronE::Init()
{
    mpi= mpip;
    M= mp;
    r= mpi/mp;
    R= M/mp;
    eta0= 2.0*r+r*r;
    if (B01PhotoHadronEFlag>0)
    {
        printf("mpi= %8.6e mp= %8.6e M= %8.6e r= %8.6e R= %8.6e eta0= %8.6e\n",
        mpi,mp,M,r,R,eta0);
    }
    return(0);
}

int B01PhotoHadronE::ReadTable()
{
    int i;
    double rd;
    FILE *fp;
    fp= fopen("Data/Electron","r");
    fscanf(fp,"%d",&ne);
    for (i=0; i<ne; i++)
    {
        fscanf(fp,"%lf",&rd); etae[i]= rd;
        fscanf(fp,"%lf",&rd); se[i] = rd;
        fscanf(fp,"%lf",&rd); de[i] = rd;
        fscanf(fp,"%lf",&rd); Be[i] = rd;
    }
    fclose(fp);
    for (i=0; i<ne; i++)
    etae[i]= etae[i]*eta0;
    if (B01PhotoHadronEFlag>0)
    {
        for (i=0; i<ne; i++)
        {
            printf("%8.6e %8.6e %8.6e %8.6e\n",
            etae[i],se[i],de[i],Be[i]);
        }
    }
    return(0);
}

int B01PhotoHadronE::Prepare(double eta)
{
    double xt1,xt2,xt3;
    double xmin,xmax;
    int f = 0;
    xt1 = 2.0*(1.0+eta);
    xt2 = eta-2.0*r;
    xt3 = eta*(eta-4.0*r*(1.0+r));
    xmax= 0.0;
    xmin= 0.0;
    if (xt3>SMALLD)
    {
        xmax= (1.0/xt1)*(xt2+sqrt(xt3));
        xmin= (1.0/xt1)*(xt2-sqrt(xt3));
    }
    else
    f = 1;
    xps= xmax;
    xms= xmin/2.0;
    //
    if (B01PhotoHadronEFlag>0)
    {
        printf("xt1= %8.6e xt2= %8.6e xt3= %8.6e\n",xt1,xt2,xt3);
        printf("xp= %8.6e xm= %8.6e\n",xp,xm);
    }
    return(f);
}

int B01PhotoHadronE::FindParameters(double eta)
{
    int i;
    double a,delta,rho,rho0;
    //
    if (eta<=etae[0])
    {
        st= se[0];
        dt= de[0];
        rho= eta/eta0;
        rho0= etae[0]/eta0;
        Bt= Be[0]*(rho-2.14)/(rho0-2.14);
        return(0);
    }
    if (eta>=etae[ne-1])
    {
        st= se[ne-1];
        dt= de[ne-1];
        Bt= Be[ne-1];
        return(0);
    }
    //
    delta= 1.0e40;
    for (i=0; i<ne-1; i++)
    {
        if ((fabs(etae[i]-eta)<delta)&&(etae[i]<=eta))
        {
            a= (eta-etae[i])/(etae[i+1]-etae[i]);
            st= (1.0-a)*se[i]+a*se[i+1];
            dt= (1.0-a)*de[i]+a*de[i+1];
            Bt= (1.0-a)*Be[i]+a*Be[i+1];
            delta= fabs(etae[i]-eta);
        }
    }
    if (B01PhotoHadronEFlag>0)
    {
        printf("parameters: %8.6e %8.6e %8.6e %8.6e\n",
        eta/eta0,st,dt,Bt);
    }
    return(0);
}

double B01PhotoHadronE::CalculateE(double eta,double x)
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
        if (B01PhotoHadronEFlag>0)
        printf("x= %8.6e ys= %8.6e t11= %8.6e t1= %8.6e\n",x,ys,t11,t1);
    }
    else if (x>=xps)
        F= 0.0;
    if (rho<2.14) F= 0.0;
    return(F);
}
