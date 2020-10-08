#define B01PhotoHadronGFlag	0
//gamma-rays
class B01PhotoHadronG
{
public:
    B01PhotoHadronG();
    ~B01PhotoHadronG();
    int Test();
    int Init();
    int ReadTable();
    int Prepare(double eta);
    int FindParameters(double eta);
    double CalculateG(double eta,double x);
    //
    double eta0;
private:
    //constants
    double mpi,M,r;
    //variables
    double eta,xp,xm;
    //parameters
    double Bt,st,dt;
    //tables: gamma
    int ng;
    double etag[NH],sg[NH],dg[NH],Bg[NH];
};

B01PhotoHadronG::B01PhotoHadronG()
{
    Init();
    ReadTable();
}

B01PhotoHadronG::~B01PhotoHadronG()
{}

int B01PhotoHadronG::Test()
{
    int i;
    double x,F;
    FILE *fp;
    //
    //define energy and find table values
    //eta= 1.5*eta0;
    //eta= 1.55*eta0;
    //eta= 1.6*eta0;
    //eta= 3.0*eta0;
    //eta= 10.0*eta0;
    eta= 30.0*eta0;
    //eta= 35.0*eta0;
    //eta= 40.0*eta0;
    //eta= 100.0*eta0;
    Prepare(eta);
    FindParameters(eta);
    //
    //fp= fopen("PhotoHadron-Gamma-1.5","w");
    //fp= fopen("PhotoHadron-Gamma-3.0","w");
    //fp= fopen("PhotoHadron-Gamma-10","w");
    fp= fopen("processes/c_codes/PhotoHadron/Data/PhotoHadron-Gamma-30","w");
    if (fp == NULL)
    {
        printf("Couldn't create the PhotoHadron-Gamma-30 file!\n");
        exit(1);
    }
    ////////fp= fopen("PhotoHadron-Gamma-100","w");
    for (i=0; i<10000; i++)
    {
        x= 1.0e-4*(i+1.0);
        F= CalculateG(eta,x);
        fprintf(fp,"%8.6e %8.6e\n",x,x*F);
    }
    fclose(fp);
    return(0);
}

int B01PhotoHadronG::Init()
{
    mpi= mpi0;
    M= mp;
    r= mpi/mp;
    eta0= 2.0*r+r*r;
    if (B01PhotoHadronGFlag>0)
    {
	     printf("mpi= %8.6e mp= %8.6e M= %8.6e r= %8.6e eta0= %8.6e\n", mpi,mp,M,r,eta0);
    }
    return(0);
}

int B01PhotoHadronG::ReadTable()
{
    int i;
    double rd;
    FILE *fp;
    fp = fopen("processes/c_codes/PhotoHadron/Data/Gamma","r");
    if (fp == NULL)
    {
        printf("Couldn't find Data/Gamma!\n");
        exit(1);
    }
    fscanf(fp,"%d",&ng);
    for (i=0; i<ng; i++)
    {
        fscanf(fp,"%lf",&rd); etag[i]= rd;
        fscanf(fp,"%lf",&rd); sg[i] = rd;
        fscanf(fp,"%lf",&rd); dg[i] = rd;
        fscanf(fp,"%lf",&rd); Bg[i] = rd;
    }
    fclose(fp);
    for (i=0; i<ng; i++)
    etag[i]= etag[i]*eta0;
    if (B01PhotoHadronGFlag>0)
    {
        for (i=0; i<ng; i++)
        {
            printf("%8.6e %8.6e %8.6e %8.6e\n",
            etag[i],sg[i],dg[i],Bg[i]);
        }
    }
    return(0);
}

int B01PhotoHadronG::Prepare(double eta)
{
    double xt1,xt2,xt3,xt31,xt32;
    xt1 = 2.0*(1.0+eta);
    xt2 = eta+r*r;
    xt31= eta-r*r-2.0*r;
    xt32= eta-r*r+2.0*r;
    xt3 = xt31*xt32;
    xp= (1.0/xt1)*(xt2+sqrt(xt3));		//x_{+} -> here xp
    xm= (1.0/xt1)*(xt2-sqrt(xt3));		//x_{-} -> here xm
    //
    if (B01PhotoHadronGFlag>0)
    {
        printf("xt1= %8.6e xt2= %8.6e xt3= %8.6e\n",xt1,xt2,xt3);
        printf("xp= %8.6e xm= %8.6e\n",xp,xm);
    }
    return(0);
}

int B01PhotoHadronG::FindParameters(double eta)
{
    int i;
    double a,delta;
    //
    if (eta<=etag[0])
    {
        st= sg[0];
        dt= dg[0];
        Bt= Bg[0];
        return(0);
    }
    if (eta>=etag[ng-1])
    {
        st= sg[ng-1];
        dt= dg[ng-1];
        Bt= Bg[ng-1];
        return(0);
    }
    //
    delta= 1.0e40;
    for (i=0; i<ng-1; i++)
    {
        if ((fabs(etag[i]-eta)<delta)&&(etag[i]<=eta))
        {
            a= (eta-etag[i])/(etag[i+1]-etag[i]);
            st= (1.0-a)*sg[i]+a*sg[i+1];
            dt= (1.0-a)*dg[i]+a*dg[i+1];
            Bt= (1.0-a)*Bg[i]+a*Bg[i+1];
            delta= fabs(etag[i]-eta);
        }
    }
    if (B01PhotoHadronGFlag>0)
    {
        printf("parameters: %8.6e %8.6e %8.6e %8.6e\n",
        eta/eta0,st,dt,Bt);
    }
    return(0);
}

double B01PhotoHadronG::CalculateG(double eta,double x)
{
    double t1,t2,t11,t21,p;
    double y,F;
    //
    p= 2.5+0.4*log(eta/eta0);
    if (x<=xm)
    F= Bt*pow(log(2.0),p);
    else if ((x>xm)&&(x<xp))
    {
        y= (x-xm)/(xp-xm);
        t11= log(x/xm);
        t1= exp(-st*pow(t11,dt));
        t21= log(2.0/(1.0+y*y));
        t2= pow(t21,p);
        F= Bt*t1*t2;
        if (B01PhotoHadronGFlag>0)
        printf("x= %13.6e y= %13.6e t11= %13.6e t1= %13.6e t2= %13.6e\n",x,y,t11,t1,t2);
    }
    else if (x>=xp)
    F= 0.0;
    return(F);
}
