#define P01PairFlag		0
#define SIZE_N_PHOTONS_SYNCHRO 100 // !!!
#define SIZE_INDEPENDENT_ELECTRON_ENERGY 50
#define SIZE_INTERNAL_INTEGRAL_ELECTRON_ENERGY 50
#define SIZE_OMEGA 50
#define KELNER_KOEFFICIENT 51589733910.0 / 1.21
#define N_PROTON 50
#define PROTON_REST_ENERGY 9.38272e+08 // eV

class P01Pair:public P01Structures
{
  public:
    P01Pair();
    ~P01Pair();
    // synchrotron_field
    double epsilon_synchro[SIZE_N_PHOTONS_SYNCHRO], nph_synchro[SIZE_N_PHOTONS_SYNCHRO];
    // protons
    double energy_proton_min; // eV
    double energy_proton_max; // eV
    double a_p;
    double E_cut; // eV
    double p_p;
    const double E_ref = 1.0e+00; // eV
    // electrons
    double electron_independent_energy_min; // eV
    double electron_independent_energy_max; // eV
    double a_el_independent;
    double Eet[SIZE_INDEPENDENT_ELECTRON_ENERGY];
    double Ee[SIZE_INDEPENDENT_ELECTRON_ENERGY];
    double SED_independent[SIZE_INDEPENDENT_ELECTRON_ENERGY];
    double SED_independent_final[SIZE_INDEPENDENT_ELECTRON_ENERGY];
    // functions
    int Process(char*, double, double, double, double);
    int CalcSpectrum(double gammap_);
    double CalcDelta(double, double); //for one specific energy
    double Sigma(double omega, double Em_, double ksi, double gammap_); //ksi= cos(theta_m)
    void PreparePhotonField(char *);
    double proton_spectrum(double E_p_eV, double p_p, double E_cut_p);
  private:
    // double Ep = -1.0, gammap_ = -1.0; //energy and Lorentz factor of proton
    double Ee_, Em_;
    double Eet_;
    double ect, ec; //E(CMB)= kT
    double omega;
    double pm, ksi;
};

P01Pair::P01Pair()
{}

P01Pair::~P01Pair()
{}

void P01Pair::PreparePhotonField(char * file_path)
{
    FILE * fd;
    FILE * fp;
    fp = fopen("processes/c_codes/PhotoHadron/output/log", "a");
    if (fp == NULL)
    {
        printf("Cannot create log!\n");
    }
    else
    {
        fprintf(fp, "Reading the following photon field:\n");
        fputs(file_path, fp);
        fprintf(fp, "\n");
    }
    fclose(fp);
    //
    fd = fopen(file_path, "r");
    if (fd == NULL)
    {
        printf("Cannot open the file with photon field!\n");
        exit(1);
    }
    // for (int i = 0; i < SIZE_N_PHOTONS_SYNCHRO; i++)
    int i = 0;
    while(!feof(fd))
    {
        fscanf(fd, "%le %le", &epsilon_synchro[i], &nph_synchro[i]);
        epsilon_synchro[i] /= me;
        nph_synchro[i] *= me;
        i++;
    }
    fclose(fd);
    printf("The photon field for BH process has been read successfully.\n");
    i = 0;
    for (int e = 0; e < SIZE_INDEPENDENT_ELECTRON_ENERGY; e++)
    {
        SED_independent_final[e] = 0.0;
    }
}

int P01Pair::Process(char * file_path, double energy_proton_min, double energy_proton_max, double p_p, double E_cut)
{
    FILE * fd;
    double max = 0;
    PreparePhotonField(file_path);
    //
    double Ep[N_PROTON], gammap[N_PROTON];
    a_p = log10(energy_proton_max / energy_proton_min) / (double)N_PROTON;
    for (int i = 0; i < N_PROTON; i++)
    {
        Ep[i] = energy_proton_min * pow(10.0, a_p*i);
        gammap[i] = Ep[i] / mp;
    }
    //
    //
    //
    electron_independent_energy_min = energy_proton_min;
    electron_independent_energy_max = energy_proton_max;
    a_el_independent = log10(electron_independent_energy_max/electron_independent_energy_min) / (double) SIZE_INDEPENDENT_ELECTRON_ENERGY;
    for (int i = 0; i < SIZE_INDEPENDENT_ELECTRON_ENERGY; i++)
    {
        Eet[i]  = electron_independent_energy_min*pow(10.0, a_el_independent*i); //energy of final-state electron in lab. system [eV]
        Ee[i]   = Eet[i]/me; //[eV]->[units of m_e]
    }
    //
    //
    //
    printf("Performing calculations of Bethe-Heitler electron-positron SED...\n");
    // printf("Done, %%:\n");
    for (int u = 1; u < N_PROTON; u++)
    {
        CalcSpectrum(gammap[u]);
        for (int e = 0; e < SIZE_INDEPENDENT_ELECTRON_ENERGY; e++)
        {
            SED_independent_final[e] += (Ep[u] - Ep[u-1]) * proton_spectrum(Ep[u], p_p, E_cut)*SED_independent[e];
        }
        // printf("\r");
        // printf("%2.1lf\n", (double)(u+1) / (double)N_PROTON * 100.0);
    }
    printf("Done!\n");
    fd = fopen("processes/c_codes/BHPairProduction/output/BH_SED.txt", "w");
    max = 0.0;
    for (int i = 0; i < SIZE_INDEPENDENT_ELECTRON_ENERGY; i++)
    {
        if (max < SED_independent_final[i])
        {
            max = SED_independent_final[i];
        }
    }
    for (int i = 0; i < SIZE_INDEPENDENT_ELECTRON_ENERGY; i++)
    {
        // Eet  = electron_independent_energy_min*pow(10.0, a_el_independent*i); //energy of final-state electron in lab. system [eV]
        if (SED_independent_final[i] >= 1.0e-05*max)
        {
            fprintf(fd, "%le %le\n", Eet[i], SED_independent_final[i]); //(Ee,SED)
        }
    }
    fclose(fd);
    return(0);
}

int P01Pair::CalcSpectrum(double gammap_)
{
    double t,Delta;
    double dN;	//dN/dEe
    FILE * fd;
    //
    t = 1.0/(2.0*pow(gammap_,3.0));	//constant factor before the integral (see eq. (62) Kelner & Aharonian 2008)
    for (int i = 0; i < SIZE_INDEPENDENT_ELECTRON_ENERGY; i++)
    {
        //loop on final electron energy (lab. system) --- calculate spectrum (table!)
        // SED_independent[i] = Eet*Eet*dN;		//SED
        Delta = CalcDelta(gammap_, Ee[i]);
        dN = t * Delta; //eq. (67) of Kelner and Aharonian
        SED_independent[i] = Ee[i]*Eet[i]*dN;			//SED
    }
    return(0);
}

double P01Pair::CalcDelta(double gammap_, double Ee_) //for one specific energy
{
    int i,j;
    double omega0,domega;
    double Em1,Em2,dEm;
    double x, y, s, integral = 0;
    double spm;
    double omega_max;
    double a_omega;
    double deps;
    //
    omega0= pow(gammap_+Ee_,2.0)/(2.0*gammap_*Ee_); //low limit for integration on omega
    Em1= (gammap_*gammap_+Ee_*Ee_)/(2.0*gammap_*Ee_); //low limit for integration on E_{-}
    for (int g = 1; g < SIZE_N_PHOTONS_SYNCHRO; g++)
    {
        x= 0.0;
        for (i = 0; i < SIZE_OMEGA; i++)
        {
            //loop on primary photon energy
            // omega_max = 1.0e+08 * omega0;
            omega_max = 2.0*gammap_*epsilon_synchro[g];
            a_omega = log10(omega_max/omega0) / (double) SIZE_OMEGA;
            omega = omega0*pow(10.0, a_omega*i);	//omega = integration variable
            domega = omega0*(pow(10.0, a_omega*(i+1)) - pow(10.0, a_omega*i));
            if (omega < 2.0 * gammap_ * epsilon_synchro[g])
            {
                if (P01PairFlag>0)
                    printf("omega= %13.6e domega= %13.6e\n",omega,domega);
                y = 0.0;
                Em2 = omega-1.0;
                dEm = (Em2-Em1) / (double) SIZE_INTERNAL_INTEGRAL_ELECTRON_ENERGY;
                if (P01PairFlag>0)
                    printf("Em1= %13.6e Em2= %13.6e dEm= %13.6e\n",Em1,Em2,dEm);
                for (j=0; j < SIZE_INTERNAL_INTEGRAL_ELECTRON_ENERGY; j++)
                {
                    //loop on final electron energy (proton restframe)
                    Em_ = Em1 + j*dEm;			//E_{-} = integration variable
                    spm= Em_*Em_ - 1.0;
                    if (spm < SMALLD)
                        spm = SMALLD;
                    pm = sqrt(spm);			    //c = 1
                    if (pm < SMALLD)
                        pm = SMALLD;
                    ksi= (gammap_*Em_-Ee_)/(gammap_*pm);	//cos(theta_m)
                    if (P01PairFlag>0)
                        printf("Em_= %13.6e pm= %13.6e ksi= %13.6e\n",Em_,pm,ksi);
                    s = Sigma(omega, Em_, ksi, gammap_);
                    y+= dEm*s/pm;
                }
                if (P01PairFlag>0)
                    printf("\n");
                // t = omega/(2.0*gammap_*ec);
                // tl= log(1.0-exp(-t));
                // x+= domega*omega*tl*y; this was in eq. (67)
                x += domega*omega*y;
            }
        }
        deps = epsilon_synchro[g] - epsilon_synchro[g-1];
        integral += x * nph_synchro[g] / pow(epsilon_synchro[g], 2) * deps;
    }
    // printf("%le\n", integral);
    integral *= KELNER_KOEFFICIENT;
    return(integral);
}

double P01Pair::Sigma(double omega, double Em_, double ksi, double gammap_)
{
    double yt1,yt2,ytemp,yp;
    double k,pp,Ep,stm; //here Ep= energy of positron
    double sq2,stm2;
    double Deltam,Y,deltapt;
    double kx,ky,kz;
    double pmx,pmy,pmz;
    double t1,t2,t3,t4,t5,t6,t7;
    double t8,t9,t10,t11,t12,t13,t14;
    double tt,sq,spp;
    double Z;
    double s;
    //
    s = 0.0;
    Z = 1.0; //proton
    //positron energy
    Ep= omega-Em_;
    if (Ep<SMALLD) Ep= SMALLD;
    if (P01PairFlag>0)
        printf("Ep= %13.6e Em_= %13.6e omega= %13.6e\n",Ep,Em_,omega);
    //positron momentum
    spp= Ep*Ep-1.0;
    if (spp<SMALLD) spp= SMALLD;
    pp= sqrt(spp);
    if (pp<SMALLD) pp= SMALLD;
    if (P01PairFlag>0)
        printf("pp= %13.6e\n",pp);
    //sin(thetam)
    k= omega;
    if (k<SMALLD) k= SMALLD;
    sq= 1.0-ksi*ksi;
    if (sq<SMALLD) sq= SMALLD;
    stm= sqrt(sq);			//sin(thetam)
    if (fabs(stm)<SMALLD) stm= SMALLD;
    //sin(thetam) (second method)
    sq2= 2.0*gammap_*Em_*Ee_-gammap_*gammap_-Ee_*Ee_;
    if (sq2<SMALLD) sq2= SMALLD;
    stm2= sqrt(sq2)/(gammap_*pm);
    if (fabs(stm2)<SMALLD) stm2= SMALLD;
    if (P01PairFlag>0)
        printf("k= %13.6e ksi= %21.14e stm= %13.6e stm2= %13.6e\n",k,ksi,stm,stm2);
    stm= stm2; //second method preferred
    //
    kx= k;
    ky= 0.0;
    kz= 0.0;
    pmx= pm*ksi;
    pmy= pm*stm;
    pmz= 0.0;
    T= sqrt((kx-pmx)*(kx-pmx)+(ky-pmy)*(ky-pmy)+(kz-pmz)*(kz-pmz));
    if (T<SMALLD) T= SMALLD;
    if (P01PairFlag>0)
        printf("pmx= %13.6e pmy= %13.6e pmz= %13.6e T= %13.6e\n",pmx,pmy,pmz,T);
    //
    tt= T-pp;
    if (tt<SMALLD) tt= SMALLD;
    deltapt= log((T+pp)/tt);
    if (deltapt<SMALLD) deltapt= SMALLD;
    if (P01PairFlag>0)
        printf("deltapt= %13.6e\n",deltapt);
    //
    tt= Ep-pp;
    if (tt<SMALLD) tt= SMALLD;
    ytemp= (Ep+pp)/tt;
    yp= log(ytemp)/pp;
    if (yp<SMALLD) yp= SMALLD;
    if (P01PairFlag>0)
        printf("yp= %13.6e\n",yp);
    //
    yt1= 2.0/(pm*pm);
    yt2= Ep*Em_+pp*pm+1.0;
    Y= yt1*log(yt2/k);
    if (P01PairFlag>0)
        printf("Y= %13.6e\n",Y);
    //
    t1= alpha*Z*Z*r0*r0*pm*pp/(2.0*k*k*k);
    //
    Deltam= Em_-pm*ksi;
    if (fabs(Deltam)<SMALLD) Deltam= SMALLD;
    if (P01PairFlag>0)
        printf("Em_= %13.6e pm= %13.6e ksi= %13.6e Deltam= %21.14e\n",Em_,pm,ksi,Deltam);
    t2= -4.0*stm*stm*(2.0*Em_*Em_+1.0)/(pm*pm*pow(Deltam,4.0));
    //
    t3= (5.0*Em_*Em_-2.0*Ep*Em_+3.0)/(pm*pm*pow(Deltam,2.0));
    t4= (pm*pm-k*k)/(T*T*pow(Deltam,2.0));
    //
    t5= (2.0*Ep)/(pm*pm*Deltam);
    //
    t6= Y/(pm*pp);
    t7= 2.0*Em_*stm*stm*(3.0*k+pm*pm*Ep)/(pow(Deltam,4.0));
    t8= (2.0*Em_*Em_*(Em_*Em_+Ep*Ep)-7.0*Em_*Em_-3.0*Ep*Em_-Ep*Ep+1.0)/(pow(Deltam,2.0));
    t9= k*(Em_*Em_-Em_*Ep-1.0)/Deltam;
    //
    t10= deltapt/(pp*T);
    t11= 2.0/(pow(Deltam,2.0));
    t12= 3.0*k/Deltam;
    t13= k*(pm*pm-k*k)/(T*T*Deltam);
    t14= 2.0*yp/Deltam;
    //
    if (P01PairFlag>0)
    {
        printf("t1= %13.6e t2= %13.6e t3= %13.6e t4= %13.6e\n",t1,t2,t3,t4);
        printf("t5= %13.6e t6= %13.6e t7= %13.6e t8= %13.6e\n",t5,t6,t7,t8);
        printf("t9= %13.6e t10= %13.6e t11= %13.6e t12= %13.6e\n",t9,t10,t11,t12);
        printf("t13= %13.6e t14= %13.6e\n",t13,t14);
    }
    //
    s= t1*(t2+t3+t4+t5+t6*(t7+t8+t9)-t10*(t11-t12-t13)-t14);
    //
    if (P01PairFlag>0)
    {
        printf("alpha= %13.6e Z= %13.6e\n",alpha,Z);
        printf("s= %13.6e\n",s);
    }
    return(s);
}

double P01Pair::proton_spectrum(double E_p_eV, double p_p, double E_cut_p)
{
    if (E_cut_p < 0)
    {
        return pow(E_p_eV/E_ref, -p_p);
    }
    else
    {
        return (pow(E_p_eV/E_ref, -p_p) * exp(-E_p_eV / E_cut_p));
    }
}
