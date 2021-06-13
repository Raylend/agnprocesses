#define B01SSCFlag	0
#define SIZE_N_PHOTONS_SYNCHRO 100 ///!1111111!!!
#define SIZE_X 500
#define SIZE_ENERGY_NEUTRINO 5000
#define SIZE_ENERGY_ELECTRON 5000
#define SIZE_ENERGY_GAMMA 5000
#define N_PROTON 50 //100
#define NEUTRINO_OSCILLATION_FACTOR 0.33333333
#define MINIMAL_FRACTION 1.0e-04
// #define KELNER_KOEFFICIENT 3.229068e-13
#define KELNER_KOEFFICIENT 1.1634328 * 1.3// now CMB normalization is fixed!
#define PROTON_REST_ENERGY 9.38272e+08 // eV

class B01SSC
{
  public:
    B01SSC();
    ~B01SSC();
    //
    double epsilon_synchro[SIZE_N_PHOTONS_SYNCHRO], nph_synchro[SIZE_N_PHOTONS_SYNCHRO];
    //
    // const double E_ref = 1.0e+09; // GeV (1.0e+09)
    const double E_ref = 1.0; // eV
    // const double E_cut = PROTON_REST_ENERGY * (PROTON_REST_ENERGY / (4.0 * 2.326680e-04)) * 0.313 * 0.1;
    // const double E_cut = 0.1 * 3.0e+20; // eV
    double energy_proton_min, energy_proton_max;
    double p_p, E_cut;
    double a_p;
    // double E1, E2;
    //
    const double a_frac = log10(1.0/MINIMAL_FRACTION) / (double) SIZE_X;
    //
    //
    double SED_neutrino_init[SIZE_X][N_PROTON];
    double SED_neutrino_intermediate[SIZE_ENERGY_NEUTRINO][N_PROTON];
    double SED_neutrino_final[SIZE_ENERGY_NEUTRINO], energy_neutrino_final[SIZE_ENERGY_NEUTRINO];
    //
    double SED_electron_init[SIZE_X][N_PROTON];
    double SED_electron_intermediate[SIZE_ENERGY_ELECTRON][N_PROTON];
    double SED_electron_final[SIZE_ENERGY_ELECTRON], energy_electron_final[SIZE_ENERGY_ELECTRON];
    //
    double SED_gamma_init[SIZE_X][N_PROTON];
    double SED_gamma_intermediate[SIZE_ENERGY_GAMMA][N_PROTON];
    double SED_gamma_final[SIZE_ENERGY_GAMMA], energy_gamma_final[SIZE_ENERGY_GAMMA];
    //
    double energy_proton[N_PROTON];
    //
    //
    void PrepareSSC(char *);
    int Process(char* file_path, double energy_proton_min_ext, double energy_proton_max_ext, double p_p_ext, double E_cut_ext);
    int Integrate(int proton_energy_number);
    // void read_protons_parameters();
    double proton_spectrum(double E_p_eV, double p_p, double E_cut_p);
    void PrepareSEDIntermediate();
    void FinalSED();
    //
    FILE *fpa;
    FILE *fpn;
    //
    B01PhotoHadronG phg;
    B01PhotoHadronP php;
    B01PhotoHadronE phe;
    B01PhotoHadronNuMu phnm;
    B01PhotoHadronAntiNuMu phanm;
    B01PhotoHadronNuE phne;
    B01PhotoHadronAntiNuE phane;
};

B01SSC::B01SSC()
{}

B01SSC::~B01SSC()
{}

void B01SSC::PrepareSSC(char * file_path)
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
    // fd = fopen("processes/c_codes/PhotoHadron/input/plank_CMB_for_Kelner.txt", "r");
    fd = fopen(file_path, "r");
    if (fd == NULL)
    {
        printf("Cannot open the file with photon field!\n");
        exit(1);
    }
    else
    {
        printf("The photon field file has been read successfully.\n");
    }
    for (int i = 0; i < SIZE_N_PHOTONS_SYNCHRO; i++)
    {
        fscanf(fd, "%le %le", &epsilon_synchro[i], &nph_synchro[i]);
    }
    fclose(fd);
}

int B01SSC::Process(char* file_path, double energy_proton_min_ext, double energy_proton_max_ext, double p_p_ext, double E_cut_ext)
{
    printf("Performing calculations of proton-photon meson production secondaries SED...\n");
    energy_proton_min = energy_proton_min_ext;
    energy_proton_max = energy_proton_max_ext;
    p_p = p_p_ext;
    E_cut = E_cut_ext;
    B01SSC::PrepareSSC(file_path);
    //
    //
    const double a_p = log10(energy_proton_max / energy_proton_min) / (double)N_PROTON;
    //
    //
    const double energy_neutrino_min = 1.0e+12; //eV
    const double energy_neutrino_max = energy_proton_max * 0.75; // eV
    const double a_neu = log10(energy_neutrino_max/energy_neutrino_min)/(double)SIZE_ENERGY_NEUTRINO;
    //
    const double energy_electron_min = 1.0e+12; // eV
    const double energy_electron_max = energy_proton_max * 0.75; // eV
    const double a_e = log10(energy_electron_max/energy_electron_min)/(double)SIZE_ENERGY_ELECTRON;
    //
    const double energy_gamma_min = 1.0e+12; // eV
    const double energy_gamma_max = energy_proton_max * 0.75; // eV
    const double a_g = log10(energy_gamma_max/energy_gamma_min)/(double)SIZE_ENERGY_GAMMA;
    // neutrinos
    for (int k = 0; k < SIZE_ENERGY_NEUTRINO; k++)
    {
        SED_neutrino_final[k] = 0.0;
    }
    for (int k = 0; k < SIZE_ENERGY_NEUTRINO; k++)
    {
        energy_neutrino_final[k] = energy_neutrino_min*pow(10.0, a_neu*k);
    }
    // electrons (+ positrons)
    for (int k = 0; k < SIZE_ENERGY_ELECTRON; k++)
    {
        SED_electron_final[k] = 0.0;
    }
    for (int k = 0; k < SIZE_ENERGY_ELECTRON; k++)
    {
        energy_electron_final[k] = energy_electron_min*pow(10.0, a_e*k);
    }
    // gamma-rays
    for (int k = 0; k < SIZE_ENERGY_GAMMA; k++)
    {
        SED_gamma_final[k] = 0.0;
    }
    for (int k = 0; k < SIZE_ENERGY_GAMMA; k++)
    {
        energy_gamma_final[k] = energy_gamma_min*pow(10.0, a_g*k);
    }
    // primary protons
    for (int d= 0; d < N_PROTON; d++)
    {
        energy_proton[d] = energy_proton_min*pow(10.0, a_p*d);
    }
    //
    for (int i = 0; i < N_PROTON; i++)
    {
        B01SSC::Integrate(i);
        // printf("%d\n", i);
    }
    PrepareSEDIntermediate();
    FinalSED();
    //
    printf("Done!\n");
    return(0);
}

int B01SSC::Integrate(int proton_energy_number)
{
    int i, j, flag1, flag2, antiflag1, antiflag2, el_flag, pos_flag;
    double pi,GeV;
    double eta0,eta;
    double eps0,eps;
    double x,f,F;
    double sg,sp,se;
    double snm,sanm,sne,sane;
    double deps;
    // double x1;
    // double x_mean = 0;
    double Ep = energy_proton[proton_energy_number]/1.0e+09; // eV -> GeV
    double temp;
    //
    pi = 3.141592654;
    GeV= 1.0e-9;				//[eV]
    //
    eta0= phg.eta0;
    eps0= 1.0e9*(eta0*mp*mp)/(4.0*Ep);		//energy threshold [eV]
    //
    if (B01SSCFlag>0)
        printf("eta0= %13.6e eps0= %13.6e [eV]\n",eta0,eps0);
    //
    // x1 = MINIMAL_FRACTION;
    for (j=1; j < SIZE_X; j++)
    {
        sg  = 0.0;
        sp  = 0.0;
        se  = 0.0;
        snm = 0.0;
        sanm= 0.0;
        sne = 0.0;
        sane= 0.0;
        // do not change 1.0e-4 without changing it in other places! (we need to parametrize it ASAP)
        x = MINIMAL_FRACTION*pow(10.0,a_frac*(double)j); //x= (E_{gamma}/E_{p}); (E_e/E_{p}), (...)
        for (i=1; i<SIZE_N_PHOTONS_SYNCHRO; i++)
        //integrate over photon field energy eps: SIZE_N_PHOTONS_SYNCHRO different values of eps
        {
            if (epsilon_synchro[i] > eps0)
            {
                // CHANGE ! eps= eps0+deps*i;			//photon energy [eV]
                //f= (1.0/(pi*pi))*(eps*eps)/(exp(eps/T)-1.0); //number density of photon field [1/(cm^{3}*eV)]
                eps = epsilon_synchro[i];
                f = nph_synchro[i]; //number density of photon field [1/(cm^{3}*eV)]
                //insert new f (number density) here
                eta= (4.0*GeV*eps*Ep)/(mp*mp);		 //approximation parameter: Phi(eta,x) -> here F(eta,x)
                // if (eta > 30.0)
                // {
                //     printf("eta = %le\n", eta);
                //     printf("energy_proton = %le\n", energy_proton[proton_energy_number]);
                // }
                deps = epsilon_synchro[i] - epsilon_synchro[i-1];
                //gamma
                phg.Prepare(eta);
                phg.FindParameters(eta);
                F= phg.CalculateG(eta,x);
                sg+= f*F*deps;
                //positron
                pos_flag = php.Prepare(eta);
                if (pos_flag == 0)
                {
                    php.FindParameters(eta);
                    F= php.CalculateP(eta,x);
                    sp+= f*F*deps;
                }
                //electron
                el_flag = phe.Prepare(eta);
                if (el_flag==0)
                {
                    phe.FindParameters(eta);
                    F= phe.CalculateE(eta,x);
                    se+= f*F*deps;
                }
                //nu_mu
                flag1 = phnm.Prepare(eta);
                if (flag1 == 0)
                {
                    phnm.FindParameters(eta);
                    F= phnm.CalculateNuMu(eta,x);
                    snm+= f*F*deps;
                }
                //anti-nu_mu
                antiflag1 = phanm.Prepare(eta);
                if (antiflag1 == 0)
                {
                    phanm.FindParameters(eta);
                    F= phanm.CalculateAntiNuMu(eta,x);
                    sanm+= f*F*deps;
                }
                //nu_e
                flag2 = phne.Prepare(eta);
                if (flag2 == 0)
                {
                    phne.FindParameters(eta);
                    F= phne.CalculateNuE(eta,x);
                    sne+= f*F*deps;
                }
                //anti-nu_e
                antiflag2 = phane.Prepare(eta);
                if (antiflag2 == 0)
                {
                    phane.FindParameters(eta);
                    F= phane.CalculateAntiNuE(eta,x);
                    sane+= f*F*deps;
                }
                //
                if (B01PlanckianCMBFlag>0)
                {
                    printf("n= %8d eps= %13.6e f= %13.6e eta= %13.6e x= %13.6e F= %13.6e\n",
                    i,eps,f,eta,x,F);
                }
            }
        }
        // x_mean += x*(sg + sp + se + snm + sanm + sne + sane)*(x - x1);
        // x1 = x;
        // fprintf(fpa,"%13.6e  %13.6e %13.6e %13.6e\n", x, x*sg, x*sp, x*se);
        // fprintf(fpn,"%13.6e  %13.6e %13.6e %13.6e %13.6e\n", x, x*snm, x*sanm, x*sne, x*sane);
        temp = x*x*energy_proton[proton_energy_number];//*1.0e+09;
        SED_electron_init[j][proton_energy_number] = temp*(sp + se) * KELNER_KOEFFICIENT;
        SED_neutrino_init[j][proton_energy_number] = temp*(snm + sanm + sne + sane) * KELNER_KOEFFICIENT;
        SED_gamma_init   [j][proton_energy_number] = temp*sg * KELNER_KOEFFICIENT;
    }
    // fprintf(fpa,"\n");
    // fprintf(fpn,"\n");
    // printf("x_mean = %le\n", x_mean);
    return(0);
}

void B01SSC::PrepareSEDIntermediate()
{
    double x;
    int k;
    // neutrinos
    for (int j = 0; j < N_PROTON; j++)
    {
        for (int l = 0; l < SIZE_ENERGY_NEUTRINO; l++)
        {
            x = energy_neutrino_final[l] / energy_proton[j];// * 1.0e-09;
            k = (int) (log10(x/MINIMAL_FRACTION)/a_frac);
            if ((k>0)&&(k<SIZE_ENERGY_NEUTRINO))
            {
                SED_neutrino_intermediate[l][j] = SED_neutrino_init[k][j];
            }
        }
    }
    // electrons (+ positrons)
    for (int j = 0; j < N_PROTON; j++)
    {
        for (int l = 0; l < SIZE_ENERGY_ELECTRON; l++)
        {
            x = energy_electron_final[l] / energy_proton[j];// * 1.0e-09;
            k = (int) (log10(x/MINIMAL_FRACTION)/a_frac);
            if ((k>0)&&(k<SIZE_ENERGY_ELECTRON))
            {
                SED_electron_intermediate[l][j] = SED_electron_init[k][j];
            }
        }
    }
    // gamma-rays
    for (int j = 0; j < N_PROTON; j++)
    {
        for (int l = 0; l < SIZE_ENERGY_GAMMA; l++)
        {
            x = energy_gamma_final[l] / energy_proton[j];// * 1.0e-09;
            k = (int) (log10(x/MINIMAL_FRACTION)/a_frac);
            if ((k>0)&&(k<SIZE_ENERGY_GAMMA))
            {
                SED_gamma_intermediate[l][j] = SED_gamma_init[k][j];
            }
        }
    }
}

void B01SSC::FinalSED()
{
    FILE * fp;
    //
    // neutrinos
    for (int l = 0; l < SIZE_ENERGY_NEUTRINO; l++)
    {
        for (int j = 1; j < N_PROTON; j++)
        {
            SED_neutrino_final[l] += SED_neutrino_intermediate[l][j]*B01SSC::proton_spectrum(energy_proton[j], p_p, E_cut)*(energy_proton[j] - energy_proton[j-1]);
        }
    }
    double max = 0;
    for (int l = 0; l < SIZE_ENERGY_NEUTRINO; l++)
    {
        SED_neutrino_final[l] = SED_neutrino_final[l]*NEUTRINO_OSCILLATION_FACTOR;
        if (max < SED_neutrino_final[l])
        {
            max = SED_neutrino_final[l];
        }
    }
    fp = fopen("processes/c_codes/PhotoHadron/output/neutrino_SED.txt", "w");
    if (fp == NULL)
    {
        printf("Cannot create the file!\n");
        exit(1);
    }
    for (int l = 0; l < SIZE_ENERGY_NEUTRINO; l++)
    {
        if (SED_neutrino_final[l] >= 1.0e-04*max)
        {
            fprintf(fp, "%le  %le\n", energy_neutrino_final[l], SED_neutrino_final[l]);
        }
    }
    fclose(fp);
    //
    // electrons (+ positrons)
    for (int l = 0; l < SIZE_ENERGY_ELECTRON; l++)
    {
        for (int j = 1; j < N_PROTON; j++)
        {
            SED_electron_final[l] += SED_electron_intermediate[l][j]*B01SSC::proton_spectrum(energy_proton[j], p_p, E_cut)*(energy_proton[j] - energy_proton[j-1]);
        }
        // for (int j = 0; j < N_PROTON-1; j++)
        // {
        //     SED_electron_final[l] += SED_electron_intermediate[l][j]*proton_spectrum(energy_proton[j], p_p)*(energy_proton[j]);// - energy_proton[j-1]);
        // }
    }
    max = 0;
    for (int l = 0; l < SIZE_ENERGY_ELECTRON; l++)
    {
        if (max < SED_electron_final[l])
        {
            max = SED_electron_final[l];
        }
    }
    fp = fopen("processes/c_codes/PhotoHadron/output/electron_SED.txt", "w");
    if (fp == NULL)
    {
        printf("Cannot create the file!\n");
        exit(1);
    }
    for (int l = 0; l < SIZE_ENERGY_ELECTRON; l++)
    {
        if (SED_electron_final[l] >= 1.0e-04*max)
        {
            fprintf(fp, "%le  %le\n", energy_electron_final[l], SED_electron_final[l]);
        }
    }
    fclose(fp);
    //
    // gamma-rays
    for (int l = 0; l < SIZE_ENERGY_GAMMA; l++)
    {
        for (int j = 1; j < N_PROTON; j++)
        {
            SED_gamma_final[l] += SED_gamma_intermediate[l][j]*B01SSC::proton_spectrum(energy_proton[j], p_p, E_cut)*(energy_proton[j] - energy_proton[j-1]);
        }
    }
    max = 0;
    for (int l = 0; l < SIZE_ENERGY_GAMMA; l++)
    {
        if (max < SED_gamma_final[l])
        {
            max = SED_gamma_final[l];
        }
    }
    fp = fopen("processes/c_codes/PhotoHadron/output/gamma_SED.txt", "w");
    if (fp == NULL)
    {
        printf("Cannot create the file!\n");
        exit(1);
    }
    for (int l = 0; l < SIZE_ENERGY_GAMMA; l++)
    {
        if (SED_gamma_final[l] >= 1.0e-04*max)
        {
            fprintf(fp, "%le  %le\n", energy_gamma_final[l], SED_gamma_final[l]);
        }
    }
    fclose(fp);
}

double B01SSC::proton_spectrum(double E_p_eV, double p_p, double E_cut_p)
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
