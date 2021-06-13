#include "P01Structures.hpp"

#define SIZE_N_PHOTONS_SYNCHRO 100 // !!!
#define SIZE_INDEPENDENT_ELECTRON_ENERGY 50

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
    int Process(char*, char *, double, double, double, double);
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