#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ELECTRON_REST_ENERGY 5.1100e+05 // eV
#define THOMSON_CROSS_SECTION 6.6524e-25 // cm^2
#define SPEED_OF_LIGHT 2.998e+10 // cm/s
#define CM_TO_EV  5.068e+04
#define SIZE_PHOTON_FIELD 6852
#define CONST_GAMMA_VHE 5000
#define SIZE_ENERGY 100
#define SMALLD 1.0e-40
// #define ENERGY_GAMMA_VHE_THRESHOLD 3.0e+10/ELECTRON_REST_ENERGY // eV
#define ENERGY_GAMMA_VHE_THRESHOLD SMALLD //3.0e+08/ELECTRON_REST_ENERGY // eV
//
//
int SIZE_GAMMA_VHE;
//
//
void pair_calculate_internal(char *photon_file, char *gamma_file);
void read_photon_field(char *photon_file);
void read_gamma_VHE(char *gamma_file);
double r_function(double, double);
double soft_underint(double n0, double eps, double e_gamma, double E_star, double r);
double hard_underint(double e_gamma, double e_e);
double pair_distribution(double e_e);
//
//
double epsilon[SIZE_PHOTON_FIELD];
double density[SIZE_PHOTON_FIELD];
double energy_gamma[CONST_GAMMA_VHE];
double SED_gamma[CONST_GAMMA_VHE];
double spectrum_gamma[CONST_GAMMA_VHE];
double E_min; // eV
double E_max; // eV
double aE;
//
//
//
void pair_calculate_internal(char *photon_file, char *gamma_file)
{
    FILE * fp;
    read_photon_field(photon_file);
    double dE;
    double energy_pair_VHE[SIZE_ENERGY];
    double SED_pair_VHE[SIZE_ENERGY];
    read_gamma_VHE(gamma_file);
    //
    //
    //
    // printf("Done, %%:\n");
    for (int i = 0; i < SIZE_ENERGY; i++)
    {
        // printf("%.1lf\n", (double)(i + 1) / (double)SIZE_ENERGY * 100.0);
        energy_pair_VHE[i] = E_min * pow(10.0, aE * (double)i);
        SED_pair_VHE[i] = pair_distribution(energy_pair_VHE[i] / ELECTRON_REST_ENERGY);
        SED_pair_VHE[i] *= (energy_pair_VHE[i] * energy_pair_VHE[i]);
        // printf("%le\n", SED_pair_VHE[i]);
    }
    //
    //
    // Derive full energy_pair_VHE in gamma-rays
    double full_gamma1 = 0.0;
    for (int i = 0; i < SIZE_GAMMA_VHE - 1; i++)
    {
        dE = (energy_gamma[i+1] - energy_gamma[i]) * ELECTRON_REST_ENERGY;
        if (energy_gamma[i] >= ENERGY_GAMMA_VHE_THRESHOLD)
        {
            full_gamma1 += dE * energy_gamma[i] * spectrum_gamma[i];
        }
    }
    printf("full_gamma1 = %le eV\n", full_gamma1);
    //
    // Derive full yet wrong energy_pair_VHE in pairs
    double full_pair1 = 0.0;
    for (int i = 0; i < SIZE_ENERGY - 1; i++)
    {
        dE = (energy_pair_VHE[i+1] - energy_pair_VHE[i]);
        full_pair1 += dE * SED_pair_VHE[i] / energy_pair_VHE[i];
    }
    printf("full_pair1 = %le\n", full_pair1);
    //
    //
    // Correction of the pair normalization
    double correction1 = full_gamma1 / full_pair1;
    for (int i = 0; i < SIZE_ENERGY; i++)
    {
        SED_pair_VHE[i] *= correction1;
    }
    //
    //
    // Get maximum value of SED_pair_VHE
    double temp1 = SMALLD;
    for (int i = 0; i < SIZE_ENERGY; i++)
    {
        if (temp1 < SED_pair_VHE[i])
        {
            temp1 = SED_pair_VHE[i];
        }
    }
    // Printing results
    fp = fopen("processes/c_codes/GammaGammaPairProduction/output/SED_gamma-gamma_pairs.txt", "w");
    if (fp == NULL)
    {
        printf("Couldn't create SED_gamma-gamma_pairs.txt!\n");
        exit(1);
    }
    for (int i = 0; i < SIZE_ENERGY; i++)
    {
        if (SED_pair_VHE[i] >= 1.0e-05 * temp1)
        {
            fprintf(fp, "%le    %le\n", energy_pair_VHE[i], SED_pair_VHE[i]);
        }
    }
    fclose(fp);
    printf("Gamma-gamma pair production calculated successfully.\n");
}

void read_photon_field(char *photon_file)
{
    FILE * fp;
    fp = fopen(photon_file, "r");
    if (fp == NULL)
    {
        printf("Can't open the file data/...\n");
        puts(photon_file);
        exit(1);
    }
    for (int i = 0; i < SIZE_PHOTON_FIELD; i++)
    {
        if (!feof(fp))
        {
            fscanf(fp, "%le %le", &epsilon[i], &density[i]);
            epsilon[i] /= ELECTRON_REST_ENERGY;
            density[i] *= ELECTRON_REST_ENERGY;
        }
        else
        {
            break;
        }
    }
    fclose(fp);
    printf("Photon field has been read successfully.\n");
}

void read_gamma_VHE(char *gamma_file)
{
    int del = 1;
    FILE * fp;
    double rd1, rd2;
    fp = fopen(gamma_file, "r");
    if (fp == NULL)
    {
        printf("Can't read VHE gamma-ray file!\n");
        puts(gamma_file);
        exit(1);
    }
    int j = 0;
    for (int i = 0; i < 5000; i++)
    {
        if (!feof(fp))
        {
            fscanf(fp, "%le %le", &rd1, &rd2);
            if ((i % del) == 0)
            {
                j = i / del;
                // printf("j = %d\n", j);
                spectrum_gamma[j] = rd2 / (rd1 * rd1) * ELECTRON_REST_ENERGY;
                energy_gamma[j] = rd1 / ELECTRON_REST_ENERGY;
                SED_gamma[j] = rd2;
                printf("%le %le\n", energy_gamma[j], SED_gamma[j]);
            }
        }
        else
        {
            break;
        }
    }
    fclose(fp);
    SIZE_GAMMA_VHE = j;
    E_min = 1.0e+07; // eV
    // E_max = energy_gamma[SIZE_GAMMA_VHE - 1] * ELECTRON_REST_ENERGY; // eV
    E_max = 3.0e+11; // eV
    aE = log10(E_max/E_min) / (double)SIZE_ENERGY;
    printf("Gamma-ray SED has been read successfully.\n");
}

double r_function(double gamma, double gamma_)
{
    double ans;
    if ((gamma <  SMALLD) || (gamma_ < SMALLD))
    {
        // printf("(gamma <  SMALLD) || (gamma_ < SMALLD)\n");
        // printf("gamma = %le, gamma_ = %le", gamma, gamma_);
        // exit(1);
        ans = 0;
    }
    else
    {
        ans = (0.5 * (gamma / gamma_ + gamma_ / gamma));
    }
    return ans;
}

double soft_underint(double n0, double eps, double e_gamma, double E_star, double r)
{
    double E = e_gamma * eps;
    double ratio = E_star / E;
    double a1, a2, a3;
    double multiplier;
    //
    //
    a1 = r - (2.0 + r) * ratio;
    a2 = 2.0 * ratio * ratio;
    a3 = -2.0 * ratio * log(ratio);
    multiplier = n0 / (E * e_gamma);
    // multiplier *= 0.75 * THOMSON_CROSS_SECTION * SPEED_OF_LIGHT;
    multiplier *= (a1 + a2 + a3);
    if (multiplier < SMALLD)
    {
        multiplier = 0.0;
    }
    return multiplier;
}

double hard_underint(double e_gamma, double e_e)
{
    double answer;
    if ((e_e > e_gamma) || (e_gamma < ENERGY_GAMMA_VHE_THRESHOLD))
    {
        answer = 0.0;
    }
    else
    {
        double e_e_primed = e_gamma - e_e;
        double r = r_function(e_e, e_e_primed);
        double E_star = (e_gamma * e_gamma) / (4.0 * e_e * e_e_primed);
        double soft_int_min = E_star / e_gamma;
        double t1, t2;
        double I_internal = 0;
        //
        //
        for (int i = 0; i < SIZE_PHOTON_FIELD - 1; i++)
        {
            t1 = soft_underint(density[i], epsilon[i], e_gamma, E_star, r);
            t2 = soft_underint(density[i + 1], epsilon[i + 1], e_gamma, E_star, r);
            I_internal += (epsilon[i + 1] - epsilon[i]) * ((t1 + t2) / 2.0);
        }
        answer = I_internal;
    }
    return answer;
}

double pair_distribution(double e_e)
{
    double distrib = 0;
    double t;
    for (int i = 0; i < SIZE_GAMMA_VHE; i++)
    {
        t = spectrum_gamma[i] * hard_underint(energy_gamma[i], e_e);
        distrib += (energy_gamma[i+1] - energy_gamma[i]) * t;
    }
    return distrib * 2.0;
}
