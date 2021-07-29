// Based on M. Kachelrieß et al. (2012)
// Computer Physics Communications 183 (2012) 1036–1043
// http://dx.doi.org/10.1016/j.cpc.2011.12.025
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define ELECTRON_REST_ENERGY 5.1100e+05 // eV
#define THOMSON_CROSS_SECTION 6.6524e-25 // cm^2
#define SIZE_PHOTON_FIELD 6852
#define SIZE_ENERGY 100
//
//
double epsilon[SIZE_PHOTON_FIELD];
double density[SIZE_PHOTON_FIELD];
//
//
void inverse_compton_interaction_rate_internal(char *, double, double, double);
double beta(double E);
double sigma_IC(long double s, long double eps_thr);
void read_photon_field(char *);
double I_gamma(double x);
double r_underint(double s, double E, double eps_thr);
double IC_rate(double E, double E_thr);
//
//
//
void inverse_compton_interaction_rate_internal(char *photon_path, double E_min, double E_max, double E_thr)
{
    FILE * fp;
    read_photon_field(photon_path);
    double energy[SIZE_ENERGY];
    double rate[SIZE_ENERGY];
    // const double E_min = 1.0e+08; // eV
    // const double E_max = 1.0e+16; // eV
    const double aE = log10(E_max/E_min) / (double)SIZE_ENERGY;
    //
    //
    //
    for (int i = 0; i < SIZE_ENERGY; i++)
    {
        energy[i] = E_min * pow(10.0, aE * (double)i);
    }
    for (int i = 0; i < SIZE_ENERGY; i++)
    {
        rate[i] = IC_rate(energy[i], E_thr);
    }
    //
    //
    //
    fp = fopen("processes/c_codes/InverseComptonInteractionRate/output/IC_interaction_rate.txt", "w");
    if (fp == NULL)
    {
        printf("Couldn't create output/... file!\n");
        exit(1);
    }
    for (int i = 0; i < SIZE_ENERGY; i++)
    {
        if (1) //(rate[i] > 0)
        {
            fprintf(fp, "%le    %le\n", energy[i], rate[i]);
        }
    }
    fclose(fp);
    printf("Calculated successfully.\n");
}

double beta(double E)
{
    // There is no 4.0 before ELECTRON_REST_ENERGY*ELECTRON_REST_ENERGY!
    return sqrt(1.0 - (ELECTRON_REST_ENERGY*ELECTRON_REST_ENERGY) / (E * E));
}

double sigma_IC(long double s, long double eps_thr)
{
    long double t11, t12, t2, t3;
    long double ymax, ymin;
    ymin = ELECTRON_REST_ENERGY * ELECTRON_REST_ENERGY / s;
    ymax = 1.0 - eps_thr;
    long double diff = (ymax - ymin);
    long double t_denom = (1.0 - ymin);
    t11 = log(ymax / ymin) / diff;
    t12 = 1.0 - (4.0 * ymin * (1.0 + ymin)) / (t_denom * t_denom);
    t2 = (4.0 * (ymin / ymax + ymin)) / (t_denom * t_denom);
    t3 = (ymax + ymin) / 2.0;
    long double multiplier;
    multiplier = 0.75 * THOMSON_CROSS_SECTION * ymin * diff / t_denom;
    return (multiplier * (t11*t12 + t2 + t3));
}

void read_photon_field(char *path)
{
    FILE * fp;
    fp = fopen(path, "r");
    if (fp == NULL)
    {
        printf("Can't open the file ...\n");
        puts(path);
        exit(1);
    }
    for (int i = 0; i < SIZE_PHOTON_FIELD; i++)
    {
        fscanf(fp, "%le %le", &epsilon[i], &density[i]);
        // if (i % 100 == 0)
        // {
        //     printf("%le %le\n", epsilon[i], density[i]);
        // }
    }
    fclose(fp);
    printf("Photon field has been read successfully.\n");
}

double I_gamma(double x)
{
    int j = 0;
    int flag = -1;
    double y = 0;
    while (epsilon[j] < x)
    {
        j++;
        if (j >= SIZE_PHOTON_FIELD-1)
        {
            // printf("Error! j > SIZE_PHOTON_FIELD\n");
            // exit(1);
            flag = 0;
            break;
        }
    }
    if (flag == 0)
    {
        y = 0;
    }
    else
    {
        double dE;
        double t1, t2;
        for (int i = j; i < SIZE_PHOTON_FIELD-1; i++)
        {
            dE = epsilon[i+1] - epsilon[i];
            t1 = density[i+1] / (epsilon[i+1]*epsilon[i+1]);
            t2 = density[i] / (epsilon[i]*epsilon[i]);
            y += (t1 + t2)/2.0 * dE;
        }
    }
    return y;
}

double r_underint(double s, double E, double eps_thr)
{
    double t1, t2;
    double t_denom;
    t1 = (s - ELECTRON_REST_ENERGY * ELECTRON_REST_ENERGY);
    t_denom = 2.0 * E * (1.0 + beta(E));
    t2 = sigma_IC(s, eps_thr) * I_gamma(t1 / t_denom);
    return (t1 * t2);
}

double IC_rate(double E, double E_thr)
{
    double eps_max = 0;
    for (int i = 0; i < SIZE_PHOTON_FIELD; i++)
    {
        if (eps_max < epsilon[i])
        {
            eps_max = epsilon[i];
        }
    }
    const double s_max = ELECTRON_REST_ENERGY * ELECTRON_REST_ENERGY + 2.0 * E * eps_max * (1.0 + beta(E));
    const double eps_thr = E_thr / E;
    const double s_min = ELECTRON_REST_ENERGY * ELECTRON_REST_ENERGY / (1.0 - eps_thr);
    const unsigned int n_int = 10000;
    double a_s = log10(s_max / s_min) / (double) n_int;
    double s[n_int];
    for (int l = 0; l < n_int; l++)
    {
        s[l] = s_min * pow(10.0, a_s * (double)l);
    }
    //
    double t1, t2;
    double y = 0;
    for (int l = 0; l < n_int - 1; l++)
    {
        t1 = r_underint(s[l+1], E, eps_thr);
        t2 = r_underint(s[l],   E, eps_thr);
        y += (t1 + t2) / 2.0 * (s[l+1] - s[l]);
    }
    return (y / (8.0 * E*E * beta(E)));
}
