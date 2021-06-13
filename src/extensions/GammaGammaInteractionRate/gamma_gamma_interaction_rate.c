#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#include <omp.h>
#include <string.h>

#define ELECTRON_REST_ENERGY 5.1100e+05 // eV
#define THOMSON_CROSS_SECTION 6.6524e-25 // cm^2
#define CM_TO_EV  5.068e+04
#define SIZE_PHOTON_FIELD 1000
#define SIZE_ENERGY 100
//
//
const double s_min = 4.0*ELECTRON_REST_ENERGY*ELECTRON_REST_ENERGY;
double epsilon[SIZE_PHOTON_FIELD];
double density[SIZE_PHOTON_FIELD];
//
//
void gamma_gamma_interaction_rate_internal(char *, double, double);
double beta(double s);
double sigma_pair(double s);
void read_photon_field(char *);
double I_gamma(double x);
double r_underint(double s, double E);
double gamma_gamma_rate(double E);
//
//
//
void gamma_gamma_interaction_rate_internal(char *photon_path, double E_min, double E_max)
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
        rate[i] = gamma_gamma_rate(energy[i]);
    }
    //
    //
    //
    fp = fopen("processes/c_codes/GammaGammaInteractionRate/output/gamma-gamma_interaction_rate.txt", "w");
    if (fp == NULL)
    {
        printf("Couldn't create output/... file!\n");
        exit(1);
    }
    for (int i = 0; i < SIZE_ENERGY; i++)
    {
        // if (i % 10 == 0)
        // {
        //     printf("%le\n", rate[i]);
        // }
        if (rate[i] > 0)
        {
            fprintf(fp, "%le    %le\n", energy[i], rate[i]);
        }
    }
    fclose(fp);
    printf("Calculated successfully.\n");
}

double beta(double s)
{
    return sqrt(1.0 - (4.0*ELECTRON_REST_ENERGY*ELECTRON_REST_ENERGY)/s);
}

double sigma_pair(double s)
{
    double t1, t2, t3;
    double b = beta(s);
    t1 = 0.75*THOMSON_CROSS_SECTION*ELECTRON_REST_ENERGY*ELECTRON_REST_ENERGY/s;
    t2 = (3.0 - pow(b, 4)) * log((1.0 + b)/(1.0 - b));
    t3 = -2.0 * b * (2.0 - b*b);
    return(t1 * (t2 + t3));
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

double r_underint(double s, double E)
{
    double t1, t2;
    t1 = s * sigma_pair(s);
    t2 = I_gamma(s / (4.0*E));
    return(t1*t2);
}

double gamma_gamma_rate(double E)
{
    double eps_max = 0;
    for (int i = 0; i < SIZE_PHOTON_FIELD; i++)
    {
        if (eps_max < epsilon[i])
        {
            eps_max = epsilon[i];
        }
    }
    double s_max = 4.0*E*eps_max;
    const unsigned int n_int = 1000;
    double a_s = log10(s_max/s_min) / (double) n_int;
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
        t1 = r_underint(s[l+1], E);
        t2 = r_underint(s[l],   E);
        y += (t1 + t2)/2.0 * (s[l+1] - s[l]);
    }
    return (y / (8.0 * E*E));
}
