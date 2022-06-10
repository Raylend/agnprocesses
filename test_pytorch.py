from math import log10
import torch
import numpy as np
import agnprocesses.ic as ic
import agnprocesses.gamma_gamma as gamma_gamma
import agnprocesses.spectra as spec
from astropy import units as u
from astropy import constants as const
import time
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt
import torchinterp1d
from functools import partial
import warnings
from scipy.integrate import simps
import os
import subprocess

# if __name__ == '__main__':
ELECTRON_REST_ENERGY = const.m_e.to(u.eV, u.mass_energy()).value
S_MIN = (2.0 * ELECTRON_REST_ENERGY)**2
print('Hey!')
# device = 'cpu'  # 'cuda'
# # if False:
if torch.cuda.is_available():
    print("If cuda is available: ", torch.cuda.is_available())
    device = 'cuda'
    print("Using 'cuda'!!!")
    print(torch.cuda.get_device_name())
else:
    device = 'cpu'
    print("Using CPU!!!")
########################################################################
# #0 Precalculate gamma-gamma interaction rate
# r_blr = ((0.036 * u.pc).to(u.cm)).value  # BLR radius in cm
# print("r_blr = {:.3e}".format(r_blr))
# size = 100_000
# primary_energy_0 = 1.0e+12  # eV
# # field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/nph"
# field = "/home/raylend/Science/agnprocesses/data/test_1eV.txt"
# en_ph = np.logspace(-5, 1.5, 6852) * u.eV
# n_ph = spec.greybody_spectrum(en_ph, 1.0 * u.eV)
# grey = spec.create_2column_table(en_ph, n_ph)
# np.savetxt(field, grey, fmt="%.6e")
# energy_gamma_node, r_gamma_gamma_node = gamma_gamma.interaction_rate(
#     field,
#     1.0e+08 * u.eV,
#     primary_energy_0 * u.eV * 5,
#     background_photon_energy_unit=u.eV,
#     background_photon_density_unit=(u.eV * u.cm**3)**(-1)
# )
# energy_gamma_node = torch.tensor(energy_gamma_node.value,
#                                  device=device, dtype=torch.float64)
# r_gamma_gamma_node = torch.tensor(r_gamma_gamma_node.value,
#                                   device=device, dtype=torch.float64)
########################################################################

f_support_interpolation = torchinterp1d.Interp1d()


def torch_interpolation(x_new_tensor: torch.tensor,
                        x_old_tensor: torch.tensor,
                        y_old_tensor: torch.tensor):

    return(
        torch.squeeze(10.0**f_support_interpolation(
            torch.log10(x_old_tensor),
            torch.log10(y_old_tensor),
            torch.log10(x_new_tensor))
        )
    )

########################################################################


def background_photon_density_interpolated(energy_new,
                                           photon_field_file_path='',
                                           device=device, dtype=torch.float64):
    field = np.loadtxt(photon_field_file_path)
    energy_old = torch.tensor(field[:, 0],
                              device=device, dtype=torch.float64)
    density_old = torch.tensor(field[:, 1],
                               device=device, dtype=torch.float64)
    return torch_interpolation(energy_new,
                               energy_old,
                               density_old)

########################################################################


def generate_random_numbers(pdf,
                            pars=[],
                            shape=(1,),
                            xmin=torch.tensor(1.0,
                                              device=device,
                                              dtype=torch.float64),
                            xmax=torch.tensor(10.0,
                                              device=device,
                                              dtype=torch.float64),
                            normalized=False,  # deprecated
                            logarithmic=True,
                            n_integration=300,
                            device=None,
                            verbose=False,  # deprecated
                            **kwargs):
    filt_min_max = (xmin <= xmax)
    if False in filt_min_max:
        print(xmin[torch.logical_not(filt_min_max)])
        raise ValueError("xmin > xmax!")
    x_final = float('Inf') * torch.ones(shape,
                                        device=device, dtype=torch.float64)
    if logarithmic:
        c = []
        for i, param in enumerate(pars):
            try:
                iter(param)
                if len(param) != len(xmin):
                    raise ValueError(
                        "Vectorized parameter in pars must have the same length as xmin and xmax do.")
                for element_number, param_value in enumerate(param):
                    z = torch.logspace(torch.log10(xmin[element_number]),
                                       torch.log10(xmax[element_number]),
                                       n_integration,
                                       device=device, dtype=torch.float64)
                    c.append(
                        torch.max(z * pdf(z,
                                          *[param_value, *pars[1:]],
                                          **kwargs))
                    )
                break
            except TypeError:
                pass
        if c == []:
            for element_number, x_min_value in enumerate(xmin):
                z = torch.logspace(torch.log10(xmin[element_number]),
                                   torch.log10(xmax[element_number]),
                                   n_integration,
                                   device=device, dtype=torch.float64)
                c.append(
                    torch.max(z * pdf(z,
                                      *pars,
                                      **kwargs))
                )
        c = (torch.tensor(c, device=device, dtype=torch.float64)
             * torch.ones(xmin.shape, device=device, dtype=torch.float64))
    else:  # not logarithmic
        c = []
        for i, param in enumerate(pars):
            try:
                iter(param)
                if len(param) != len(xmin):
                    raise ValueError(
                        "Vectorized parameter in pars must have the same length as xmin and xmax do.")
                for element_number, param_value in enumerate(param):
                    z = torch.linspace(xmin[element_number],
                                       xmax[element_number],
                                       n_integration,
                                       device=device, dtype=torch.float64)
                    c.append(
                        torch.max(pdf(z,
                                      *[param_value, *pars[1:]],
                                      **kwargs))
                    )
                break
            except TypeError:
                pass
        if c == []:
            for element_number, x_min_value in enumerate(xmin):
                z = torch.linspace(xmin[element_number],
                                   xmax[element_number],
                                   n_integration,
                                   device=device, dtype=torch.float64)
                c.append(
                    torch.max(pdf(z,
                                  *pars,
                                  **kwargs))
                )
        c = (torch.tensor(c, device=device, dtype=torch.float64)
             * torch.ones(xmin.shape, device=device, dtype=torch.float64))
    # print("c = ", c)
    if True in torch.isnan(c):
        print("Nan occured in c!")
        print("Shape of c: ", c.shape)
        print("Shape of NaNs of c: ", c[torch.isnan(c)].shape)
        print("xmin for Nans: ", xmin[torch.isnan(c)])
        print("xmax for Nans: ", xmax[torch.isnan(c)])
        c[torch.isnan(c)] = torch.max(c[torch.logical_not(torch.isnan(c))])
    while len(x_final[x_final == float('Inf')]) > 0:
        assignment = (x_final == float('Inf'))
        generate_size = len(x_final[assignment])
        if logarithmic:
            x = 10.0**((torch.log10(xmax) -
                        torch.log10(xmin)) *
                       torch.rand(shape,
                                  device=device, dtype=torch.float64) +
                       torch.log10(xmin))
            p = (pdf(x, *pars, **kwargs) * x)[assignment]
            x = x[assignment]
        else:
            x = ((xmax - xmin) *
                 torch.rand(shape,
                            device=device,
                            dtype=torch.float64) + xmin)
            p = (pdf(x, *pars, **kwargs))[assignment]
            x = x[assignment]
        if True in torch.isnan(x):
            raise RuntimeError("Nan occured in x!")
        if True in torch.isnan(p):
            raise RuntimeError("Nan occured in p!")
        y_new = torch.rand(
            generate_size,
            device=device, dtype=torch.float64) * c[assignment]
        filt = (y_new < p)
        x[torch.logical_not(filt)] = float('Inf')
        x_final[assignment] = x
    return x_final


########################################################################
def generate_random_distance_traveled(e_particle: torch.tensor,
                                      e_particle_node: torch.tensor,
                                      r_particle_node: torch.tensor,
                                      device=device, dtype=torch.float64):
    """
    Generates tensor of random particle traveled distances
    at e_particle energy points given that the
    interaction rate r_particle_node are
    computed at e_particle_node energy points.
    """
    r_particle = torch_interpolation(
        e_particle,
        e_particle_node,
        r_particle_node
    )
    uniform_random = torch.rand(e_particle.shape, device=device,
                                dtype=torch.float64)
    return (-torch.log(uniform_random) / r_particle)


########################################################################
def generate_random_cosine(shape=(1,),
                           cosmin=-1.0,  # or tensor with size of shape
                           cosmax=1.0,  # or tensor with size of shape
                           device=device, dtype=torch.float64):
    return(
        (cosmax - cosmin) * torch.rand(max(shape),
                                       device=device,
                                       dtype=torch.float64) + cosmin
    )


########################################################################
def beta(s):
    # Kahelriess
    return (1.0 - S_MIN / s)**0.5


#######################################################################
def sigma_pair_cross_section_reduced(s):
    """
    Eq. (5) of Kahelriess without s in the denominator and
    without constants in the numerator.
    """
    b = beta(s)
    t2 = (3.0 - b**4) * torch.log((1.0 + b) / (1.0 - b))
    t3 = -2.0 * b * (2.0 - b * b)
    return((t2 + t3))


########################################################################
def auxiliary_pair_production_function(x,  # min
                                       x_max,
                                       photon_field_file_path='',
                                       device=device,
                                       dtype=torch.float64):
    dim2 = 100
    ys = []
    for i in range(0, max(x.shape)):
        x_primed = torch.logspace(
            float(torch.log10(x[i])),
            float(torch.log10(x_max[i])),
            dim2,
            device=device
        )
        y = background_photon_density_interpolated(
            x_primed,
            photon_field_file_path=photon_field_file_path,
            device=device, dtype=torch.float64) / x_primed**2
        ys.append(torch.trapz(y, x_primed))
    ys = torch.tensor(ys, device=device, dtype=torch.float64)
    print(f'ys.shape = {ys.shape}')
    return (ys)


#######################################################################
# eps_max = 1000.0  # eV
# s_precalc = torch.logspace(float(torch.log10(torch.tensor(
#     [S_MIN / (4.0 * 1.0e+15), ], device=device, dtype=torch.float64))),
#     float(torch.log10(torch.tensor(
#         [10000.0, ], device=device, dtype=torch.float64))),
#     10000,
#     device=device, dtype=torch.float64)
#
# print(f"s_precalc.shape = {s_precalc.shape}")
#
# f_precalc = auxiliary_pair_production_function(s_precalc,
#                                                photon_field_file_path=field,
#                                                device=device, dtype=torch.float64)
#
# print(f"f_precalc.shape = {f_precalc.shape}")
#
#
# def auxiliary_pair_production_function_precalculated(x):
#     return torch_interpolation(x, s_precalc, f_precalc)
#


#######################################################################
def pair_production_differential_cross_section_s(
        s,
        energy,
        photon_field_file_path='',
        device=device, dtype=torch.float64):
    return (sigma_pair_cross_section_reduced(s) *
            (auxiliary_pair_production_function(
                s / (4.0 * energy),
                eps_max * torch.ones(energy.shape,
                                     device=device, dtype=torch.float64),
                photon_field_file_path=photon_field_file_path)
             / (8.0 * energy**2))
            )


########################################################################
# def generate_random_s_new(energy,
#                           device=device, dtype=torch.float64):
#     return (generate_random_numbers(
#         pair_production_differential_cross_section_s,
#         pars=[energy, ],
#         shape=energy.shape,
#         xmin=S_MIN * torch.ones(energy.shape,
#                                 device=device, dtype=torch.float64),
#         xmax=4.0 * energy * eps_max * torch.ones(energy.shape,
#                                                  device=device, dtype=torch.float64),
#         normalized=True,
#         logarithmic=True,
#         n_integration=300,
#         device=device,
#         verbose=False)
#     )


########################################################################
def pair_production_differential_cross_section_y(y, s):
    """
    For given s, the energy fraction y of the lowest energy secondary
    lepton (electron or positron) in the pair production process is
    sampled according to the corresponding differential cross section
    (see eq. (11) M. Kachelrieß et al. /
    Computer Physics Communications 183 (2012) 1036–1043
    http://dx.doi.org/10.1016/j.cpc.2011.12.025)
    """
    beta = (1.0 - S_MIN / s)**0.5
    t11 = y**2 / (1.0 - y) + 1.0 - y
    t12 = (1.0 - beta**2) / (1.0 - y)
    t13 = (-(1.0 - beta**2)**2) / (4.0 * y * (1.0 - y)**2)
    t1 = t11 + t12 + t13
    t2 = 1.0 + (2.0 * beta**2) * (1.0 - beta**2)
    return (t1 / (y * t2))


########################################################################
def IC_differential_cross_section_y(y, s):
    """
    For given s, the energy fraction y of the electron which has been
    undergone IC scattering is
    sampled according to the corresponding
    differential cross section
    (see eq. (12) M. Kachelrieß et al. /
    Computer Physics Communications 183 (2012) 1036–1043
    http://dx.doi.org/10.1016/j.cpc.2011.12.025)
    """
    y_min = ELECTRON_REST_ENERGY**2 / s
    return (((1.0 + y**2) / 2.0 -
             (2.0 * y_min * (y - y_min) * (1.0 - y)) /
             (y * (1.0 - y_min)**2)) / y
            )


#######################################################################
electron_ic_energy_loss_rate_constant = (0.75 *
                                         (6.6524e-25)**(-1) *
                                         (5.068e+22)**(-1))  # eV / cm


def ic_electron_energy_loss_distance_rate(s, eps_thr):
    y_min = ELECTRON_REST_ENERGY**2 / s
    y_max = 1.0 - eps_thr
    one_minus_y_min = 1.0 - y_min
    t11 = -torch.log(y_max) / eps_thr - 1.0
    t12 = 1.0 - (4.0 * y_min * (1.0 + 2.0 * y_min)) / one_minus_y_min**2
    t2 = 1.0 / 6.0 * eps_thr * (1.0 * 2.0 * y_max)
    t3 = ((2.0 * y_min * (1.0 + 2.0 * y_min / y_max) * eps_thr) /
          one_minus_y_min**2)
    return (electron_ic_energy_loss_rate_constant *
            y_min * eps_thr / one_minus_y_min *
            (t11 * t12 + t2 + t3))


#######################################################################
def generate_random_y(s,
                      eps_thr=None,
                      logarithmic=True,
                      device=device,
                      dtype=torch.float64,
                      process='PP'):  # or 'IC', i.e.
    # Pair Production or Inverse Compton
    """
    Generates random fraction of the squared total energy s for
    electrons (positrons) produced in gamma-gamma collisions
    in accordance with
    pair_production_differential_cross_section_y(y, s)
    or IC_differential_cross_section_y.

    Returns a torch.tensor of the requested shape.
    """
    shape = s.shape
    if process == 'PP':
        return generate_random_numbers(
            pair_production_differential_cross_section_y,
            [s, ],
            shape=shape,
            xmin=ELECTRON_REST_ENERGY**2 / s * 0.99,
            xmax=torch.ones(shape, device=device,
                            dtype=dtype) * 0.99,
            logarithmic=logarithmic,
            device=device,
            normalized=True,
            n_integration=300)
    elif process == 'IC':
        if eps_thr is None:
            eps_thr = 0.001
        return generate_random_numbers(
            IC_differential_cross_section_y,
            [s, ],
            shape=shape,
            xmin=ELECTRON_REST_ENERGY**2 / s * 0.99,
            xmax=(torch.ones(shape, device=device,
                             dtype=dtype) - eps_thr) * 0.99,
            logarithmic=logarithmic,
            device=device,
            normalized=True,
            n_integration=300)
    else:
        raise ValueError(
            "Invalid value of 'process' variable: must be 'PP' or 'IC'."
        )


#######################################################################
def generate_random_background_photon_energy(
        shape,
        photon_field_file_path,
        device=device,
        energy_photon_min=None,
        energy_photon_max=None,
        **kwargs
):
    if energy_photon_min is None or energy_photon_max is None:
        field = np.loadtxt(photon_field_file_path)
    if energy_photon_min is None:
        energy_photon_min = field[0, 0]
        xmin = torch.ones(shape, device=device,
                          dtype=torch.float64) * energy_photon_min
    else:
        xmin = torch.ones(shape, device=device,
                          dtype=torch.float64) * energy_photon_min
    if energy_photon_max is None:
        energy_photon_max = field[-1, 0]
        xmax = torch.ones(shape, device=device,
                          dtype=torch.float64) * energy_photon_max
    else:
        xmax = torch.ones(shape, device=device,
                          dtype=torch.float64) * energy_photon_max
    pdf = partial(background_photon_density_interpolated,
                  photon_field_file_path=photon_field_file_path,
                  device=device,
                  dtype=torch.float64)
    return (generate_random_numbers(
        pdf,
        pars=[],
        shape=shape,
        xmin=xmin,
        xmax=xmax,
        normalized=True,
        logarithmic=True,
        n_integration=3000,
        device=device,
        **kwargs
    ))


########################################################################
def generate_random_s(energy,
                      photon_field_file_path,
                      random_cosine=True,
                      device=device,
                      process='PP',  # or 'IC'
                      energy_ic_threshold=None,
                      energy_photon_max=None,
                      dtype=torch.float64,
                      **cosine_kwargs):
    if process == 'PP':
        energy_photon_min = S_MIN / (4.0 * energy) * 1.010
        filt_less_than_zero = (energy_photon_min < 0.0)
        if True in filt_less_than_zero:
            print('energy_photon_min[filt_less_than_zero] = ',
                  energy_photon_min[filt_less_than_zero])
            print('energy[filt_less_than_zero] = ',
                  energy[filt_less_than_zero])
            raise RuntimeError(
                "There is less than zero value in energy_photon_min!")
        epsilon = generate_random_background_photon_energy(
            energy.shape,
            photon_field_file_path,
            device=device,
            energy_photon_min=energy_photon_min,
            energy_photon_max=energy_photon_max
        )
        if random_cosine:
            cosmax = 1.0 - S_MIN / (2.0 * epsilon * energy)
            cosine = generate_random_cosine(energy.shape,
                                            cosmax=cosmax,
                                            device=device,
                                            **cosine_kwargs)
        else:
            cosine = -1.0
        return (2.0 * epsilon * (1.0 - cosine) * energy)
    #######################################################################
    elif process == 'IC':
        if energy_ic_threshold is None:
            energy_ic_threshold = 0.001 * torch.min(energy)
        eps_thr = energy_ic_threshold / energy
        b = (1.0 - ELECTRON_REST_ENERGY**2 / energy**2)**0.5
        epsilon = generate_random_background_photon_energy(
            energy.shape,
            photon_field_file_path,
            device=device,
            energy_photon_min=(eps_thr * ELECTRON_REST_ENERGY**2 /
                               (2.0 * (1.0 + b) * (1.0 - eps_thr) * energy)) * 1.010,
            energy_photon_max=energy_photon_max
        )
        if random_cosine:
            cosmax = (1.0 - (eps_thr * ELECTRON_REST_ENERGY**2) /
                            (2.0 * (1.0 - eps_thr) * energy * epsilon)
                      ) / b
            cosine = generate_random_cosine(energy.shape,
                                            cosmax=cosmax,
                                            device=device,
                                            **cosine_kwargs)
        else:
            cosine = -1.0
        return (ELECTRON_REST_ENERGY**2 +
                2.0 * energy * epsilon * (1.0 - b * cosine))


########################################################################
def center_energy(energy_bins):
    return 10.0**((np.log10(energy_bins[1:]) +
                   np.log10(energy_bins[:-1])) /
                  2.0)


########################################################################
def make_2d_energy_histogram(
    primary_energy,
    another_energy,
    bins_primary=None,
    bins_another=None,
    verbose=False,
    density=False
):
    if type(primary_energy) == torch.Tensor:
        primary_energy = primary_energy.detach().to('cpu').numpy()
    if type(another_energy) == torch.Tensor:
        another_energy = another_energy.detach().to('cpu').numpy()
    if bins_primary is None:
        bins_primary = 30
    if bins_another is None:
        bins_another = 30
    f = np.histogram2d(
        primary_energy,
        another_energy,
        bins=[bins_primary, bins_another],
        density=density
    )
    if verbose:
        return f
    else:
        return f[0]


########################################################################
def reweight_2d_histogram(histogram_2d,
                          primary_energy_center,
                          another_energy_center,
                          original_spectrum=spec.power_law,
                          original_params=[1.0, 1.0, 1.0],
                          new_spectrum=spec.power_law,
                          new_params=[1.0, 1.0, 1.0],
                          original_keyargs={},
                          new_keyargs={},
                          reweight_array='primary',  # or 'another'
                          return_1d=True,
                          number_of_particles=1,
                          old=False):
    if type(primary_energy_center) == torch.Tensor:
        primary_energy_center = primary_energy_center.detach().to(
            'cpu'
        ).numpy()
    if type(another_energy_center) == torch.Tensor:
        another_energy_center = another_energy_center.detach().to(
            'cpu'
        ).numpy()
    if reweight_array == 'primary':
        x = primary_energy_center
    elif reweight_array == 'another':
        x = another_energy_center
    else:
        warnings.warn("Reweighting with primary energy array",
                      UserWarning)
        x = primary_energy_center
    new_spec = new_spectrum(x, *new_params, **new_keyargs)
    old_spec = original_spectrum(x, *original_params, **original_keyargs)
    k_modif = (new_spec / old_spec)
    n_particles_theory_old = simps(old_spec, x)
    h = np.copy(histogram_2d)
    if reweight_array == 'primary':
        for i in range(0, h.shape[0]):
            h[i, :] = h[i, :] * k_modif[i]
    elif reweight_array == 'another':
        for i in range(0, h.shape[0]):
            h[:, i] = h[:, i] * k_modif[i]
    else:
        warnings.warn("Reweighting with primary energy array",
                      UserWarning)
        for i in range(0, h.shape[0]):
            h[i, :] = h[i, :] * k_modif[i]
    h = (h * n_particles_theory_old /
         number_of_particles)
    if return_1d:
        if reweight_array == 'primary':
            return np.sum(h, axis=0).reshape(
                another_energy_center.shape)
        elif reweight_array == 'another':
            return np.sum(h, axis=1).reshape(
                primary_energy_center.shape)
        else:
            return np.sum(h, axis=0).reshape(
                primary_energy_center.shape)
    else:
        return h


########################################################################
def make_SED(
        energy_array,
        energy_bins,
        primary_energy,
        primary_energy_bins=None,
        original_spectrum=spec.power_law,
        original_params=[1.0, 1.0, 1.0],
        new_spectrum=spec.power_law,
        new_params=[1.0, 1.0, 1.0],
        original_keyargs={},
        new_keyargs={},
        reweight_array='primary',
        number_of_particles=1,
        density=False,
        histogram_2d_precalculated=None,
        old=False):
    if type(primary_energy) == torch.Tensor:
        primary_energy = primary_energy.detach().to('cpu').numpy()
    if type(energy_array) == torch.Tensor:
        energy_array = energy_array.detach().to('cpu').numpy()
    if primary_energy_bins is None:
        primary_energy_bins = energy_bins
    try:
        energy_bins = energy_bins.to(u.eV).value
    except:
        pass
    try:
        primary_energy_bins = primary_energy_bins.to(u.eV).value
    except:
        pass
    if histogram_2d_precalculated is None and energy_array is not None:
        histogram_2d = make_2d_energy_histogram(
            primary_energy,
            energy_array,
            bins_primary=primary_energy_bins,
            bins_another=energy_bins,
            verbose=False,
            density=density
        )
    else:
        histogram_2d = np.copy(histogram_2d_precalculated)
    primary_energy_center = center_energy(primary_energy_bins)
    another_energy_center = center_energy(energy_bins)
    spectrum = reweight_2d_histogram(
        histogram_2d,
        primary_energy_center,
        another_energy_center,
        original_spectrum=original_spectrum,
        original_params=original_params,
        new_spectrum=new_spectrum,
        new_params=new_params,
        original_keyargs=original_keyargs,
        new_keyargs=new_keyargs,
        reweight_array=reweight_array,  # or 'another'
        return_1d=True,
        number_of_particles=number_of_particles,
        old=old
    )
    spectrum = spectrum / (energy_bins[1:] - energy_bins[:-1])
    return (another_energy_center, spectrum * another_energy_center**2)


########################################################################
def date_time_string_human_friendly():
    now = datetime.now()
    return (now.strftime("%d-%m-%Y_%H-%M-%S"))  # dd-mm-YY H-M-S


########################################################################
def get_current_date_time():
    return datetime.now()


########################################################################
def time_delta_to_seconds(dt):
    return (dt.seconds)


#######################################################################
def make_sed_from_monte_carlo_cash_files(
    folder: str,
    energy_bins,  # array-like
    primary_energy_bins=None,
    cascade_type_of_particles='all',  # or 'survived', or 'cascade' (only)
    original_spectrum=spec.power_law,
    original_params=[1.0, 1.0, 1.0],
    new_spectrum=spec.power_law,
    new_params=[1.0, 1.0, 1.0],
    original_keyargs={},
    new_keyargs={},
    reweight_array='primary',
    number_of_particles=1,
    particles_of_interest='gamma',  # or 'electron'
    density=False,
    verbose=False,
    output='sed',  # or 'histogram_2d_precalculated'
    device='cpu',
    old=False
):
    """
    This function takes two positional arguments:
    1) folder: string with the project / absolute path to
    the folder with Monte-Carlo process cash files;
    2) energy_bins: nd.array of energy bins which will be used for
    the SED construction. It may be also an array-like astropy.Qunatity.

    particles_of_interest argument: the SED of which type of particles
    will be constructed.

    device argument: on which device torch.tensors
    will be loaded from cash files.

    Other keyword arguments are the same as for the make_sed().
    """
    # torch.load('tensors.pt', map_location=torch.device('cpu'))
    # os shows all content of the directory 'folder'
    # string by string:
    if not os.path.isdir(folder):
        folder = os.getcwd() + '/data/torch/' + folder
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            "Cannot find files in folder: ", folder
        )
    cmd = 'ls ' + folder + '/' + particles_of_interest + 's*' + ' -1'
    cmd_line_rtrn = (subprocess.check_output(cmd, shell=True)[:-1]).decode(
        'utf-8'
    )
    cash_files = cmd_line_rtrn.split('\n')
    hist = 0.0
    counter = 0
    for file in cash_files:
        tens = torch.load(file,
                          map_location=torch.device(device))
        if cascade_type_of_particles == 'all':
            hist = hist + make_2d_energy_histogram(
                tens[0, :],
                tens[1, :],
                bins_primary=primary_energy_bins,
                bins_another=energy_bins,
                verbose=verbose,
                density=density
            )
        elif cascade_type_of_particles == 'survived':
            filt_survived = (tens[0, :] == tens[1, :])
            hist = hist + make_2d_energy_histogram(
                tens[0, :][filt_survived],
                tens[1, :][filt_survived],
                bins_primary=primary_energy_bins,
                bins_another=energy_bins,
                verbose=verbose,
                density=density
            )
        elif cascade_type_of_particles == 'cascade':
            filt_cascade = torch.logical_not(tens[0, :] == tens[1, :])
            hist = hist + make_2d_energy_histogram(
                tens[0, :][filt_cascade],
                tens[1, :][filt_cascade],
                bins_primary=primary_energy_bins,
                bins_another=energy_bins,
                verbose=verbose,
                density=density
            )
        else:
            raise ValueError(
                "Unknown option of cascade_type_of_particles!"
            )
    if output == 'sed':
        return (make_SED(
            None,
            energy_bins,
            None,
            primary_energy_bins=primary_energy_bins,
            original_spectrum=original_spectrum,
            original_params=original_params,
            new_spectrum=new_spectrum,
            new_params=new_params,
            original_keyargs=original_keyargs,
            new_keyargs=new_keyargs,
            reweight_array=reweight_array,
            number_of_particles=number_of_particles,
            density=density,
            histogram_2d_precalculated=hist,
            old=old)
        )
    elif output == 'histogram_2d_precalculated':
        return hist
    else:
        raise ValueError("Unknown value of 'output' option!")


#######################################################################
def monte_carlo_process(
        primary_energy_tensor_size: int,
        region_size,  # : astropy.Quantity or str
        photon_field_file_path: str,
        injection_place='zero',  # 'zero' means in the beginning of the region;
        # 'uniform' means it will be uniformly distributed between 0 and region_size
        primary_spectrum=spec.power_law,
        primary_spectrum_params=[1.0, 1.0, 1.0],
        primary_energy_min=1.0e+08 * u.eV,
        primary_energy_max=1.0e+13 * u.eV,
        energy_ic_threshold=1.0e+06 * u.eV,
        primary_energy_tensor_user=None,
        primary_particle='gamma',  # or 'electron'
        observable_energy_min=1.0e+08 * u.eV,
        observable_energy_max=1.0e+13 * u.eV,
        background_photon_energy_unit=u.eV,
        background_photon_density_unit=(u.eV * u.cm**3)**(-1),
        energy_photon_max=None,
        terminator_step_number=None,
        folder='output',
        ic_losses_below_threshold=False,
        device=device, dtype=torch.float64):
    """
    gammas and electrons consist of 4 rows:

    0) primary energy of individual particles;
    1) current (observable) energy of individual particles;
    2) distance traveled by individual particles;
    3) distance of the primary particle injection [0; region_size).
    """
    t_start = get_current_date_time()
    print('Monte-Carlo process started: ', date_time_string_human_friendly())
    folder = 'data/torch/' + str(folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    new_electrons = None
    new_positrons = None
    new_gammas = None
    shape = (primary_energy_tensor_size, )
    region_size = region_size.to(u.cm).value
    if energy_photon_max is not None:
        energy_photon_max = energy_photon_max.to(u.eV).value
    gammas = torch.zeros((4, primary_energy_tensor_size),
                         device=device, dtype=torch.float64)
    electrons = torch.zeros((4, primary_energy_tensor_size),
                            device=device, dtype=torch.float64)
    ######################################################################
    if injection_place == 'zero':
        gammas[2, :] = 0.0
        electrons[2, :] = 0.0
        gammas[3, :] = 0.0
        electrons[3, :] = 0.0
    elif injection_place == 'uniform':
        inj_coord = ((region_size - 0.0) *
                     torch.rand(shape,
                                device=device,
                                dtype=torch.float64))
        gammas[2, :] = torch.clone(inj_coord)
        electrons[2, :] = torch.clone(inj_coord)
        gammas[3, :] = torch.clone(inj_coord)
        electrons[3, :] = torch.clone(inj_coord)
    else:
        raise ValueError(
            "Unknown option for the 'unjection_place' parameter!\n" +
            "It must be either 'zero' or 'unifrom'!"
        )
    ######################################################################
    primary_energy_min = primary_energy_min.to(u.eV).value
    primary_energy_max = primary_energy_max.to(u.eV).value
    if primary_energy_tensor_user is not None:
        if primary_particle == 'gamma':
            gammas[0, :] = primary_energy_tensor_user
        elif primary_particle == 'electron':
            electrons[0, :] = primary_energy_tensor_user
        else:
            raise ValueError(
                "Unknown type of primary particle! Valid type is either 'gamma' or 'electron'!")
    else:
        if primary_particle == 'gamma':
            no_gammas = False
            no_electrons = True
            gammas[0, :] = generate_random_numbers(
                primary_spectrum,
                primary_spectrum_params,  # gamma, norm, en_ref
                shape=shape,
                xmin=(primary_energy_min *
                      torch.ones(shape, device=device,
                                 dtype=torch.float64)),
                xmax=(primary_energy_max *
                      torch.ones(shape, device=device,
                                 dtype=torch.float64)),
                device=device,
                logarithmic=True,
                n_integration=300,
                normalized=True
            )
        elif primary_particle == 'electron':
            no_electrons = False
            no_gammas = True
            electrons[0, :] = generate_random_numbers(
                primary_spectrum,
                primary_spectrum_params,  # gamma, norm, en_ref
                shape=shape,
                xmin=(primary_energy_min *
                      torch.ones(shape, device=device,
                                 dtype=torch.float64)),
                xmax=(primary_energy_max *
                      torch.ones(shape, device=device,
                                 dtype=torch.float64)),
                device=device,
                logarithmic=True,
                n_integration=300,
                normalized=True
            )
        else:
            raise ValueError(
                "Unknown type of primary particle! Valid type is either 'gamma' or 'electron'!")
    ######################################################################
    # observable energy is equal to primary energy at the first step
    gammas[1, :] = gammas[0, :]
    electrons[1, :] = electrons[0, :]
    ######################################################################
    energy_ic_threshold = energy_ic_threshold.to(u.eV).value
    observable_energy_min = observable_energy_min.to(u.eV).value
    # observable_energy_min = np.min([observable_energy_min,
    #                                 energy_ic_threshold])
    observable_energy_max = observable_energy_max.to(u.eV).value
    if observable_energy_max <= observable_energy_min:
        raise ValueError(
            "observable_energy_min must be less than observable_energy_max!")
    en = np.logspace(np.log10(min([observable_energy_min,
                                   primary_energy_min])),
                     np.log10(max([observable_energy_max,
                                   primary_energy_max])),
                     10_000)
    e_gamma_node, r_gamma_node = gamma_gamma.interaction_rate(
        photon_field_file_path,
        min([observable_energy_min,
             primary_energy_min]) * u.eV * 0.90,
        max([observable_energy_max,
             primary_energy_max]) * u.eV * 1.10,
        background_photon_energy_unit=background_photon_energy_unit,
        background_photon_density_unit=background_photon_density_unit)
    save_table = spec.create_2column_table(
        e_gamma_node,
        r_gamma_node
    )
    # np.savetxt(
    #     "/home/raylend/Science/agnprocesses/processes/EBL_models/Gimore2012_gamma-gamma_int_rate_z=0.015_final.txt",
    #     save_table,
    #     fmt="%.6e"
    # )
    e_ic_node, r_ic_node = ic.IC_interaction_rate(
        photon_field_file_path,
        min([observable_energy_min,
             primary_energy_min]) * u.eV * 0.90,
        max([observable_energy_max,
             primary_energy_max]) * u.eV * 1.10,
        energy_ic_threshold * u.eV,
        background_photon_energy_unit=background_photon_energy_unit,
        background_photon_density_unit=background_photon_density_unit)
    r_gamma_node = spec.to_current_energy(en * u.eV,
                                          e_gamma_node,
                                          r_gamma_node).to(u.cm**(-1))
    r_ic_node = spec.to_current_energy(en * u.eV,
                                       e_ic_node,
                                       r_ic_node).to(u.cm**(-1))
    e_gamma_node = torch.tensor(en, device=device,
                                dtype=torch.float64)
    r_gamma_node = torch.tensor(r_gamma_node.value, device=device,
                                dtype=torch.float64)
    e_ic_node = torch.tensor(en, device=device,
                             dtype=torch.float64)
    r_ic_node = torch.tensor(r_ic_node.value, device=device,
                             dtype=torch.float64)
    ######################################################################
    nstep = 0
    while (len(gammas[1, :]) > 0) or (len(electrons[1, :]) > 0):
        print(f"Step number {nstep}:")
        # Sample the distance traveled
        if no_gammas == False:
            traveled = generate_random_distance_traveled(
                gammas[1, :],
                e_gamma_node,
                r_gamma_node)
            gammas[2, :] += traveled
            print("Median distance traveled by gamma rays at this step: {:.3e} cm = {:.2f} % of region size".format(
                torch.median(gammas[2, :]),
                torch.median(gammas[2, :]) / region_size * 100.0
            ))
        if no_electrons == False:
            traveled_electrons = generate_random_distance_traveled(
                electrons[1, :],
                e_ic_node,
                r_ic_node)
            electrons[2, :] += traveled_electrons
            print("Median distance traveled by electrons at this step: {:.3e} cm = {:.2f} % of region size".format(
                torch.median(electrons[2, :]),
                torch.median(electrons[2, :]) / region_size * 100.0
            ))
        ###################################################################
        # Save out escaped particles and
        # update gammas and electrons removing escaped particles.
        if no_gammas == False:  # i.e. there are some gamma rays in the moment
            filter_escaped = (gammas[2, :] > region_size)
            filter_escaped = torch.logical_or(filter_escaped,
                                              (gammas[1, 0] <
                                               observable_energy_min))
            filter_not_escaped = torch.logical_not(filter_escaped)
            if len(filter_escaped[filter_escaped == True]) > 0:
                torch.save(torch.stack([gammas[0, :][filter_escaped],
                                        gammas[1, :][filter_escaped],
                                        gammas[2, :][filter_escaped],
                                        gammas[3, :][filter_escaped]]),
                           folder + '/' + 'gammas_' +
                           'step000' + str(nstep) +
                           '_' + date_time_string_human_friendly() + '.pt')
            gammas = torch.stack([gammas[0, :][filter_not_escaped],
                                  gammas[1, :][filter_not_escaped],
                                  gammas[2, :][filter_not_escaped],
                                  gammas[3, :][filter_not_escaped]])
        if no_electrons == False:
            filter_escaped = (electrons[2, :] > region_size)
            filter_escaped = torch.logical_or(filter_escaped,
                                              (electrons[1, 0] <
                                               observable_energy_min))
            filter_not_escaped = torch.logical_not(filter_escaped)
            traveled_electrons = traveled_electrons[filter_not_escaped]
            if len(filter_escaped[filter_escaped == True]) > 0:
                torch.save(torch.stack([electrons[0, :][filter_escaped],
                                        electrons[1, :][filter_escaped],
                                        electrons[2, :][filter_escaped],
                                        electrons[3, :][filter_escaped]]),
                           folder + '/' + 'electrons_' +
                           'step000' + str(nstep) +
                           '_' + date_time_string_human_friendly() + '.pt')
            electrons = torch.stack([electrons[0, :][filter_not_escaped],
                                     electrons[1, :][filter_not_escaped],
                                     electrons[2, :][filter_not_escaped],
                                     electrons[3, :][filter_not_escaped]])
        if len(gammas[1, :]) == 0:
            no_gammas = True
        if len(electrons[1, :]) == 0:
            no_electrons = True
        if no_electrons and no_gammas:
            break
        ###################################################################
        # Generate s
        if no_gammas == False:
            s_gammas = generate_random_s(
                gammas[1, :],
                photon_field_file_path,
                random_cosine=True,
                process='PP',
                device=device,
                energy_photon_max=energy_photon_max)
            print("HERE GENERATES S_GAMMAS!!!")
            filt_s_gammas = (s_gammas >= S_MIN)
            print("There are {:d} gamma interactions under S_MIN".format(
                (filt_s_gammas[filt_s_gammas == False]).shape[0])
            )
            s_gammas = s_gammas[filt_s_gammas]
        if no_electrons == False:
            eps_thr = energy_ic_threshold / electrons[1, :]
            s_electrons = generate_random_s(
                electrons[1, :],
                photon_field_file_path,
                random_cosine=True,
                process='IC',
                energy_ic_threshold=energy_ic_threshold,
                device=device,
                energy_photon_max=energy_photon_max)
            filt_s_electrons = (s_electrons >= (ELECTRON_REST_ENERGY**2 /
                                (1.0 - eps_thr)))
            print(
                "There are {:d} electron interactions under ELECTRON_REST_ENERGY**2 / (1.0 - eps_thr)".format(
                    (filt_s_electrons[filt_s_electrons == False]).shape[0])
            )
            s_electrons = s_electrons[filt_s_electrons]
            eps_thr = eps_thr[filt_s_electrons]
            traveled_electrons = traveled_electrons[filt_s_electrons]
            if ic_losses_below_threshold:
                electron_losses = (ic_electron_energy_loss_distance_rate(
                    s_electrons, eps_thr) * traveled_electrons
                )
                filt_less_than_zero = (electron_losses < 0)
                if True in filt_less_than_zero:
                    print(electron_losses[filt_less_than_zero])
                    print(
                        "There are negative losses of electrons: {:d} cases".format(
                            electron_losses[filt_less_than_zero].shape[0]
                        )
                    )
                electron_losses[filt_less_than_zero] = 0.0
            else:
                electron_losses = 0.0
            electrons[1, :] = electrons[1, :] - electron_losses
            not_lost = (electrons[1, :] > observable_energy_min)
            electrons = torch.stack([electrons[0, :][not_lost],
                                     electrons[1, :][not_lost],
                                     electrons[2, :][not_lost],
                                     electrons[3, :][not_lost]])
            loss_fraction = (len(not_lost[not_lost == False]) /
                             len(not_lost))
            s_electrons = s_electrons[not_lost]
            eps_thr = eps_thr[not_lost]
            print(
                "There are {:.2f} % electrons lost in IC scattering process under the threshold".format(
                    loss_fraction * 100.0)
            )
        ###################################################################
        # Sample transfered energy
        if no_gammas == False:
            y_from_gammas = generate_random_y(
                s_gammas,
                process='PP',
                logarithmic=True,
                device=device,
                dtype=torch.float64
            )
        if no_electrons == False:
            y_from_electrons = generate_random_y(
                s_electrons,
                eps_thr=eps_thr,
                process='IC',
                logarithmic=True,  # AHTUNG ATTENTION !!!
                device=device,
                dtype=torch.float64
            )
            filt_greater_than_one = (y_from_electrons > 1.0)
            if True in filt_greater_than_one:
                warnings.warn(
                    "There is y_from_electrons greater than one!",
                    UserWarning)
                print(
                    "There are {:d} cases of y_from_electrons greater than one!".format(
                        len(y_from_electrons[filt_greater_than_one])
                    )
                )
                # print("y_from_electrons[filt_greater_than_one] = ",
                #       y_from_electrons[filt_greater_than_one])
                y_from_electrons[filt_greater_than_one] = (
                    1.0 - eps_thr[filt_greater_than_one]
                )
        ###################################################################
        # Create cascade particles
        if no_gammas == False:
            new_electrons = torch.stack(
                [gammas[0, :],
                 y_from_gammas * gammas[1, :],
                 gammas[2, :],
                 gammas[3, :]]
            )
            new_positrons = torch.stack(
                [gammas[0, :],
                 (1.0 - y_from_gammas) * gammas[1, :],
                 gammas[2, :],
                 gammas[3, :]]
            )
            new_electrons = torch.cat([new_electrons, new_positrons],
                                      dim=1)
            new_positrons = None
            gammas[1, :] = 0.0  # all gammas were absorbed
            no_gammas = True
        if no_electrons == False:
            new_gammas = torch.stack(
                [electrons[0, :],
                 (1.0 - y_from_electrons) * electrons[1, :],
                 electrons[2, :],
                 electrons[3, :]])
            electrons = torch.stack(
                [electrons[0, :],
                 y_from_electrons * electrons[1, :],
                 electrons[2, :],
                 electrons[3, :]])
            no_electrons = False
        if new_electrons is not None:
            if no_electrons == False:
                electrons = torch.cat([electrons, new_electrons], dim=1)
            else:
                electrons = new_electrons
                no_electrons = False
            new_electrons = None
        if new_gammas is not None:
            gammas = new_gammas
            new_gammas = None
            no_gammas = False
        ###################################################################
        if len(gammas[1, :]) == 0:
            no_gammas = True
        if len(electrons[1, :]) == 0:
            no_electrons = True
        if no_electrons and no_gammas:
            break
        print(
            f"There are {gammas[1, :][gammas[1, :] > 0].shape[0]} gamma rays to be tracked further")
        print(
            "Maximum energy among gamma rays: {:.3e} eV".format(
                torch.max(gammas[1, :])
            ))
        print(
            "Median traveled distance of gamma rays: {:.3e} cm = {:.2f}% of region size".format(
                torch.median(gammas[2, :]),
                torch.median(gammas[2, :]) / region_size * 100.0
            )
        )
        print(
            f"There are {electrons[1, :][electrons[1, :] > 0].shape[0]} electrons to be tracked further")
        print(
            "Maximum energy among electrons: {:.3e} eV".format(
                torch.max(electrons[1, :])
            ))
        print(
            "Median traveled distance of electrons: {:.3e} cm = {:.2f}% of region size".format(
                torch.median(electrons[2, :]),
                torch.median(electrons[2, :]) / region_size * 100.0
            ))
        print("\n")
        nstep += 1
        if (nstep >= terminator_step_number):
            break
        if no_electrons == True and no_gammas == True:
            break
    print("Done!")
    print('Monte-Carlo process finished: ', date_time_string_human_friendly())
    t_stop = get_current_date_time()
    print("Total time of calculations: ",
          (
              t_stop - t_start
          )
          )
    return None


############################################################################
# s = 1.0e+15 * torch.ones((100_000, ), device='cuda', dtype=torch.float64)
# s = torch.logspace(log10(ELECTRON_REST_ENERGY**2),
#                    16,
#                    1_000,
#                    device='cuda', dtype=torch.float64)
# eps_thr = 1.0e-06
# y = generate_random_y(s,
#                       eps_thr=eps_thr,
#                       logarithmic=True,
#                       device=device,
#                       dtype=torch.float64,
#                       process='IC')
# print('max of y = {:.6e}'.format(torch.max(y)))
# print('median of y = {:.6e}'.format(torch.median(y)))
# print('min of y = {:.6e}'.format(torch.min(y)))
