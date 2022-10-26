from math import log10
import torch
import numpy as np
import agnprocesses.ic as ic
import agnprocesses.gamma_gamma as gamma_gamma
import agnprocesses.spectra as spec
import agnprocesses.synchro as synchro
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

ELECTRON_REST_ENERGY = const.m_e.to(u.eV, u.mass_energy()).value
S_MIN = (2.0 * ELECTRON_REST_ENERGY)**2
print('Hey!')
#########################################################################
# device = 'cpu'  # 'cuda'
# # if False:
# if torch.cuda.is_available():
#     print("If cuda is available: ", torch.cuda.is_available())
#     device = 'cuda'
#     print("Using 'cuda'!!!")
#     print(torch.cuda.get_device_name())
# else:
#     device = 'cpu'
#     print("Using CPU!!!")
########################################################################
f_support_interpolation = torchinterp1d.interp1d


def torch_interpolation(x_old_tensor: torch.tensor,
                        y_old_tensor: torch.tensor,
                        x_new_tensor: torch.tensor):
    return(
        torch.squeeze(10.0**f_support_interpolation(
            torch.log10(x_old_tensor),
            torch.log10(y_old_tensor),
            torch.log10(x_new_tensor))
        )
    )


def generate_random_numbers(pdf,
                            pars=[],
                            shape=(1,),
                            xmin=torch.tensor(1.0,
                                              device=None,
                                              dtype=torch.float64),
                            xmax=torch.tensor(10.0,
                                              device=None,
                                              dtype=torch.float64),
                            normalized=False,  # deprecated
                            logarithmic=True,
                            n_integration=3000,
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
            if pars != []:
                try:
                    iter(pars[0])
                    if len(pars[0]) != len(xmin):
                        raise ValueError(
                            "Vectorized parameter in pars must have the same length as xmin and xmax do."
                        )
                    x = 10.0**((torch.log10(xmax[assignment]) -
                                torch.log10(xmin[assignment])) *
                               torch.rand((generate_size, ),
                                          device=device,
                                          dtype=torch.float64) +
                               torch.log10(xmin[assignment]))
                    p = pdf(x,
                            *[pars[0][assignment], *pars[1:]],
                            **kwargs) * x
                except (IndexError, TypeError) as exc:
                    x = 10.0**((torch.log10(xmax[assignment]) -
                                torch.log10(xmin[assignment])) *
                               torch.rand((generate_size, ),
                                          device=device,
                                          dtype=torch.float64) +
                               torch.log10(xmin[assignment]))
                    p = pdf(x,
                            *pars,
                            **kwargs) * x
            else:  # pars == []
                x = 10.0**((torch.log10(xmax[assignment]) -
                            torch.log10(xmin[assignment])) *
                           torch.rand((generate_size, ),
                                      device=device,
                                      dtype=torch.float64) +
                           torch.log10(xmin[assignment]))
                p = pdf(x, **kwargs) * x
        else:
            try:
                iter(pars[0])
                if len(pars[0]) != len(xmin):
                    raise ValueError(
                        "Vectorized parameter in pars must have the same length as xmin and xmax do."
                    )
                x = ((xmax[assignment] - xmin[assignment]) *
                     torch.rand((generate_size, ),
                                device=device,
                                dtype=torch.float64) + xmin[assignment])
                p = pdf(x,
                        *[pars[0][assignment], *pars[1:]],
                        **kwargs)
            except (IndexError, TypeError) as exc:
                x = ((xmax[assignment] - xmin[assignment]) *
                     torch.rand((generate_size, ),
                                device=device,
                                dtype=torch.float64) + xmin[assignment])
                p = pdf(x,
                        *pars,
                        **kwargs)
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
                                      device=None, dtype=torch.float64):
    """
    Generates tensor of random particle traveled distances
    at e_particle energy points given that the
    interaction rate r_particle_node are
    computed at e_particle_node energy points.
    """
    r_particle = torch_interpolation(
        e_particle_node,
        r_particle_node,
        e_particle
    )
    uniform_random = torch.rand(e_particle.shape, device=device,
                                dtype=torch.float64)
    return (-torch.log(uniform_random) / r_particle)


########################################################################
def generate_random_cosine(shape=(1,),
                           cosmin=-1.0,  # or tensor with size of shape
                           cosmax=1.0,  # or tensor with size of shape
                           device=None, dtype=torch.float64):
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
# def auxiliary_pair_production_function(x,  # min
#                                        x_max,
#                                        photon_field_file_path='',
#                                        device=None,
#                                        dtype=torch.float64):
#     dim2 = 100
#     ys = []
#     for i in range(0, max(x.shape)):
#         x_primed = torch.logspace(
#             float(torch.log10(x[i])),
#             float(torch.log10(x_max[i])),
#             dim2,
#             device=device
#         )
#         y = background_photon_density_interpolated(
#             x_primed,
#             photon_field_file_path=photon_field_file_path,
#             device=device, dtype=torch.float64) / x_primed**2
#         ys.append(torch.trapz(y, x_primed))
#     ys = torch.tensor(ys, device=device, dtype=torch.float64)
#     print(f'ys.shape = {ys.shape}')
#     return (ys)
#
# def pair_production_differential_cross_section_s(
#         s,
#         energy,
#         photon_field_file_path='',
#         device=None, dtype=torch.float64):
#     return (sigma_pair_cross_section_reduced(s) *
#             (auxiliary_pair_production_function(
#                 s / (4.0 * energy),
#                 eps_max * torch.ones(energy.shape,
#                                      device=device, dtype=torch.float64),
#                 photon_field_file_path=photon_field_file_path)
#              / (8.0 * energy**2))
#             )
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
                      device=None,
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


def calculate_windowed_max_min_ratio(y_input, x_input,
                                     std_window_width=5,
                                     device=None):
    y = torch.clone(y_input).reshape(np.max(y_input.shape))
    x = torch.clone(x_input).reshape(np.max(x_input.shape))
    max_min_ratio = torch.zeros((y.shape[0] - std_window_width, ))
    for i in range(0, max_min_ratio.shape[0]):
        sh = int(std_window_width / 2)
        y_ind = i + sh
        if std_window_width % 2 == 1:
            max_min_ratio[i] = (torch.max(y[y_ind - sh:y_ind + sh + 1]) /
                                torch.min(y[y_ind - sh:y_ind + sh + 1]))
        else:
            max_min_ratio[i] = (torch.max(y[y_ind - sh:y_ind + sh]) /
                                torch.min(y[y_ind - sh:y_ind + sh]))
    return max_min_ratio


def continuum_and_spikes(y_input, x_input,
                         std_window_width=5,
                         filt_value=3.0,
                         device=None,
                         empirical_line_correction_factor=2.0):
    y = torch.clone(y_input).reshape(np.max(y_input.shape))
    x = torch.clone(x_input).reshape(np.max(x_input.shape))
    sh = int(std_window_width / 2)
    max_min_ratio = calculate_windowed_max_min_ratio(
        y, x, std_window_width=std_window_width, device=device
    )
    if std_window_width % 2 == 1:
        x = x[sh + 1: -sh][:-1]
        y = y[sh + 1: -sh][:-1]
    else:
        x = x[sh:-sh][:-1]
        y = y[sh:-sh][:-1]
    filt_spike = (max_min_ratio > filt_value)
    difference = torch.diff(filt_spike)
    x_starters = x[difference == True][0::2]
    x_enders = x[difference == True][1::2]
    x_continuum = torch.clone(x)
    y_continuum = torch.clone(y)
    if len(x_starters) > len(x_enders):
        x_starters = x_starters[:-1]
    if len(x_enders) > len(x_starters):
        x_enders = x_enders[:-1]
    x_spikes = torch.zeros(x_starters.shape, device=device)
    y_spikes = torch.zeros(x_starters.shape, device=device)
    weight_individual_spikes = torch.zeros(y_spikes.shape, device=device)
    for i in range(0, len(x_starters)):
        start = int(torch.argwhere((x_starters[i] == x)))
        end = int(torch.argwhere((x_enders[i] == x)))
        x_spikes[i] = float(x[int((start + end) / 2)])
        y_spikes[i] = torch.max(
            y[start:(end + 1)]
        )
        weight_individual_spikes[i] = torch.trapezoid(y[start:(end + 1)],
                                                      x[start:(end + 1)])
        y_continuum[start:(end + 1)] = torch.min(y)
    weight_individual_spikes *= empirical_line_correction_factor
    weight_total_spikes = torch.sum(weight_individual_spikes)
    weight_continuum = torch.trapezoid(y_continuum, x_continuum)
    weight_total = weight_continuum + weight_total_spikes
    weight_continuum /= weight_total
    weight_total_spikes /= weight_total
    weight_individual_spikes /= weight_total
    # print("Probability of continuum is {:.3e}".format(weight_continuum))
    # print("Probability of spikes is {:.3e}".format(weight_total_spikes))
    return ((weight_continuum, weight_individual_spikes,
             x_continuum, y_continuum,
             x_spikes, y_spikes))


def generate_random_background_photon_energy(
    shape,
    field,
    device=None,
    energy_photon_min=None,
    energy_photon_max=None,
    std_window_width=5,
    filt_value=3.0,
    empirical_line_correction_factor=2.0,
    **kwargs
):
    epsilons = torch.zeros(shape, device=device, dtype=torch.float64)
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
    weight_global_continuum, weight_global_individual_spikes, \
        x_global_continuum, y_global_continuum, \
        x_global_spikes, y_global_spikes = continuum_and_spikes(
            field[:, 1], field[:, 0],
            std_window_width=std_window_width,
            filt_value=filt_value,
            device=device,
            empirical_line_correction_factor=empirical_line_correction_factor
        )
    # print("weight_global_continuum = {:.3e}".format(float(
    #     weight_global_continuum))
    # )
    # print("weight_global_lines = {:.3e}".format(float(
    #     1.0 - weight_global_continuum))
    # )
    # print("x_global_spikes:")
    # print(x_global_spikes)
    # print("x_global_continuum.shape = ", x_global_continuum.shape)
    spikes_question = torch.rand(xmax.shape,
                                 device=device,
                                 dtype=torch.float64)
    filt_spikes = torch.zeros(shape, device=device, dtype=torch.bool)
    spikes = []
    for i in range(0, shape[0]):
        filt_spikes_min_max = torch.logical_and(
            (x_global_spikes >= xmin[i]),
            (x_global_spikes <= xmax[i])
        )
        integral_spikes = torch.sum(
            weight_global_individual_spikes[filt_spikes_min_max]
        )
        spikes_question[i] *= (integral_spikes /
                               (1.0 - weight_global_continuum))
        filt_spikes[i] = (spikes_question[i] < integral_spikes)
        if filt_spikes[i] == True:
            spikes.append(
                float(np.random.choice(
                    (x_global_spikes[filt_spikes_min_max]
                     ).detach().to('cpu').numpy(),
                    p=(weight_global_individual_spikes[
                        filt_spikes_min_max
                    ] / integral_spikes).detach().to('cpu').numpy()
                ))
            )
    epsilons[filt_spikes] = torch.tensor(spikes,
                                         device=device,
                                         dtype=torch.float64)
    filt_continuum = torch.logical_not(filt_spikes)
    pdf = partial(torch_interpolation,
                  x_global_continuum,
                  y_global_continuum)
    epsilons[filt_continuum] = generate_random_numbers(
        pdf,
        pars=[],
        shape=xmin[filt_continuum].shape,
        xmin=xmin[filt_continuum],
        xmax=xmax[filt_continuum],
        logarithmic=True,
        n_integration=3000,
        device=device,
        **kwargs
    )
    return epsilons


def generate_random_s(energy,
                      field,
                      random_cosine=True,
                      device=None,
                      process='PP',  # or 'IC'
                      energy_ic_threshold=None,
                      energy_photon_max=None,
                      dtype=torch.float64,
                      std_window_width=5,
                      filt_value=3.0,
                      empirical_line_correction_factor=2.0,
                      **cosine_kwargs):
    if process == 'PP':
        energy_photon_min = S_MIN / (4.0 * energy) * 1.010
        epsilon = generate_random_background_photon_energy(
            energy.shape,
            field,
            device=device,
            energy_photon_min=energy_photon_min,
            energy_photon_max=energy_photon_max,
            std_window_width=std_window_width,
            filt_value=filt_value,
            empirical_line_correction_factor=empirical_line_correction_factor
        )
        if random_cosine:
            cosmax = 1.0 - S_MIN / (2.0 * epsilon * energy)
            filt_cosmax = (cosmax > 1.0)
            if len(cosmax[filt_cosmax] > 0):
                print("cosmax > 1! at {}".format(
                    len(filt_cosmax[filt_cosmax == True])
                ))
                # raise RuntimeError("cosmax > 1!")
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
            field,
            device=device,
            energy_photon_min=(eps_thr * ELECTRON_REST_ENERGY**2 /
                               (2.0 * (1.0 + b) * (1.0 - eps_thr) * energy)) * 1.010,
            energy_photon_max=energy_photon_max,
            std_window_width=std_window_width,
            filt_value=filt_value,
            empirical_line_correction_factor=empirical_line_correction_factor
        )
        if random_cosine:
            cosmax = (1.0 - (eps_thr * ELECTRON_REST_ENERGY**2) /
                            (2.0 * (1.0 - eps_thr) * energy * epsilon)
                      ) / b
            filt_cosmax = (cosmax > 1.0)
            if len(cosmax[filt_cosmax] > 0):
                print("cosmax > 1! at {}".format(
                    len(filt_cosmax[filt_cosmax == True])
                ))
                # raise RuntimeError("cosmax > 1!")
            cosine = generate_random_cosine(energy.shape,
                                            cosmax=cosmax,
                                            device=device,
                                            **cosine_kwargs)
        else:
            cosine = -1.0
        return (ELECTRON_REST_ENERGY**2 +
                2.0 * energy * epsilon * (1.0 - b * cosine))


def get_center_energy(energy_bins):
    return 10.0**((np.log10(energy_bins[1:]) +
                   np.log10(energy_bins[:-1])) /
                  2.0)


def get_energy_bins(center_energy_array):
    """
    Given an array with centers of energy bins returns an array of bins edges.
    """
    a = np.log10(center_energy_array[1] / center_energy_array[0])
    e0 = center_energy_array[0] / 10**(0.5 * a)
    return(e0 * 10.0**(a * np.arange(0,
                                     np.max(center_energy_array.shape),
                                     step=1)
                       )
           )


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
                          density=False,
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
        raise ValueError(
            "Only 'primary' or 'another' options of the reweight_array parameter are possible"
        )
    h = (h * n_particles_theory_old /
         number_of_particles)
    if return_1d:
        if reweight_array == 'primary':
            if density == False:
                return np.sum(h, axis=0).reshape(
                    another_energy_center.shape)
            else:
                eb = get_energy_bins(primary_energy_center)
                return np.sum(h * (eb[1:] - eb[:-1]), axis=0).reshape(
                    another_energy_center.shape)
        elif reweight_array == 'another':
            if density == False:
                return np.sum(h, axis=1).reshape(
                    primary_energy_center.shape)
            else:
                eb = get_energy_bins(another_energy_center)
                return np.sum(h * (eb[1:] - eb[:-1]), axis=1).reshape(
                    primary_energy_center.shape)
        else:
            raise ValueError(
                "Only 'primary' or 'another' options of the reweight_array parameter are possible"
            )
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
    primary_energy_center = get_center_energy(primary_energy_bins)
    another_energy_center = get_center_energy(energy_bins)
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
        density=density,
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
    minimum_injection_coord=None,
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
    if minimum_injection_coord is None or minimum_injection_coord < 0:
        minimum_injection_coord = 0.0
    for file in cash_files:
        tens = torch.load(file,
                          map_location=torch.device(device))
        if cascade_type_of_particles == 'all':
            filt_coord = (tens[3, :] >= minimum_injection_coord)
            hist = hist + make_2d_energy_histogram(
                tens[0, :][filt_coord],
                tens[1, :][filt_coord],
                bins_primary=primary_energy_bins,
                bins_another=energy_bins,
                verbose=verbose,
                density=density
            )
        elif cascade_type_of_particles == 'survived':
            filt_survived = (tens[0, :] == tens[1, :])
            filt_coord = (tens[3, :][filt_survived] >= minimum_injection_coord)
            hist = hist + make_2d_energy_histogram(
                tens[0, :][filt_survived][filt_coord],
                tens[1, :][filt_survived][filt_coord],
                bins_primary=primary_energy_bins,
                bins_another=energy_bins,
                verbose=verbose,
                density=density
            )
        elif cascade_type_of_particles == 'cascade':
            filt_cascade = torch.logical_not(tens[0, :] == tens[1, :])
            filt_coord = (tens[3, :][filt_cascade] >= minimum_injection_coord)
            hist = hist + make_2d_energy_histogram(
                tens[0, :][filt_cascade][filt_coord],
                tens[1, :][filt_cascade][filt_coord],
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
        region_size,  # : astropy.Quantity
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
        synchrotron_losses=False,
        magnetic_field_strength=None,
        # prefactor defining minimal synchrotron photon energy
        synchro_nu_min_prefactor=1.0e-01,
        # prefactor defining maximal synchrotron photon energy
        synchro_nu_max_prefactor=1.0e+01,
        std_window_width=5,
        filt_value=3.0,
        empirical_line_correction_factor=2.0,
        device=None,
        dtype=torch.float64):
    """
    gammas and electrons consist of 4 rows:

    0) primary energy of individual particles;
    1) current (observable) energy of individual particles;
    2) distance traveled by individual particles;
    3) distance of the primary particle injection [0; region_size).
    """
    t_start = get_current_date_time()
    print('Monte-Carlo process started: ', date_time_string_human_friendly())
    field_numpy = np.loadtxt(photon_field_file_path)
    field = torch.tensor(field_numpy, device=device, dtype=torch.float64)
    folder = 'data/torch/' + str(folder)
    try:
        os.makedirs(folder, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            "The electromagnetic_cascades output folder already exists!\n" +
            "Please, change the name of the folder parameter or rename/delete" +
            " the existing folder."
        )
    new_electrons = None
    new_positrons = None
    new_gammas = None
    shape = (primary_energy_tensor_size, )
    region_size = region_size.to(u.cm).value
    if energy_photon_max is not None:
        energy_photon_max = energy_photon_max.to(u.eV).value
    else:
        energy_photon_max = field[-1, 0]
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
            "Unknown option for the 'injection_place' parameter!\n" +
            "It must be either 'zero' or 'uniform'!"
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
    observable_energy_max = observable_energy_max.to(u.eV).value
    if observable_energy_max <= observable_energy_min:
        raise ValueError(
            "'observable_energy_min' must be less than 'observable_energy_max'!")
    en = np.logspace(np.log10(min([observable_energy_min,
                                   primary_energy_min])),
                     np.log10(max([observable_energy_max,
                                   primary_energy_max])),
                     10_000)
    el_min = min([observable_energy_min,
                  primary_energy_min]) * u.eV * 0.90
    el_max = max([observable_energy_max,
                  primary_energy_max]) * u.eV * 1.10
    if synchrotron_losses:
        if magnetic_field_strength is None:
            raise ValueError(
                "'magnetic_field_strength' should be defined as an astropy Quantity!"
            )
        else:
            try:
                magnetic_field_strength = magnetic_field_strength.to(u.G)
            except:
                try:
                    magnetic_field_strength = magnetic_field_strength.to(
                        u.g**0.5 * u.cm**(-0.5) * u.s**(-1)
                    )
                except:
                    raise ValueError(
                        "Magnetic field strength must be in gauss units!"
                    )
            if magnetic_field_strength.unit == u.G:
                magnetic_field_strength = (magnetic_field_strength.value *
                                           u.g**0.5 * u.cm**(-0.5) * u.s**(-1))
            elif magnetic_field_strength.unit == u.T:
                raise ValueError(
                    "Please, define 'magnetic_field_strength' in gauss units"
                )
            omega_B = ((const.e.gauss * magnetic_field_strength) /
                       (const.m_e * const.c)).to(u.s**(-1))
            print("omega_B = {:.3e}".format(omega_B))
            synchro_nu_min = (synchro_nu_min_prefactor * 3.0 * const.h *
                              (el_min / ELECTRON_REST_ENERGY / u.eV)**2 *
                              omega_B /
                              (4.0 * np.pi)).to(u.eV)
            print("minimal energy of synchrotron photons is {:.3e}".format(
                synchro_nu_min))
            synchro_nu_max = (synchro_nu_max_prefactor * 3.0 * const.h *
                              (el_max / ELECTRON_REST_ENERGY / u.eV)**2 *
                              omega_B /
                              (4.0 * np.pi)).to(u.eV)
            print("maximal energy of synchrotron photons is {:.3e}".format(
                synchro_nu_max))
            synchro_nu_array = np.logspace(
                np.log10(synchro_nu_min.value),
                np.log10(synchro_nu_max.value),
                100
            ) * u.eV  # array with synchrotron photon energies for synchro spec
            e_synchro_node = np.logspace(
                np.log10(el_min.value),
                np.log10(el_max.value),
                100
            ) * u.eV  # array with electron energies for losses dE/dt calculation
            de_dt_synchro_node = synchro.energy_losses_rate(
                e_synchro_node,
                magnetic_field_strength,
                particle_mass=const.m_e.cgs,
                particle_charge=const.e.gauss,
                synchro_nu_min_prefactor=synchro_nu_min_prefactor,
                synchro_nu_max_prefactor=synchro_nu_max_prefactor
            )
            de_dt_synchro_node = spec.to_current_energy(
                en * u.eV,
                e_synchro_node,
                de_dt_synchro_node
            ).to(u.eV / u.s)
            e_synchro_node = torch.tensor(
                en, device=device, dtype=dtype
            )  # eV
            de_dt_synchro_node = torch.tensor(
                de_dt_synchro_node.value, device=device, dtype=dtype
            )  # eV / s

    e_gamma_node, r_gamma_node = gamma_gamma.interaction_rate(
        photon_field_file_path,
        el_min,
        el_max,
        background_photon_energy_unit=background_photon_energy_unit,
        background_photon_density_unit=background_photon_density_unit)
    e_ic_node, r_ic_node = ic.IC_interaction_rate(
        photon_field_file_path,
        el_min,
        el_max,
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
                r_gamma_node,
                device=device
            )
            gammas[2, :] += traveled
        if no_electrons == False:
            traveled_electrons = generate_random_distance_traveled(
                electrons[1, :],
                e_ic_node,
                r_ic_node,
                device=device
            )
            electrons[2, :] += traveled_electrons
        ###################################################################
        # Save out escaped particles and
        # update gammas and electrons removing escaped particles.
        if no_gammas == False:  # i.e. there are some gamma rays in the moment
            filter_escaped = torch.logical_or(
                (gammas[2, :] > region_size),
                (gammas[1, :] < observable_energy_min)
            )
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
            filter_escaped = torch.logical_or(
                (electrons[2, :] > region_size),
                (electrons[1, :] < observable_energy_min)
            )
            filter_not_escaped = torch.logical_not(filter_escaped)
            # distance covered by electrons which did not escape :
            # (at the current step)
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
            energy_photon_min = S_MIN / (4.0 * gammas[1, :]) * 1.010
            filt_more_than_max = (energy_photon_min > energy_photon_max)
            # remove sterile gamma rays
            nless = len(filt_more_than_max[filt_more_than_max == True])
            if nless > 0:
                print(f"Removing {nless} gamma rays under the threshold")
                torch.save(torch.stack([gammas[0, :][filt_more_than_max],
                                        gammas[1, :][filt_more_than_max],
                                        gammas[2, :][filt_more_than_max],
                                        gammas[3, :][filt_more_than_max]]),
                           folder + '/' + 'gammas_' +
                           'step000' + str(nstep) +
                           '_' + date_time_string_human_friendly() + '.pt')
                filt_less_than_max = torch.logical_not(filt_more_than_max)
                gammas = torch.stack([gammas[0, :][filt_less_than_max],
                                      gammas[1, :][filt_less_than_max],
                                      gammas[2, :][filt_less_than_max],
                                      gammas[3, :][filt_less_than_max]])
            s_gammas = generate_random_s(
                gammas[1, :],
                field,
                random_cosine=True,
                process='PP',
                device=device,
                energy_photon_max=energy_photon_max,
                std_window_width=std_window_width,
                filt_value=filt_value,
                empirical_line_correction_factor=empirical_line_correction_factor
            )
            filt_s_gammas = (s_gammas >= S_MIN)
            print("There are {:d} gamma interactions under S_MIN".format(
                len(s_gammas[torch.logical_not(filt_s_gammas)]))
            )
            # s_gammas = s_gammas[filt_s_gammas]
        if no_electrons == False:
            eps_thr = energy_ic_threshold / electrons[1, :]
            s_electrons = generate_random_s(
                electrons[1, :],
                field,
                random_cosine=True,
                process='IC',
                energy_ic_threshold=energy_ic_threshold,
                device=device,
                energy_photon_max=energy_photon_max,
                std_window_width=std_window_width,
                filt_value=filt_value,
                empirical_line_correction_factor=empirical_line_correction_factor
            )
            filt_s_electrons = (s_electrons >= (ELECTRON_REST_ENERGY**2 /
                                                (1.0 - eps_thr)))
            print(
                "There are {:d} electron interactions under ELECTRON_REST_ENERGY**2 / (1.0 - eps_thr)".format(
                    (filt_s_electrons[filt_s_electrons == False]).shape[0])
            )
            # s_electrons = s_electrons[filt_s_electrons]
            # eps_thr = eps_thr[filt_s_electrons]
            # traveled_electrons = traveled_electrons[filt_s_electrons]
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


if __name__ == '__main__':
    path = 'data/PKS1510-089/nph'
    field_numpy = np.loadtxt(path)
    emin = field_numpy[0, 0]
    emax = field_numpy[-1, 0]
    s = 1000
    a = np.log10(emax / emin) / s
    r = np.arange(0, s + 1, step=1)
    bins = emin * 10.0**(a * r)
    field = torch.tensor(field_numpy, device='cuda', dtype=torch.float64)
    t_start_numpy = get_current_date_time()
    print("Sampling started at: ",
          date_time_string_human_friendly())
    shape = (100_000, )
    epsilons = generate_random_background_photon_energy(
        shape,
        field,
        device='cuda',
        energy_photon_min=emin,
        energy_photon_max=emax,
        std_window_width=5,
        filt_value=3.0,
        empirical_line_correction_factor=2.0
    ).detach().to('cpu').numpy()
    t_end_numpy = get_current_date_time()
    print("Sampling ended at: ",
          date_time_string_human_friendly())
    print("Sampling took ", t_end_numpy - t_start_numpy)
    print("Number of zeros = ", epsilons[epsilons == 0.0].shape[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    filt = np.logical_and((field_numpy[:, 0] >= emin),
                          (field_numpy[:, 0] <= emax))
    x = field_numpy[:, 0][filt]
    y = field_numpy[:, 1][filt] / simps(field_numpy[:, 1][filt],
                                        field_numpy[:, 0][filt])
    plt.plot(
        x, y,
        marker=None,
        linestyle='-',
        linewidth=1,
        color='k',
        label='field from the file'
    )
    plt.hist(
        epsilons, bins=bins,
        density=True,
        log=True,
        color='b',
        histtype='step',
        linewidth=1,
        zorder=100
    )

    plt.xlabel('energy, ', fontsize=18)
    plt.xticks(fontsize=12)
    plt.ylabel('PDF', fontsize=18)
    plt.yticks(fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(1e+02, 1e+03)
    # ax.set_ylim(1e-10, 1e-06)
    plt.legend()  # loc='upper left')
    # plt.savefig("numpy2.pdf")
    plt.show()
