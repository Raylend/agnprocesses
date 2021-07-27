import torch
import numpy as np
import processes.ic as ic
import processes.gamma_gamma as gamma_gamma
import processes.spectra as spec
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

# if __name__ == '__main__':
ELECTRON_REST_ENERGY = const.m_e.to(u.eV, u.mass_energy()).value
S_MIN = (2.0 * ELECTRON_REST_ENERGY)**2
print('Hey!')
device = 'cuda'
# if False:
# if torch.cuda.is_available():
#     print("If cuda is available: ", torch.cuda.is_available())
#     device = 'cuda'
#     print("Using 'cuda'!!!")
#     print(torch.cuda.get_device_name())
# else:
#     device = 'cpu'
#     print("Using CPU!!!")
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
        print("Shape of c: ", c.shape)
        print("Shape of NaNs of c: ", c[torch.isnan(c)].shape)
        print("xmin for Nans: ", xmin[torch.isnan(c)])
        print("xmax for Nans: ", xmax[torch.isnan(c)])
        raise RuntimeError("Nan occured in c!")
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
# def exponential_distribution(x: torch.tensor,
#                              lambda_x: torch.tensor):
#     """
#     Returns pdf of the exponential distribution at x given that
#     for each value of an element of x there is a correpsonding value of
#     an element of lambda.
#     """
#     return (lambda_x * torch.exp(-lambda_x * x))
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
def generate_random_y(s,
                      eps_thr=None,
                      logarithmic=True,
                      device=device,
                      dtype=torch.float64,
                      process='PP'):  # or 'IC', i.e.
    # Pair Production or Inverse Compt
    """
    Generates random fraction of the squared total energy s for
    electrons (positrons) produced in gamma-gamma collisions
    in accordance with
    pair_production_differential_cross_section_y(y, s).

    Returns a torch.tensor of the requested shape.
    """
    shape = s.shape
    if process == 'PP':
        return generate_random_numbers(
            pair_production_differential_cross_section_y,
            [s, ],
            shape=shape,
            xmin=ELECTRON_REST_ENERGY**2 / s * 1.010,
            xmax=torch.ones(shape, device=device,
                            dtype=dtype) * 0.990,
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
            xmin=ELECTRON_REST_ENERGY**2 / s * 1.010,
            xmax=(torch.ones(shape, device=device,
                             dtype=dtype) - eps_thr) * 0.990,
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
    density=True
):
    if type(primary_energy) == torch.Tensor:
        primary_energy = primary_energy.detach().to('cpu').numpy()
    if type(another_energy) == torch.Tensor:
        another_energy = another_energy.detach().to('cpu').numpy()
    if bins_primary is None:
        bins_primary = 20
    if bins_another is None:
        bins_another = 20
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
                          return_1d=True):
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
    k_modif = (new_spectrum(x, *new_params, **new_keyargs) /
               original_spectrum(x, *original_params, **original_keyargs))
    histogram_2d = histogram_2d * k_modif
    if return_1d:
        if reweight_array == 'primary':
            return np.sum(histogram_2d, axis=0)
        elif reweight_array == 'another':
            return np.sum(histogram_2d, axis=1)
        else:
            return np.sum(histogram_2d, axis=0)
    else:
        return histogram_2d


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
        reweight_array='primary'):
    if type(primary_energy) == torch.Tensor:
        primary_energy = primary_energy.detach().to('cpu').numpy()
    if type(energy_array) == torch.Tensor:
        energy_array = energy_array.detach().to('cpu').numpy()
    if primary_energy_bins is None:
        primary_energy_bins = energy_bins
    histogram_2d = make_2d_energy_histogram(
        primary_energy,
        energy_array,
        bins_primary=primary_energy_bins,
        bins_another=energy_bins,
        verbose=False,
        density=False
    )
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
        return_1d=True
    ) / len(energy_array)
    spectrum = spectrum / (energy_bins[1:] - energy_bins[:-1])
    return (another_energy_center, spectrum * another_energy_center**2)


########################################################################
def get_date_time():
    now = datetime.now()
    return (now.strftime("%d-%m-%Y_%H-%M-%S"))  # dd-mm-YY H-M-S


#######################################################################
def monte_carlo_process(
        primary_energy_tensor_size: int,
        region_size,  # : astropy.Quantity
        photon_field_file_path: str,
        primary_spectrum=spec.power_law,
        primary_spectrum_params=[1.0, 1.0, 1.0],
        primary_energy_min=1.0e+08 * u.eV,
        primary_energy_max=1.0e+13 * u.eV,
        energy_ic_threshold=1.0e+07 * u.eV,
        primary_energy_tensor_user=None,
        primary_particle='gamma',  # or 'electron'
        observable_energy_min=1.0e+08 * u.eV,
        observable_energy_max=1.0e+13 * u.eV,
        background_photon_energy_unit=u.eV,
        background_photon_density_unit=(u.eV * u.cm**3)**(-1),
        energy_photon_max=None,
        terminator_step_number=None,
        folder='output',
        device=device, dtype=torch.float64):
    """
    gammas and electrons consist of 3 rows:

    0) primary energy of individual particles;
    1) current (observable) energy of individual particles;
    2) distance traveled by individual particles.
    """
    print('Monte-Carlo process started: ', get_date_time())
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
    gammas = torch.zeros((3, primary_energy_tensor_size),
                         device=device, dtype=torch.float64)
    electrons = torch.zeros((3, primary_energy_tensor_size),
                            device=device, dtype=torch.float64)
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
                xmin=(primary_energy_min.to(u.eV).value *
                      torch.ones(shape, device=device,
                                 dtype=torch.float64)),
                xmax=(primary_energy_max.to(u.eV).value *
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
                xmin=(primary_energy_min.to(u.eV).value *
                      torch.ones(shape, device=device,
                                 dtype=torch.float64)),
                xmax=(primary_energy_max.to(u.eV).value *
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
    en = np.logspace(np.log10(observable_energy_min),
                     np.log10(observable_energy_max),
                     10_000)
    e_gamma_node, r_gamma_node = gamma_gamma.interaction_rate(
        photon_field_file_path,
        observable_energy_min * u.eV * 0.90,
        observable_energy_max * u.eV * 1.10,
        background_photon_energy_unit=background_photon_energy_unit,
        background_photon_density_unit=background_photon_density_unit)
    e_ic_node, r_ic_node = ic.IC_interaction_rate(
        photon_field_file_path,
        observable_energy_min * u.eV * 0.90,
        observable_energy_max * u.eV * 1.10,
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
    while (len(gammas) > 0) or (len(electrons) > 0):
        print(f"Step number {nstep}:")
        # Sample the distance traveled
        if no_gammas == False:
            traveled = generate_random_distance_traveled(
                gammas[1, :],
                e_gamma_node,
                r_gamma_node)
            print("Median distance traveled by gamma rays at this step: {:.3e} cm = {:.2f} % of region size".format(
                torch.median(traveled),
                torch.median(traveled) / region_size * 100.0
            ))
            gammas[2, :] += traveled
        if no_electrons == False:
            traveled = generate_random_distance_traveled(
                electrons[1, :],
                e_ic_node,
                r_ic_node)
            electrons[2, :] += traveled
            print("Median distance traveled by electrons at this step: {:.3e} cm = {:.2f} % of region size".format(
                torch.median(traveled),
                torch.median(traveled) / region_size * 100.0
            ))
        ###################################################################
        # Save out escaped particles and
        # update gammas and electrons removing escaped particles.
        if no_gammas == False:
            filter_escaped = (gammas[2, :] > region_size)
            filter_escaped = torch.logical_or(filter_escaped,
                                              (gammas[1, 0] <
                                               observable_energy_min))
            filter_not_escaped = torch.logical_not(filter_escaped)
            if len(filter_escaped[filter_escaped == True]) > 0:
                torch.save(torch.stack([gammas[0, :][filter_escaped],
                                        gammas[1, :][filter_escaped],
                                        gammas[2, :][filter_escaped]]),
                           folder + '/' + 'gammas_' +
                           'step000' + str(nstep) +
                           '_' + get_date_time() + '.pt')
            gammas = torch.stack([gammas[0, :][filter_not_escaped],
                                  gammas[1, :][filter_not_escaped],
                                  gammas[2, :][filter_not_escaped]])
        if no_electrons == False:
            filter_escaped = (electrons[2, :] > region_size)
            filter_escaped = torch.logical_or(filter_escaped,
                                              (electrons[1, 0] <
                                               observable_energy_min))
            filter_not_escaped = torch.logical_not(filter_escaped)
            if len(filter_escaped[filter_escaped == True]) > 0:
                torch.save(torch.stack([electrons[0, :][filter_escaped],
                                        electrons[1, :][filter_escaped],
                                        electrons[2, :][filter_escaped]]),
                           folder + '/' + 'electrons_' +
                           'step000' + str(nstep) +
                           '_' + get_date_time() + '.pt')
            electrons = torch.stack([electrons[0, :][filter_not_escaped],
                                     electrons[1, :][filter_not_escaped],
                                     electrons[2, :][filter_not_escaped]])
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
        if no_electrons == False:
            s_electrons = generate_random_s(
                electrons[1, :],
                photon_field_file_path,
                random_cosine=True,
                process='IC',
                energy_ic_threshold=energy_ic_threshold,
                device=device,
                energy_photon_max=energy_photon_max)
        ###################################################################
        # Sample transfered energy
        if no_gammas == False:
            y_from_gammas = generate_random_y(s_gammas,
                                              process='PP',
                                              logarithmic=True,
                                              device=device,
                                              dtype=torch.float64)
        if no_electrons == False:
            eps_thr = energy_ic_threshold / electrons[1, :]
            y_from_electrons = generate_random_y(s_electrons,
                                                 eps_thr=eps_thr,
                                                 process='IC',
                                                 logarithmic=True,
                                                 device=device,
                                                 dtype=torch.float64)
            filt_greater_than_one = (y_from_electrons > 1.0)
            if True in filt_greater_than_one:
                warnings.warn(
                    "There is y_from_electrons greater than one!",
                    UserWarning)
                print("y_from_electrons[filt_greater_than_one] = ",
                      y_from_electrons[filt_greater_than_one])
                y_from_electrons[filt_greater_than_one] = (
                    1.0 - eps_thr[filt_greater_than_one]
                )
        ###################################################################
        # Create cascade particles
        if no_gammas == False:
            new_electrons = torch.stack(
                [gammas[0, :],
                 y_from_gammas * gammas[1, :],
                 gammas[2, :]])
            new_positrons = torch.stack(
                [gammas[0, :],
                 (1.0 - y_from_gammas) * gammas[1, :],
                 gammas[2, :]])
            new_electrons = torch.cat([new_electrons, new_positrons],
                                      dim=1)
            new_positrons = None
            gammas[1, :] = 0.0  # all gammas were absorbed
            no_gammas = True
        if no_electrons == False:
            new_gammas = torch.stack(
                [electrons[0, :],
                 (1.0 - y_from_electrons) * electrons[1, :],
                 electrons[2, :]])
            electrons = torch.stack(
                [electrons[0, :],
                 y_from_electrons * electrons[1, :],
                 electrons[2, :]])
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
        if (nstep > terminator_step_number):
            break
        if no_electrons == True and no_gammas == True:
            break
    print("Done!")
    print('Monte-Carlo process finished: ', get_date_time())
    return None


########################################################################
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# start_event.record()
# ########################################################################
# # #1 generate tensor with initial energy of primary gamma rays
# primary_energy = primary_energy_0 * torch.ones((size,), device=device, dtype=torch.float64)
# ########################################################################
# # #2 sample distance traveled by primary gamma rays
# print("For {:e} eV interaction rate is {} 1 / cm".format(
#     primary_energy_0,
#     torch_interpolation(torch.tensor([primary_energy_0, ],
#                                      device=device, dtype=torch.float64),
#                         energy_gamma_node,
#                         r_gamma_gamma_node)
# )
# )
# distance_traveled = generate_random_gamma_distance_traveled(
#     primary_energy,
#     energy_gamma_node,
#     r_gamma_gamma_node
# )
# print("mean: {:.2e} cm".format(torch.mean(distance_traveled)))
# ########################################################################
# # #3 Get escaped and interacted gamma rays
# region_size = 0.1 * r_blr
# print("region size = {:.2} cm = {:.2} [r_blr]".format(
#     region_size, region_size / r_blr
# ))
# escaped = (distance_traveled > region_size)
# escaped_energy = primary_energy[escaped]
# interacted_energy = primary_energy[torch.logical_not(escaped)]
# print("Interacted {:.6f} % of photons".format(
#     (1 - len(escaped_energy) / len(primary_energy)) * 100.0)
# )
# print("Escaped {:.6f} % of photons".format(
#     (len(escaped_energy) / len(primary_energy)) * 100.0)
# )
# ########################################################################
# # #4 Generate s
# s = generate_random_s(interacted_energy,
#                       field,
#                       random_cosine=True,
#                       device=device, dtype=torch.float64)
# # x = S_MIN / (4.0 * interacted_energy)
# # x_max = torch.ones(interacted_energy.shape, device=device, dtype=torch.float64) * 10.0  # eV
# # aux = auxiliary_pair_production_function(x,
# #                                          x_max,
# #                                          photon_field_file_path=field,
# #                                          device=device, dtype=torch.float64)
# print("Mean square root of s is {:.3e}".format((
#     torch.mean(s)
# )**0.5))
# print("Max square root of s is {:.3e}".format((
#     torch.max(s)
# )**0.5))
# print("Min square root of s is {:.3e}".format((
#     torch.min(s)
# )**0.5))
# # #5 Choose s >= (2 * ELECTRON_REST_ENERGY)**2
# filt_s = (s >= S_MIN)
# s = s[filt_s]
# print("Fraction of gamma rays generated pairs: {:.6f} %".format(
#     len(s) / len(interacted_energy) * 100.0
# ))
# # #6 Compare with theoretical differential cross section
# theory_size = 10000
# s_theory = torch.logspace(torch.log10(torch.min(s) * 1.1),
#                           torch.log10(torch.max(s)),
#                           theory_size,
#                           device=device, dtype=torch.float64)
# rho_s = pair_production_differential_cross_section_s(
#     s_theory,
#     torch.ones(s_theory.shape, device=device, dtype=torch.float64) * torch.max(primary_energy),
#     photon_field_file_path=field,
#     device=device, dtype=torch.float64)
# print(f'rho_s = {rho_s}')
# # sigma_s = sigma_pair_cross_section_reduced(s_theory)
# # # ########################################################################
# # # # #6 Generate electron energy
# # # y = generate_random_y(s,
# # #                       logarithmic=True,
# # #                       device=device, dtype=torch.float64)
# # # print("Mean y = {:.2e}".format(torch.mean(y)))
# # # electron_energy = y * interacted_energy
# # # positron_energy = (1.0 - y) * interacted_energy
# # # lepton_energy = torch.cat((electron_energy, positron_energy), dim=-1)
# # # ########################################################################
# # # # Run some things here
# # # # E_gamma = torch.tensor(E_gamma, device=device, dtype=torch.float64)
# # # # print(E_gamma.device)
# # # # x = spec.power_law(E_gamma, 2.0, en_ref=1.0, norm=1.0)
# # # # print(x.device)
# # # # print(x)
# # # # x, p, z = generate_random_numbers(spec.power_law,
# # # #                                   [1, 100.0, 10.0],
# # # #                                   shape=(1_000_000,),
# # # #                                   xmin=torch.tensor(1.,
# # # #                                                     device=device, dtype=torch.float64),
# # # #                                   xmax=torch.tensor(1000.,
# # # #                                                     device=device, dtype=torch.float64),
# # # #                                   normalized=False,
# # # #                                   logarithmic=True,
# # # #                                   n_integration=300,
# # # #                                   device=device,
# # # #                                   verbose=True)
# # # # x = x.detach().to('cpu').numpy()
# # # # p = p.detach().to('cpu').numpy()
# # # # z = z.detach().to('cpu').numpy()
# # # #
# # # # gammas = torch.tensor([x,
# # # #                      np.zeros(x.shape),
# # # #                      np.ones(x.shape),
# # # #                      np.ones(x.shape)],
# # # #                     device=device, dtype=torch.float64)
# # # # print(gammas.shape)
# # # # ########################################################################
# # # # photon_field_file_path = (
# # # #     "/home/raylend/Science/agnprocesses/data/PKS1510-089/nph"
# # # # )
# # # # epsilon = generate_random_background_photon_energy(
# # # #     (100_000,),
# # # #     photon_field_file_path,
# # # #     device=device,
# # # #     energy_photon_min=None  # (0.05 * ELECTRON_REST_ENERGY**2 /
# # # #     # max(primary_energy))
# # # # )
# # # # field = np.loadtxt(photon_field_file_path)
# # # # # 0.05 * ELECTRON_REST_ENERGY**2 / max(primary_energy)
# # # # energy_photon_min = field[0, 0]
# # # # print(f"energy_photon_min = {energy_photon_min}")
# # # # energy_photon_max = field[-1, 0]
# # # # print(f"energy_photon_max = {energy_photon_max}")
# # # # pdf = partial(background_photon_density_interpolated,
# # # #               photon_field_file_path=photon_field_file_path,
# # # #               device=device, dtype=torch.float64)
# # # # en = torch.logspace(torch.log10(torch.tensor(energy_photon_min,
# # # #                                              device=device, dtype=torch.float64)),
# # # #                     torch.log10(torch.tensor(energy_photon_max,
# # # #                                              device=device, dtype=torch.float64)),
# # # #                     10000,
# # # #                     device=device, dtype=torch.float64)
# # # # density = pdf(en)
# # # # norm = torch.trapz(density, en)
# # # # mean_photon_energy = torch.trapz(density * en, en) / norm
# # # # print("mean photon energy = ", mean_photon_energy)
# # # # density = (density.detach().to('cpu').numpy() /
# # # #            norm.detach().to('cpu').numpy())
# # # # en = en.detach().to('cpu').numpy()
# # # # epsilon = epsilon.detach().to('cpu').numpy()
# # # ########################################################################
# end_event.record()
# torch.cuda.synchronize()  # Wait for the events to be recorded!
# elapsed_time_ms = start_event.elapsed_time(end_event)
# print(elapsed_time_ms)
# fig, ax = plt.subplots(figsize=(8, 6))
# x = s_theory.detach().to('cpu').numpy()
#
# h = plt.hist(s.detach().to('cpu').numpy(),
#              bins=x[::100],
#              density=True,
#              log=False,
#              label='histogram',
#              alpha=0.5)
#
# y = rho_s.detach().to('cpu').numpy()
# filt_y = (y > 0)
# y = y[filt_y]
# x = x[filt_y]
# integral = simps(y, x)
# y = y / integral
#
# plt.plot(x,
#          y,
#          linewidth=3,
#          linestyle='-',
#          label='theoretical differential distribution')
#
# # h = plt.hist(epsilon,
# #              density=True,
# #              bins=field[0::10, 0],
# #              log=True,
# #              label='histogram')
# # #
# # # plt.plot(z, p,
# # #          marker=None,
# # #          linestyle='-',
# # #          linewidth=3,
# # #          label='pdf line')
# #
# # # plt.plot(field[:, 0], field[:, 1],
# # #          marker=None,
# # #          linestyle='--',
# # #          linewidth=2,
# # #          label='original',
# # #          zorder=10)
# # #
# # plt.plot(en, density,
# #          marker=None,
# #          linestyle='-',
# #          linewidth=3,
# #          label='interpolation')
# #
# # # plt.xlabel('energy, ' + 'eV', fontsize=18)
# # plt.ylabel('SED, ' + r'eV cm$^{-2}$s$^{-1}$', fontsize=18)
# # plt.xticks(fontsize=12)
# # plt.yticks(fontsize=12)
# plt.xlabel('s, ' + r'eV$^{2}$', fontsize=18)
# plt.ylabel(r'$\rho$', fontsize=18)
# ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_xlim(1.0e+08, 1.0e+12)
# # ax.set_ylim(1.0e-01, 3.0e+02)
# # ax.grid()
# # ax.grid()
# # fig.savefig('test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf')
# plt.legend()
# # plt.legend(loc='upper left')
# plt.show()
