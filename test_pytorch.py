import torch
import numpy as np
import processes.gamma_gamma as gamma_gamma
import processes.spectra as spec
from astropy import units as u
from astropy import constants as const
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import torchinterp1d
from functools import partial
import warnings
from scipy.integrate import simps

# if __name__ == '__main__':
ELECTRON_REST_ENERGY = const.m_e.to(u.eV, u.mass_energy()).value
S_MIN = (2.0 * ELECTRON_REST_ENERGY)**2
print('Hey!')
# if False:
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
r_blr = ((0.036 * u.pc).to(u.cm)).value  # BLR radius in cm
print("r_blr = {:.3e}".format(r_blr))
size = 100_000
primary_energy_0 = 1.0e+12  # eV
# field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/nph"
field = "/home/raylend/Science/agnprocesses/data/test_1eV.txt"
en_ph = np.logspace(-5, 1.5, 6852) * u.eV
n_ph = spec.greybody_spectrum(en_ph, 1.0 * u.eV)
grey = spec.create_2column_table(en_ph, n_ph)
np.savetxt(field, grey, fmt="%.6e")
energy_gamma_node, r_gamma_gamma_node = gamma_gamma.interaction_rate(
    field,
    1.0e+08 * u.eV,
    primary_energy_0 * u.eV * 5,
    background_photon_energy_unit=u.eV,
    background_photon_density_unit=(u.eV * u.cm**3)**(-1)
)
energy_gamma_node = torch.tensor(energy_gamma_node.value,
                                 device=device)
r_gamma_gamma_node = torch.tensor(r_gamma_gamma_node.value,
                                  device=device)
########################################################################


def torch_interpolation(x_new_tensor: torch.tensor,
                        x_old_tensor: torch.tensor,
                        y_old_tensor: torch.tensor):
    f_support = torchinterp1d.Interp1d()
    return(
        torch.squeeze(f_support(x_old_tensor,
                                y_old_tensor,
                                x_new_tensor))
    )

########################################################################


def background_photon_density_interpolated(energy_new,
                                           photon_field_file_path='',
                                           device=device):
    field = np.loadtxt(photon_field_file_path)
    energy_old = torch.tensor(field[:, 0],
                              device=device)
    density_old = torch.tensor(field[:, 1],
                               device=device)
    return torch_interpolation(energy_new,
                               energy_old,
                               density_old)

########################################################################


def generate_random_numbers(pdf,
                            pars=[],
                            shape=(1,),
                            xmin=torch.tensor(1.0,
                                              device=device),
                            xmax=torch.tensor(10.0,
                                              device=device),
                            normalized=False,
                            logarithmic=True,
                            n_integration=1000,
                            device=None,
                            verbose=False,
                            **kwargs):
    if logarithmic:
        z = torch.logspace(torch.min(torch.log10(xmin)),
                           torch.max(torch.log10(xmax)),
                           n_integration,
                           device=device)
    else:
        z = torch.linspace(torch.min(xmin),
                           torch.max(xmax),
                           n_integration,
                           device=device)
    y_theory = pdf(z, *pars, **kwargs)
    if not normalized:
        norm = torch.trapz(y_theory, z)
    else:
        norm = 1.0
    y_theory = y_theory / norm
    x_final = float('Inf') * torch.ones(shape,
                                        device=device)
    generate_size = max(shape)
    # filt_main = (torch.ones(xmin.shape,
    #                         device=device) == 1)
    # elements of filt_main are True if the ranodm value has not been yet generated
    if logarithmic:
        # c1 = torch.log(torch.tensor(xmax / xmin, device=device))
        # c2 = xmax - xmin
        c = torch.max(y_theory * z)  # * c1 / c2)
    else:
        c = torch.max(y_theory)
    while generate_size > 0:
        if logarithmic:
            x = 10.0**((torch.log10(xmax) -
                        torch.log10(xmin)) *
                       torch.rand((generate_size,),
                                  device=device) +
                       torch.log10(xmin))
            p = (pdf(x, *pars, **kwargs) / norm * x)
        else:
            x = (xmax - xmin) * torch.rand(generate_size,
                                           device=device) + xmin
            p = pdf(x, *pars, **kwargs) / norm
        y_new = torch.rand((generate_size,),
                           device=device) * c
        filt = (y_new < p)
        x[torch.logical_not(filt)] = float('Inf')
        x_final[x_final == float('Inf')] = x
        xmax = xmax[torch.logical_not(filt)]
        xmin = xmin[torch.logical_not(filt)]
        for i, param in enumerate(pars):
            try:
                iter(param)
                pars[i] = param[torch.logical_not(filt)]
            except TypeError:
                pass
        generate_size = len(x_final[x_final == float('Inf')])
    if verbose:
        return (x_final, y_theory, z)
    else:
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


def generate_random_gamma_distance_traveled(e_gamma: torch.tensor,
                                            e_gamma_node: torch.tensor,
                                            r_gamma_node: torch.tensor,
                                            device=device):
    """
    Generates tensor of random gamma-ray traveled distances
    at e_gamma energy points given that gamma-gamma
    interaction rate r_gamma_node are
    computed at e_gamma_node energy points.
    """
    r_gamma = torch_interpolation(
        e_gamma,
        e_gamma_node,
        r_gamma_node
    )
    uniform_random = torch.rand(e_gamma.shape, device=device)
    return (-torch.log(uniform_random) / r_gamma)

########################################################################


def generate_random_cosine(shape=(1,),
                           cosmin=-1.0,  # or tensor with size of shape
                           cosmax=1.0,  # or tensor with size of shape
                           device=device):
    return(
        (cosmax - cosmin) * torch.rand(max(shape),
                                       device=device) + cosmin
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
                                       device=device):
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
            device=device) / x_primed**2
        ys.append(torch.trapz(y, x_primed))
    ys = torch.tensor(ys, device=device)
    print(f'ys.shape = {ys.shape}')
    return (ys)


#######################################################################
eps_max = 1000.0  # eV
# s_precalc = torch.logspace(float(torch.log10(torch.tensor(
#     [S_MIN / (4.0 * 1.0e+15), ], device=device))),
#     float(torch.log10(torch.tensor(
#         [10000.0, ], device=device))),
#     10000,
#     device=device)
#
# print(f"s_precalc.shape = {s_precalc.shape}")
#
# f_precalc = auxiliary_pair_production_function(s_precalc,
#                                                photon_field_file_path=field,
#                                                device=device)
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
        device=device):
    return (sigma_pair_cross_section_reduced(s) *
            (auxiliary_pair_production_function(
                s / (4.0 * energy),
                eps_max * torch.ones(energy.shape,
                                     device=device),
                photon_field_file_path=photon_field_file_path)
             / (8.0 * energy**2))
            )


########################################################################
def generate_random_s_new(energy,
                          device=device):
    return (generate_random_numbers(
        pair_production_differential_cross_section_s,
        pars=[energy, ],
        shape=energy.shape,
        xmin=S_MIN * torch.ones(energy.shape,
                                device=device),
        xmax=4.0 * energy * eps_max * torch.ones(energy.shape,
                                                 device=device),
        normalized=True,
        logarithmic=True,
        n_integration=len(energy),
        device=device,
        verbose=False)
    )


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
    beta = (1 - S_MIN / s)**0.5
    t11 = y**2 / (1 - y) + 1 - y
    t12 = (1 - beta**2) / (1 - y)
    t13 = (-(1 - beta**2)**2) / (4 * y * (1 - y)**2)
    t1 = t11 + t12 + t13
    t2 = 1 + (2 * beta**2) * (1 - beta**2)
    return (t1 / (y * t2))


#######################################################################
def generate_random_y(s,
                      logarithmic=True,
                      device=device,
                      **kwargs):
    """
    Generates random fraction of the squared total energy s for
    electrons (positrons) produced in gamma-gamma collisions
    in accordance with
    pair_production_differential_cross_section_y(y, s).

    Returns a torch.tensor of the requested shape.
    """
    shape = s.shape
    y = generate_random_numbers(
        pair_production_differential_cross_section_y,
        [s, ],
        shape=shape,
        xmin=ELECTRON_REST_ENERGY**2 / s,
        xmax=torch.ones(shape, device=device),
        logarithmic=logarithmic,
        device=device,
        normalized=True,
        n_integration=max(shape))
    return y


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
        xmin = torch.ones(shape, device=device) * energy_photon_min
    else:
        xmin = torch.ones(shape, device=device) * energy_photon_min
    if energy_photon_max is None:
        energy_photon_max = field[-1, 0]
        xmax = torch.ones(shape, device=device) * energy_photon_max
    else:
        xmax = torch.ones(shape, device=device) * energy_photon_max
    pdf = partial(background_photon_density_interpolated,
                  photon_field_file_path=photon_field_file_path,
                  device=device)
    return (generate_random_numbers(
        pdf,
        pars=[],
        shape=shape,
        xmin=xmin,
        xmax=xmax,
        normalized=True,
        logarithmic=True,
        n_integration=100_000,
        device=device,
        **kwargs
    ))


########################################################################
def generate_random_s(energy,
                      photon_field_file_path,
                      random_cosine=True,
                      device=device,
                      **cosine_kwargs):
    epsilon = generate_random_background_photon_energy(
        energy.shape,
        photon_field_file_path,
        device=device,
        energy_photon_min=S_MIN / (4.0 * energy)
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
def electron_positron_production(energy_tensor,
                                 region_size_cm):
    interaction_rate = 1
    interaction_rate_sampled = generate(interaction_rate)
    if_sterile = (1.0 / interaction_rate_sampled > region_size)
    energy_escaped = energy_tensor[if_sterile]
    energy_tensor = energy_tensor[torch.logical_not(if_sterile)]
    s = generate_s(energy_tensor)
    y_electron = generate_y(s)
    energy_electron = y_electron * s**0.5
    energy_positron = (1.0 - y_electron) * s**0.5
    return {
        "gamma rays escaped": energy_escaped,
        "electrons": energy_electron,
        "positrons": energy_positron
    }


########################################################################
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
########################################################################
# #1 generate tensor with initial energy of primary gamma rays
primary_energy = primary_energy_0 * torch.ones((size,), device=device)
########################################################################
# #2 sample distance traveled by primary gamma rays
print("For {:e} eV interaction rate is {} 1 / cm".format(
    primary_energy_0,
    torch_interpolation(torch.tensor([primary_energy_0, ],
                                     device=device),
                        energy_gamma_node,
                        r_gamma_gamma_node)
)
)
distance_traveled = generate_random_gamma_distance_traveled(
    primary_energy,
    energy_gamma_node,
    r_gamma_gamma_node
)
print("mean: {:.2e} cm".format(torch.mean(distance_traveled)))
########################################################################
# #3 Get escaped and interacted gamma rays
region_size = 0.1 * r_blr
print("region size = {:.2} cm = {:.2} [r_blr]".format(
    region_size, region_size / r_blr
))
escaped = (distance_traveled > region_size)
escaped_energy = primary_energy[escaped]
interacted_energy = primary_energy[torch.logical_not(escaped)]
print("Interacted {:.6f} % of photons".format(
    (1 - len(escaped_energy) / len(primary_energy)) * 100.0)
)
print("Escaped {:.6f} % of photons".format(
    (len(escaped_energy) / len(primary_energy)) * 100.0)
)
########################################################################
# #4 Generate s
s = generate_random_s(interacted_energy,
                      field,
                      random_cosine=True,
                      device=device)
# x = S_MIN / (4.0 * interacted_energy)
# x_max = torch.ones(interacted_energy.shape, device=device) * 10.0  # eV
# aux = auxiliary_pair_production_function(x,
#                                          x_max,
#                                          photon_field_file_path=field,
#                                          device=device)
print("Mean square root of s is {:.3e}".format((
    torch.mean(s)
)**0.5))
print("Max square root of s is {:.3e}".format((
    torch.max(s)
)**0.5))
print("Min square root of s is {:.3e}".format((
    torch.min(s)
)**0.5))
# #5 Choose s >= (2 * ELECTRON_REST_ENERGY)**2
filt_s = (s >= S_MIN)
s = s[filt_s]
print("Fraction of gamma rays generated pairs: {:.6f} %".format(
    len(s) / len(interacted_energy) * 100.0
))
# #6 Compare with theoretical differential cross section
theory_size = 10000
s_theory = torch.logspace(torch.log10(torch.min(s) * 1.1),
                          torch.log10(torch.max(s)),
                          theory_size,
                          device=device)
rho_s = pair_production_differential_cross_section_s(
    s_theory,
    torch.ones(s_theory.shape, device=device) * torch.max(primary_energy),
    photon_field_file_path=field,
    device=device)
print(f'rho_s = {rho_s}')
# sigma_s = sigma_pair_cross_section_reduced(s_theory)
# # ########################################################################
# # # #6 Generate electron energy
# # y = generate_random_y(s,
# #                       logarithmic=True,
# #                       device=device)
# # print("Mean y = {:.2e}".format(torch.mean(y)))
# # electron_energy = y * interacted_energy
# # positron_energy = (1.0 - y) * interacted_energy
# # lepton_energy = torch.cat((electron_energy, positron_energy), dim=-1)
# # ########################################################################
# # # Run some things here
# # # E_gamma = torch.tensor(E_gamma, device=device)
# # # print(E_gamma.device)
# # # x = spec.power_law(E_gamma, 2.0, en_ref=1.0, norm=1.0)
# # # print(x.device)
# # # print(x)
# # # x, p, z = generate_random_numbers(spec.power_law,
# # #                                   [1, 100.0, 10.0],
# # #                                   shape=(1_000_000,),
# # #                                   xmin=torch.tensor(1.,
# # #                                                     device=device),
# # #                                   xmax=torch.tensor(1000.,
# # #                                                     device=device),
# # #                                   normalized=False,
# # #                                   logarithmic=True,
# # #                                   n_integration=1000,
# # #                                   device=device,
# # #                                   verbose=True)
# # # x = x.detach().to('cpu').numpy()
# # # p = p.detach().to('cpu').numpy()
# # # z = z.detach().to('cpu').numpy()
# # #
# # # tens = torch.tensor([x,
# # #                      np.zeros(x.shape),
# # #                      np.ones(x.shape),
# # #                      np.ones(x.shape)],
# # #                     device=device)
# # # print(tens.shape)
# # # ########################################################################
# # # photon_field_file_path = (
# # #     "/home/raylend/Science/agnprocesses/data/PKS1510-089/nph"
# # # )
# # # epsilon = generate_random_background_photon_energy(
# # #     (100_000,),
# # #     photon_field_file_path,
# # #     device=device,
# # #     energy_photon_min=None  # (0.05 * ELECTRON_REST_ENERGY**2 /
# # #     # max(primary_energy))
# # # )
# # # field = np.loadtxt(photon_field_file_path)
# # # # 0.05 * ELECTRON_REST_ENERGY**2 / max(primary_energy)
# # # energy_photon_min = field[0, 0]
# # # print(f"energy_photon_min = {energy_photon_min}")
# # # energy_photon_max = field[-1, 0]
# # # print(f"energy_photon_max = {energy_photon_max}")
# # # pdf = partial(background_photon_density_interpolated,
# # #               photon_field_file_path=photon_field_file_path,
# # #               device=device)
# # # en = torch.logspace(torch.log10(torch.tensor(energy_photon_min,
# # #                                              device=device)),
# # #                     torch.log10(torch.tensor(energy_photon_max,
# # #                                              device=device)),
# # #                     10000,
# # #                     device=device)
# # # density = pdf(en)
# # # norm = torch.trapz(density, en)
# # # mean_photon_energy = torch.trapz(density * en, en) / norm
# # # print("mean photon energy = ", mean_photon_energy)
# # # density = (density.detach().to('cpu').numpy() /
# # #            norm.detach().to('cpu').numpy())
# # # en = en.detach().to('cpu').numpy()
# # # epsilon = epsilon.detach().to('cpu').numpy()
# # ########################################################################
end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded!
elapsed_time_ms = start_event.elapsed_time(end_event)
print(elapsed_time_ms)
fig, ax = plt.subplots(figsize=(8, 6))
x = s_theory.detach().to('cpu').numpy()

h = plt.hist(s.detach().to('cpu').numpy(),
             bins=x[::100],
             density=True,
             log=False,
             label='histogram',
             alpha=0.5)

y = rho_s.detach().to('cpu').numpy()
filt_y = (y > 0)
y = y[filt_y]
x = x[filt_y]
integral = simps(y, x)
y = y / integral
# y = y / np.max(y) * np.max(h[0])
# print('y = ', y)
# y2 = sigma_s.detach().to('cpu').numpy()
# y = y / x
# y = y / np.max(y) * np.max(h[0])
# print('y = ', y)
# y2 = y2 / np.max(y2) * np.max(h[0])
# print('y2 = ', y2)
plt.plot(x,
         y,
         linewidth=3,
         linestyle='-',
         label='theoretical differential distribution')

# plt.plot(x,
#          y2,
#          linewidth=3,
#          linestyle='--',
#          label='theoretical sigma pair production distribution')
#
# F_empirical = np.cumsum(h[0]) * (h[1][1:] - h[1][:-1])
# F_empirical = F_empirical / np.max(F_empirical) * np.max(h[0])
# x_emirical = center_energy(h[1])
#
# plt.plot(x_emirical,
#          F_empirical,
#          linewidth=3,
#          linestyle=':',
#          label='F empirical')


# h = plt.hist(epsilon,
#              density=True,
#              bins=field[0::10, 0],
#              log=True,
#              label='histogram')
# #
# # plt.plot(z, p,
# #          marker=None,
# #          linestyle='-',
# #          linewidth=3,
# #          label='pdf line')
#
# # plt.plot(field[:, 0], field[:, 1],
# #          marker=None,
# #          linestyle='--',
# #          linewidth=2,
# #          label='original',
# #          zorder=10)
# #
# plt.plot(en, density,
#          marker=None,
#          linestyle='-',
#          linewidth=3,
#          label='interpolation')
#
# # plt.xlabel('energy, ' + 'eV', fontsize=18)
# plt.ylabel('SED, ' + r'eV cm$^{-2}$s$^{-1}$', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.xlabel('s, ' + r'eV$^{2}$', fontsize=18)
plt.ylabel(r'$\rho$', fontsize=18)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlim(1.0e+08, 1.0e+12)
# ax.set_ylim(1.0e-01, 3.0e+02)
# ax.grid()
# ax.grid()
# fig.savefig('test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf')
plt.legend()
# plt.legend(loc='upper left')
plt.show()
