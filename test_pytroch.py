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

if __name__ == '__main__':
    ELECTRON_REST_ENERGY = const.m_e.to(u.eV, u.mass_energy()).value
    print("ELECTRON_REST_ENERGY = {:.6e}".format(ELECTRON_REST_ENERGY))
    print('Hey!')
    print("If cuda is available: ",
          torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ########################################################################
    def torch_interpolation(x_new_tensor: torch.tensor,
                            x_old_tensor: torch.tensor,
                            y_old_tensor: torch.tensor):
        f_support = torchinterp1d.Interp1d()
        return(
            f_support(x_old_tensor,
                      y_old_tensor,
                      x_new_tensor)
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
    #
    # def gamma_gamma_interaction_rate_interpolated(e_gamma_new_tensor,
    #                                               e_gamma_old_tensor,
    #                                               r_gamma_old_tensor):
    #     f_support = torchinterp1d.Interp1d()
    #     return(
    #         f_support(e_gamma_old_tensor,
    #                   r_gamma_old_tensor,
    #                   e_gamma_new_tensor)
    #     )
    #
    ########################################################################

    def generate_random_numbers(pdf,
                                pars=None,
                                shape=(1,),
                                xmin=torch.tensor(1.0,
                                                  device=device),
                                xmax=torch.tensor(10.0,
                                                  device=device),
                                normalized=False,
                                logarithmic=True,
                                n_integration=1000,
                                device=None,
                                **kwargs):
        with torch.no_grad():
            if logarithmic:
                z = torch.logspace(torch.log10(xmin),
                                   torch.log10(xmax),
                                   n_integration,
                                   device=device)
            else:
                z = torch.linspace(xmin,
                                   xmax,
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
            if logarithmic:
                # c1 = torch.log(torch.tensor(xmax / xmin, device=device))
                # c2 = xmax - xmin
                c = max(y_theory * z)  # * c1 / c2)
            else:
                c = max(y_theory)
            while generate_size > 0:
                if logarithmic:
                    x = 10.0**((torch.log10(xmax) -
                                torch.log10(xmin)) *
                               torch.rand((generate_size,),
                                          device=device) +
                               torch.log10(xmin))
                    p = (pdf(x, *pars, **kwargs) / norm * x)  # *
                    # c1 / c2)
                else:
                    x = (xmax - xmin) * torch.rand(generate_size,
                                                   device=device) + xmin
                    p = pdf(x, *pars, **kwargs) / norm
                y_new = torch.rand((generate_size,),
                                   device=device) * c
                filt = (y_new < p)
                x_new = x[filt]
                generated = len(x_new)
                start_point = len(x_final[x_final != float('Inf')])
                generate_size = max(shape) - generated - start_point
                x_final[start_point:start_point + generated] = x_new

        return (x_final, y_theory, z)

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

    def generate_random_gamma_path_length(e_gamma: torch.tensor,
                                          e_gamma_node: torch.tensor,
                                          r_gamma_node: torch.tensor,
                                          device=device):
        """
        Generates tensor of random gamma-ray path lengths at e_gamma energy
        points given that gamma-gamma interaction rate r_gamma_node
        computed at e_gamma_node energy points.
        """
        r_gamma = torch_interpolation(
            e_gamma,
            e_gamma_node,
            r_gamma_node
        )
        uniform_random = torch.rand(e_gamma.shape, device=device)
        return (-torch.log(unifrom_random) / r_gamma)

    ########################################################################
    def generate_random_cosine(shape=(1,),
                               cosmin=torch.tensor(-1.0,
                                                   device=device),
                               cosmax=torch.tensor(1.0,
                                                   device=device),
                               device=device):
        return(
            (cosmax - cosmin) * torch.rand(max(shape),
                                           device=device) + cosmin
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
        beta = (1 - 4 * ELECTRON_REST_ENERGY**2 / s)**0.5
        t11 = y**2 / (1 - y) + 1 - y
        t12 = (1 - beta**2) / (1 - y)
        t13 = (-(1 - beta**2)**2) / (4 * y * (1 - y)**2)
        t1 = t11 + t12 + t13
        t2 = 1 + (2 * beta**2) * (1 - beta**2)
        return (t1 / (y * t2))

    #######################################################################
    def generate_random_y(s,
                          shape=(1,),
                          ymin=torch.tensor(1.0e-06,
                                            device=device),
                          ymax=torch.tensor(1.0,
                                            device=device),
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
        y = torch.zeros(shape,
                        device=device)
        # bottle neck?
        for i, value in enumerate(s):
            y[i] = generate_random_numbers(
                pair_production_differential_cross_section_y,
                [s[i], ],
                shape=(1,),
                xmin=ymin,
                xmax=ymax,
                logarithmic=logarithmic,
                device=device,
                **kwargs)
        return y

    #######################################################################
    def generate_random_background_photon_energy(
            shape,
            photon_field_file_path,
            device=device
    ):
        field = np.loadtxt(photon_field_file_path)
        energy_photon_min = field[0, 0]
        energy_photon_max = field[-1, 0]
        pdf = partial(background_photon_density_interpolated,
                      photon_field_file_path=photon_field_file_path,
                      device=device)
        return (generate_random_numbers(
            pdf,
            pars=None,
            shape=shape,
            xmin=torch.tensor(energy_photon_min,
                              device=device),
            xmax=torch.tensor(energy_photon_max,
                              device=device),
            normalized=False,
            logarithmic=True,
            n_integration=10000,
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
            device=device
        )
        if random_cosine:
            cosine = generate_random_cosine(energy.shape,
                                            device=device,
                                            **cosine_kwargs)
        else:
            cosine = -torch.ones(energy.shape,
                                 device=device)
        return (2.0 * epsilon * (1.0 - cosine) * energy)

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
    # field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/nph"
    # E_gamma, r_gamma_gamma = gamma_gamma.interaction_rate(field,
    #                                                       1.0e+08 * u.eV,
    #                                                       5.0e+12 * u.eV,
    #                                                       background_photon_energy_unit=u.eV,
    #                                                       background_photon_density_unit=(u.eV * u.cm**3)**(-1))
    print(torch.cuda.get_device_name())
    ########################################################################
    en = torch.logspace(0, 3, 1000, device=device)
    spec_points = spec.power_law(en, 1.0, norm=100.0, en_ref=10.0)
    f_support = torchinterp1d.Interp1d()
    print(f_support(en, spec_points, en))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # start = time.monotonic()

    # Run some things here
    # E_gamma = torch.tensor(E_gamma, device=device)
    # print(E_gamma.device)
    # x = spec.power_law(E_gamma, 2.0, en_ref=1.0, norm=1.0)
    # print(x.device)
    # print(x)
    x, p, z = generate_random_numbers(spec.power_law,
                                      [1, 100.0, 10.0],
                                      shape=(1_000_000,),
                                      xmin=torch.tensor(1.,
                                                        device=device),
                                      xmax=torch.tensor(1000.,
                                                        device=device),
                                      normalized=False,
                                      logarithmic=True,
                                      n_integration=1000,
                                      device=device)
    # y = float('Inf') * torch.ones((10,),
    #                               device=device)
    # print(len(y[y != float('Inf')]))
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    # finish = time.monotonic()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    # elapsed_time_ms = timedelta(seconds=finish - start)
    print(elapsed_time_ms)
    x = x.detach().to('cpu').numpy()
    p = p.detach().to('cpu').numpy()
    z = z.detach().to('cpu').numpy()

    tens = torch.tensor([x,
                         np.zeros(x.shape),
                         np.ones(x.shape),
                         np.ones(x.shape)],
                        device=device)
    print(tens.shape)
    ########################################################################
    fig, ax = plt.subplots(figsize=(8, 6))

    h = plt.hist(x,
                 density=True,
                 bins=z[0::10],
                 log=True,
                 label='histogram')

    plt.plot(z, p,
             marker=None,
             linestyle='-',
             linewidth=3,
             label='pdf line')

    # plt.xlabel('energy, ' + 'eV', fontsize=18)
    # plt.xticks(fontsize=12)
    # plt.ylabel('SED, ' + r'eV cm$^{-2}$s$^{-1}$', fontsize=18)
    # plt.yticks(fontsize=12)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(1.0e+08, 1.0e+12)
    # ax.set_ylim(1.0e-01, 3.0e+02)
    # ax.grid()
    # ax.grid()
    # fig.savefig('test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf')
    plt.legend()
    # plt.legend(loc='upper left')
    plt.show()
