"""
Took 4 days, 7:11:50.918525
"""
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy import interpolate
from scipy.integrate import simps
import agnprocesses.spectra as spec
import agnprocesses.ic as ic
import agnprocesses.gamma_gamma as gamma_gamma
import emcee
import test_pytorch

device = 'cuda'
field = "data/PKS1510-089/nph"
shape = (100_000,)

field_numpy = np.loadtxt(field)
norm = simps(field_numpy[:, 1], field_numpy[:, 0])

f_aux_nph_for_numpy = interpolate.interp1d(
    np.log10(field_numpy[:, 0]), np.log10(field_numpy[:, 1]),
    kind='linear',
    copy=True,
    bounds_error=False,
    fill_value=1.0e-40
)


def nph_for_numpy_interpolated(e):
    return(10.0**f_aux_nph_for_numpy(np.log10(e)))


# 2. Current background photon energy sampling
t_start_numpy = test_pytorch.get_current_date_time()
print("Torch sampling started at: ",
      test_pytorch.date_time_string_human_friendly())


def calculate_windowed_max_min_ratio(y_input, x_input, std_window_width=9):
    try:
        y = np.copy(y_input).reshape(np.max(y_input.shape))
        x = np.copy(x_input).reshape(np.max(x_input.shape))
    except:
        raise ValueError(
            "y and x must have a shape, which can be reshaped into one-dimensional shape"
        )
    max_min_ratio = np.zeros((y.shape[0] - std_window_width, ))
    for i in range(0, max_min_ratio.shape[0]):
        sh = int(std_window_width / 2)
        y_ind = i + sh
        if std_window_width % 2 == 1:
            # max_min_ratio[i] = np.std(y[y_ind - sh:y_ind + sh + 1] /
            #                  np.mean(y[y_ind - sh:y_ind + sh + 1]))
            max_min_ratio[i] = (np.max(y[y_ind - sh:y_ind + sh + 1]) /
                                np.min(y[y_ind - sh:y_ind + sh + 1]))
        else:
            # max_min_ratio[i] = np.std(y[y_ind - sh:y_ind + sh] /
            #                  np.mean(y[y_ind - sh:y_ind + sh]))
            max_min_ratio[i] = (np.max(y[y_ind - sh:y_ind + sh]) /
                                np.min(y[y_ind - sh:y_ind + sh]))
    return max_min_ratio


def separate_continuum_and_spikes(y_input, x_input,
                                  std_window_width=9,
                                  filt_value=0.5):
    try:
        y = np.copy(y_input).reshape(np.max(y_input.shape))
        x = np.copy(x_input).reshape(np.max(x_input.shape))
        sh = int(std_window_width / 2)
    except:
        raise ValueError(
            "y and x must have a shape, which can be reshaped into one-dimensional shape"
        )
    max_min_ratio = calculate_windowed_max_min_ratio(
        y, x, std_window_width=std_window_width
    )
    if std_window_width % 2 == 1:
        x = x[sh + 1: -sh][:-1]
        y = y[sh + 1: -sh][:-1]
    else:
        x = x[sh:-sh][:-1]
        y = y[sh:-sh][:-1]
    filt_spike = (max_min_ratio > filt_value)
    difference = np.diff(filt_spike)
    x_starters = x[difference == True][0::2]
    x_enders = x[difference == True][1::2]
    x_continuum = x[difference == False]
    y_continuum = y[difference == False]
    if len(x_starters) > len(x_enders):
        x_starters = x_starters[:-1]
    if len(x_enders) > len(x_starters):
        x_enders = x_enders[:-1]
    x_spikes = np.zeros(x_starters.shape)
    y_spikes = np.zeros(x_starters.shape)
    for i in range(0, len(x_starters)):
        x_spikes[i] = float(x[int((np.argwhere(x_starters[i] == x) +
                                   np.argwhere(x_enders[i] == x)) / 2)])
        y_spikes[i] = np.max(
            y[int(np.argwhere(x_starters[i] == x)):
              int(np.argwhere(x_enders[i] == x))]
        )
    return (x_spikes, y_spikes)


std_window_width = 5
max_min_ratio = calculate_windowed_max_min_ratio(
    field_numpy[:, 1],
    field_numpy[:, 0],
    std_window_width=std_window_width
)

x_spikes, y_spikes = separate_continuum_and_spikes(
    field_numpy[:, 1],
    field_numpy[:, 0],
    std_window_width=std_window_width,
    filt_value=4.0
)

print("Number of detected high lines = ", len(x_spikes))

t_end_numpy = test_pytorch.get_current_date_time()
print("Numpy sampling ended at: ",
      test_pytorch.date_time_string_human_friendly())
print("Numpy sampling took ", t_end_numpy - t_start_numpy)
###############################################################################
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(
    field_numpy[:, 0], field_numpy[:, 1] / np.max(field_numpy),
    marker=None,
    linestyle='-',
    linewidth=1,
    color='k',
    label='field from the file'
)
plt.plot(
    field_numpy[int(std_window_width / 2) + 1:-int(std_window_width / 2), 0],
    max_min_ratio,
    linestyle='-',
    linewidth=1,
    color='r',
    label='max_min_ratio'
)

for i in range(0, len(x_spikes)):
    xx = np.ones((50, )) * x_spikes[i]
    yy = np.logspace(-9, np.log10(y_spikes[i]), 50) / np.max(field_numpy)
    plt.plot(xx, yy, linestyle="--", color="orange", linewidth=1)

# for i in range(0, len(x_starters)):
#     xx = np.ones((50, )) * x_starters[i]
#     yy = np.logspace(-9, 0, 50)
#     plt.plot(xx, yy, linestyle="-", color="green", linewidth=1)
#
# for i in range(0, len(x_enders)):
#     xx = np.ones((50, )) * x_enders[i]
#     yy = np.logspace(-9, 0, 50)
#     plt.plot(xx, yy, linestyle="-", color="orange", linewidth=1)

# plt.xlabel('energy, ', fontsize=18)
# plt.xticks(fontsize=12)
# plt.ylabel('PDF', fontsize=18)
# plt.yticks(fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlim(1e+02, 1e+03)
# ax.set_ylim(1e-10, 1e-06)
plt.legend()  # loc='upper left')
# plt.savefig("numpy2.pdf")
plt.show()
