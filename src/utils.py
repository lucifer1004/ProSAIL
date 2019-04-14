import numpy as np
import multiprocessing


def generate_noise(index):
    return np.random.normal(loc=0.0, scale=0.1 * np.random.rand(), size=2101)


def generate_noise_batch(num=320000):
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    noise = pool.map(generate_noise, range(num))
    noise = np.array(noise).reshape(num, 2101)
    return noise


def normalize_spectrum(spec):
    res = (spec - wl_min) / (wl_max - wl_min)
    res[res < 0] = 0
    res[res > 1] = 1
    return res


def denormalize_spectrum(spec):
    return spec * (wl_max - wl_min) + wl_min


# def generate_normalized_noised_spectra():
#     noised_spectra = train + noise
#     noised_spectra_norm = noised_spectra.copy()
#     for i in range(2101):
#         noised_spectra_norm[:, i] = (noised_spectra_norm[:, i] - wl_min[i]) / (wl_max[i] - wl_mi[i)
#     noised_spectra_norm[noised_spectra_norm < 0] = 0
#     noised_spectra_norm[noised_spectra_norm > 1] = 1
#     return noised_spectra_norm

if __name__ == '__init__':
    train = np.load('../data/spectra.npy')
    wl_max = np.zeros(2101)
    wl_min = np.zeros(2101)
    for i in range(2101):
        wl_max[i] = np.max(train[:, i])
        wl_min[i] = np.min(train[:, i])
