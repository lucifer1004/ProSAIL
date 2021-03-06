import prosail as ps
import numpy as np
import SALib
from SALib.sample import saltelli
from SALib.util import read_param_file
import multiprocessing


# Params: N, cab, caw, car, cbrown, cm, lai, lidfa, psoil, rsoil, hspot, tts, tto, psi, ant
def generate_spectrum(p):
    return ps.run_prosail(n=p[0],
                          cab=p[1],
                          cw=p[2],
                          car=p[3],
                          cbrown=p[4],
                          cm=p[5],
                          lai=p[6],
                          lidfa=p[7],
                          psoil=p[8],
                          rsoil=p[9],
                          hspot=p[10],
                          tts=p[11],
                          tto=p[12],
                          psi=p[13],
                          ant=p[14],
                          prospect_version="D")


def generate_spectra(sample_number=10000, bounds='../assets/prosail_param_bounds.txt', save_to_npy=False,
                     spectra_save='../data/spectra.npy', params_save='../data/params.npy',
                     params_norm_save='../data/params_norm.npy'):
    param_dimension = 15
    wavelength_start = 400
    wavelength_end = 2500
    wavelength_num = wavelength_end - wavelength_start + 1

    problem = read_param_file(bounds)
    params = saltelli.sample(problem, sample_number)
    params_norm = params.copy()

    num = sample_number * (param_dimension + 1) * 2
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    spec = pool.map(generate_spectrum, params)
    spec = np.array(spec).reshape(num, 2101)

    for i in range(num):
        p = params[i]
        for j in range(param_dimension):
            params_norm[i][j] = (p[j] - problem['bounds'][j][0]) / (problem['bounds'][j][1] - problem['bounds'][j][0])

    if save_to_npy:
        np.save(spectra_save, spec)
        np.save(params_save, params)

    return spec, params, params_norm
