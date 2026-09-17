"""Angular-spectrum free-space propagation.

Vendored from pyDHM (https://github.com/catrujilla/pyDHM), file
``pyDHM/numericalPropagation.py``, MIT License,
Copyright (c) 2022 Ana Doblas and Carlos Alejandro Trujillo (EAFIT).

Only ``angularSpectrum`` was used by this project, so it is reproduced here
verbatim (``from math import pi`` replaced by ``np.pi``) to drop the pyDHM
dependency entirely.
"""

import numpy as np

__all__ = ["angularSpectrum"]


def angularSpectrum(field, z, wavelength, dx, dy):
    """Diffract a complex field using the angular spectrum approximation.

    Parameters
    ----------
    field : array_like
        Complex field to propagate.
    z : float
        Propagation distance, in the same length units as ``wavelength``,
        ``dx`` and ``dy``. ``z = 0`` returns the input field unchanged.
    wavelength : float
        Wavelength.
    dx, dy : float
        Sampling pitches along x and y.

    Returns
    -------
    numpy.ndarray
        The propagated complex field, same shape as ``field``.

    Notes
    -----
    Spatial frequencies beyond the evanescent cutoff
    (``fx**2 + fy**2 > 1/wavelength**2``) make the square root imaginary; NumPy
    returns NaN there for real input, which is the original pyDHM behaviour and
    is preserved here.
    """
    field = np.array(field)
    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * N)
    dfy = 1 / (dy * M)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    phase = np.exp(1j * z * 2 * np.pi * np.sqrt(
        np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))
    ))

    tmp = field_spec * phase

    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out
