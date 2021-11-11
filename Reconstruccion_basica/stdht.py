#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:00:51 2021

@author: gmarzik
"""
import librosa
import numpy as np
import scipy
from FHT import FHT_FFT

def sdht(signal, frame_length, frame_step, window="hamming"):
    """Compute Short-Time Discrete Hartley Transform of `signal`.

    No padding is applied to the signals.

    Parameters
    ----------
    signal : Time-domain input signal of shape `(n_samples,)`.

    frame_length : Window length and DHT frame length in samples.

    frame_step : Number of samples between adjacent DHT columns.

    window : Window specification passed to ``librosa.filters.get_window``.
        Default: "hamming".  Window to use for DHT.

    Returns
    -------
    dht : Real-valued F-T domain DHT matrix of shape `(frame_length, n_frames)`
    """
    framed = librosa.util.frame(signal, frame_length, frame_step)
    if window is not None:
        window = librosa.filters.get_window(window, frame_length, fftbins=True).astype(
            signal.dtype
        )
        framed = framed * window[:, np.newaxis]
    framed_dht = np.zeros(framed.shape)    
    for i in range(framed.shape[1]):
        framed_dht[:,i] = FHT_FFT(framed[:,i])    
    return framed_dht


def isdht(dht, *, frame_step, frame_length=None, window="hamming"):
    """Compute Inverse Short-Time Discrete Hartley Transform of `dht`.

    Parameters other than `dht` are keyword-only.

    Parameters
    ----------
    dht : DHT matrix from `sdht`.

    frame_step : Number of samples between adjacent DHT columns (should be the
        same value that was passed to `sdht`).

    frame_length : Ignored. Window length and DHT frame length in samples.
        Can be None (default) or same value as passed to `sdht`.

    window : Window specification passed to ``librosa.filters.get_window``.
        Default: "hamming".  Window to use for IDHT.

    Returns
    -------
    signal : Time-domain signal reconstructed from `dht` of shape `(n_samples,)`.
        Note that `n_samples` may be different from the original signals' lengths as passed to `sdht_torch`,
        because no padding is applied.
    """
    frame_length2, n_frames = dht.shape
    assert frame_length in {None, frame_length2}
    signal_t = np.zeros(dht.shape)
    for i in range(dht.shape[1]):
        signal_t[:,i] = (1/dht.shape[0])*FHT_FFT(dht[:,i])
    signal = overlap_add(
        signal_t, frame_step=frame_step
    )
    if window is not None:
        window = librosa.filters.get_window(window, frame_length2, fftbins=True).astype(
            dht.dtype
        )
        window_frames = np.tile(window[:, np.newaxis], (1, n_frames))
        window_signal = overlap_add(window_frames, frame_step=frame_step)
        signal = signal / window_signal
    return signal


def overlap_add(framed, *, frame_step, frame_length=None):
    """Overlap-add ("deframe") a framed signal.

    Parameters other than `framed` are keyword-only.

    Parameters
    ----------
    framed : array_like of shape `(..., frame_length, n_frames)`.

    frame_step : Overlap to use when adding frames.

    Returns
    -------
    deframed : Overlap-add ("deframed") signal.
        np.ndarray of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
    """
    *shape_rest, frame_length2, n_frames = framed.shape
    assert frame_length in {None, frame_length2}
    deframed_size = (n_frames - 1) * frame_step + frame_length2
    deframed = np.zeros((*shape_rest, deframed_size), dtype=framed.dtype)
    for i in range(n_frames):
        pos = i * frame_step
        deframed[..., pos : pos + frame_length2] += framed[..., i]
    return deframed
