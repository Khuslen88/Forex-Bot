"""
denoising.py — Signal-processing-based denoising for forex OHLC price series.

This is the NEW module added on top of the capstone forex bot. It implements two
denoising methods drawn from the literature reviewed in Assignment 6:

  1. Wavelet denoising    (Zhao & Khushi, 2021 — arXiv:2102.04861)
  2. Empirical Mode Decomposition (EMD) denoising (Jin, Jin & Chen, 2022 — PeerJ CS)

The denoiser is applied to OHLC columns BEFORE technical indicators are computed,
so all downstream features (RSI, MACD, Bollinger, ATR, SMA, …) see a cleaner signal.

Course topic connection: Signal Processing Basics (Topic 5).
- Wavelets are derived from Fourier analysis and act as multi-resolution band-pass filters.
- EMD adaptively decomposes a non-stationary signal into intrinsic mode functions (IMFs).
- Standard moving averages used in capstone are FIR low-pass filters; wavelet/EMD
  preserve sharp edges (transitions) better than fixed-window MAs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Optional imports — installed via requirements.txt
try:
    import pywt  # PyWavelets
except ImportError:  # pragma: no cover
    pywt = None

try:
    from PyEMD import EMD  # EMD-signal package
except ImportError:  # pragma: no cover
    EMD = None


OHLC_COLS = ("Open", "High", "Low", "Close")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Wavelet denoising
# ─────────────────────────────────────────────────────────────────────────────
def wavelet_denoise_series(
    x: np.ndarray,
    wavelet: str = "sym15",
    level: int = 2,
    mode: str = "soft",
) -> np.ndarray:
    """
    Denoise a 1-D signal with discrete wavelet transform + universal soft thresholding.

    Following Zhao & Khushi (2021) we default to the sym15 wavelet and zero out the
    high-frequency detail coefficients of the top `level` decomposition levels —
    this removes microstructure noise while preserving trend and large moves.

    Parameters
    ----------
    x : 1-D numpy array of length N (e.g. close prices)
    wavelet : pywt wavelet name (sym15, db4, haar, …)
    level : number of decomposition levels whose detail coefficients are thresholded
    mode : 'soft' (recommended) or 'hard' thresholding

    Returns
    -------
    Denoised array of length N.
    """
    if pywt is None:
        raise ImportError("PyWavelets not installed. `pip install PyWavelets`")

    x = np.array(x, dtype=float, copy=True)  # writable copy — PyWavelets requires it
    if x.ndim != 1:
        raise ValueError("wavelet_denoise_series expects 1-D input")

    # Decompose
    coeffs = pywt.wavedec(x, wavelet, mode="periodization", level=level)
    # cA, cD_level, cD_level-1, …, cD_1
    # Universal threshold = sigma * sqrt(2 ln N), sigma estimated from finest detail
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(x)))

    # Threshold all detail coeffs (skip cA — keep approximation)
    new_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]
    ]

    denoised = pywt.waverec(new_coeffs, wavelet, mode="periodization")
    # waverec can return length N or N+1 depending on parity; trim
    return denoised[: len(x)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMD denoising
# ─────────────────────────────────────────────────────────────────────────────
def emd_denoise_series(
    x: np.ndarray,
    n_imfs_to_remove: int = 1,
    max_imf: int = 8,
) -> np.ndarray:
    """
    Denoise a 1-D signal by removing the highest-frequency IMFs from its EMD.

    Following Jin et al. (2022): decompose the price series into intrinsic mode
    functions; the first IMF carries the highest-frequency content (typically
    market microstructure noise), so subtracting the first 1–2 IMFs yields a
    smoother signal that retains medium- and low-frequency structure.

    Parameters
    ----------
    x : 1-D numpy array
    n_imfs_to_remove : how many high-frequency IMFs to discard (1 or 2 is typical)
    max_imf : safety cap on EMD decomposition

    Returns
    -------
    Denoised array of same length as x.
    """
    if EMD is None:
        raise ImportError("EMD-signal not installed. `pip install EMD-signal`")

    x = np.array(x, dtype=float, copy=True)  # writable copy
    if x.ndim != 1:
        raise ValueError("emd_denoise_series expects 1-D input")

    emd = EMD()
    imfs = emd(x, max_imf=max_imf)  # shape (n_imfs, N)
    if imfs.shape[0] <= n_imfs_to_remove:
        # Not enough IMFs — return original
        return x.copy()

    # Reconstruct using IMFs from index n_imfs_to_remove onward + residue
    return imfs[n_imfs_to_remove:].sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-level helper
# ─────────────────────────────────────────────────────────────────────────────
def denoise_ohlc(
    df: pd.DataFrame,
    method: str = "wavelet",
    columns=OHLC_COLS,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply denoising to OHLC columns of a DataFrame, returning a new DataFrame.

    Parameters
    ----------
    df : input DataFrame with at least the OHLC columns
    method : 'wavelet', 'emd', or 'none' (passthrough)
    columns : iterable of column names to denoise
    **kwargs : forwarded to the per-series denoiser

    Returns
    -------
    Copy of df with denoised columns.
    """
    if method == "none":
        return df.copy()

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        if method == "wavelet":
            out[col] = wavelet_denoise_series(out[col].to_numpy(), **kwargs)
        elif method == "emd":
            out[col] = emd_denoise_series(out[col].to_numpy(), **kwargs)
        else:
            raise ValueError(f"Unknown denoising method: {method!r}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Causal (look-ahead-bias-free) rolling-window denoiser — for live/online use
# ─────────────────────────────────────────────────────────────────────────────
def denoise_ohlc_causal(
    df: pd.DataFrame,
    method: str = "wavelet",
    window: int = 256,
    refresh_every: int = 1,
    min_warmup: int = 32,
    columns=OHLC_COLS,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply denoising in a CAUSAL rolling window so that the denoised value at
    time t depends only on prices observed at times <= t.  This is the version
    that mirrors how the denoiser would run in live trading, and is what
    Assignment 7 specifies as the primary preprocessing pipeline.

    Parameters
    ----------
    df : input DataFrame (OHLC columns required)
    method : 'wavelet' (recommended for 1H) or 'emd' (recommended on daily or
             with refresh_every >= 24 on 1H, since EMD is ~50x slower)
    window : rolling-window length in bars.  256 ≈ 10 trading days at 1H.
    refresh_every : how often to RE-RUN the denoiser.  1 = every bar (exact).
                    24 = once per day on 1H data (much faster, tiny info-leak
                    only at chunk boundaries).
    min_warmup : do not denoise until at least this many bars are available;
                 return raw price instead.
    columns : OHLC columns to denoise.
    **kwargs : forwarded to the per-series denoiser.

    Returns
    -------
    Copy of df with denoised OHLC columns.
    """
    if method == "none":
        return df.copy()

    out = df.copy()
    n = len(df)

    for col in columns:
        if col not in out.columns:
            continue
        raw = df[col].to_numpy()
        denoised = raw.copy()

        # Cached denoised chunk between refreshes
        cached_chunk = None
        cached_start = -1

        for t in range(n):
            if t < min_warmup:
                # Warmup period: pass raw value through
                continue

            # Decide whether to re-run the denoiser at this bar
            need_refresh = (cached_chunk is None) or ((t - cached_start) % refresh_every == 0)

            if need_refresh:
                lo = max(0, t - window + 1)
                chunk = raw[lo:t + 1]
                if method == "wavelet":
                    cached_chunk = wavelet_denoise_series(chunk, **kwargs)
                elif method == "emd":
                    cached_chunk = emd_denoise_series(chunk, **kwargs)
                else:
                    raise ValueError(f"Unknown denoising method: {method!r}")
                cached_start = t

            # Index into the cached denoised chunk for the current bar
            chunk_idx = t - max(0, cached_start - len(cached_chunk) + 1)
            chunk_idx = min(chunk_idx, len(cached_chunk) - 1)
            denoised[t] = cached_chunk[chunk_idx]

        out[col] = denoised

    return out


__all__ = [
    "wavelet_denoise_series",
    "emd_denoise_series",
    "denoise_ohlc",
    "denoise_ohlc_causal",
]
