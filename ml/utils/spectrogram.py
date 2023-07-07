# mypy: disable-error-code="import"
"""Defines spectrogram functions.

This file contains utilities for converting waveforms to MFCCs and back. This
can be a more useful representation to use for training models than raw
waveforms, since it's easier for models to learn patterns in the MFCCs than
in the waveforms.
"""

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as A
from torch import Tensor, nn
from torchaudio.transforms import GriffinLim, InverseSpectrogram, Spectrogram

from ml.utils.amp import autocast_tensors
from ml.utils.logging import configure_logging
from ml.utils.numpy import as_numpy_array

logger = logging.getLogger(__name__)

Array = Tensor | np.ndarray

try:
    import pyworld
except ImportError:
    pyworld = None


class _Normalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Normalization parameters.
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))
        self.register_buffer("count", torch.zeros(1, dtype=torch.int64))

    mean: Tensor
    std: Tensor
    count: Tensor

    def normalize(self, x: Tensor) -> Tensor:
        mfcc_mean, mfcc_std = x.mean(), x.std()
        next_count = self.count + 1
        alpha, beta = self.count / next_count, 1 / next_count
        new_mean = self.mean * alpha + mfcc_mean * beta
        new_std = self.std * alpha + mfcc_std * beta
        self.mean.copy_(new_mean, non_blocking=True)
        self.std.copy_(new_std, non_blocking=True)
        self.count.copy_(next_count, non_blocking=True)
        x = (x - new_mean) / new_std
        return x

    def denormalize(self, mfcc: Tensor) -> Tensor:
        return mfcc * self.std + self.mean


class AudioMfccConverter(_Normalizer):
    """Defines a module for converting waveforms to MFCCs and back.

    This module returns the normalized MFCCs from the waveforms. It uses
    the pseudoinverse of the mel filterbanks and the DCT matrix to convert
    MFCCs back to spectrograms, and then uses the Griffin-Lim algorithm to
    convert spectrograms back to waveforms. The pseudoinverse is used because
    it's faster than doing gradient decent every time we want to generate a
    spectrogram.

    Parameters:
        sample_rate: Sample rate of the audio.
        n_mfcc: Number of MFCC bands.
        n_mels: Number of Mel bands.
        n_fft: Number of FFT bands.
        hop_length: Hop length for the STFT.
        win_length: Window length for the STFT.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int | None = None,
        win_length: int | None = None,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft

        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2, normalized=False)
        self.griffin_lim = GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2)

        mel_fb = A.melscale_fbanks(n_fft // 2 + 1, 0.0, sample_rate // 2, n_mels, sample_rate)
        self.register_buffer("mel_fb", mel_fb)
        self.register_buffer("inv_mel_fb", torch.linalg.pinv(mel_fb))

        dct_mat = A.create_dct(n_mfcc, n_mels, "ortho")
        self.register_buffer("dct_mat", dct_mat)
        self.register_buffer("inv_dct_mat", torch.linalg.pinv(dct_mat))

    mel_fb: Tensor
    inv_mel_fb: Tensor
    dct_mat: Tensor
    inv_dct_mat: Tensor

    def audio_to_spec(self, waveform: Tensor) -> Tensor:
        """Converts a waveform to MFCCs.

        Args:
            waveform: Tensor of shape ``(..., num_samples)``.

        Returns:
            Tensor of shape ``(..., n_mfcc, num_frames)``.
        """
        with autocast_tensors(waveform, enabled=False) as waveform:
            spec = self.spec(waveform)
            mel_spec = torch.einsum("...ct,cf->...ft", spec, self.mel_fb)
            log_mel_spec = torch.log(mel_spec + 1e-6)
            mfcc = torch.einsum("...ct,cf->...ft", log_mel_spec, self.dct_mat)
            mfcc = self.normalize(mfcc)
            return mfcc

    def spec_to_audio(self, mfcc: Tensor) -> Tensor:
        """Converts MFCCs to a waveform.

        Args:
            mfcc: Tensor of shape ``(..., n_mfcc, num_frames)``.

        Returns:
            Tensor of shape ``(..., num_samples)``.
        """
        with autocast_tensors(mfcc.detach(), enabled=False) as mfcc:
            mfcc = self.denormalize(mfcc)
            log_mel_spec = torch.einsum("...ft,fc->...ct", mfcc, self.inv_dct_mat)
            mel_spec = torch.exp(log_mel_spec) - 1e-6
            spec = torch.einsum("...ft,fc->...ct", mel_spec, self.inv_mel_fb).clamp_min_(1e-8)
            waveform = self.griffin_lim(spec)
            return waveform


class AudioStftConverter(_Normalizer):
    """Defines a class for converting waveforms to spectrograms and back.

    This is an exact forward and backward transformation, meaning that the
    input can be reconstructed perfectly from the output. However, oftentimes
    the phase information is not easy to deal with for downstream networks.

    Parameters:
        n_fft: Number of FFT bands.
        hop_length: Hop length for the STFT.
        win_length: Window length for the STFT.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int | None = None,
        win_length: int | None = None,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 4

        self.stft = Spectrogram(self.n_fft, self.win_length, self.hop_length, power=None, normalized=True)
        self.istft = InverseSpectrogram(self.n_fft, self.win_length, self.hop_length, normalized=True)

    def normalize(self, mag: Tensor) -> Tensor:
        log_mag = torch.log(mag + 1e-6)
        return super().normalize(log_mag)

    def denormalize(self, log_mag: Tensor) -> Tensor:
        log_mag = super().denormalize(log_mag)
        return torch.exp(log_mag) - 1e-6

    def audio_to_spec(self, waveform: Tensor) -> Tensor:
        """Converts a waveform to a spectrogram.

        This version keeps the phase information, in a parallel channel with
        the magnitude information.

        Args:
            waveform: Tensor of shape ``(..., num_samples)``.

        Returns:
            Tensor of shape ``(..., 2, num_frames, n_fft // 2 + 1)``.
            The first channel is the magnitude, the second is the phase.
        """
        with autocast_tensors(waveform, enabled=False) as waveform:
            spec = self.stft(waveform.detach())
            mag = self.normalize(spec.abs())
            phase = spec.angle()
            return torch.stack((mag, phase), -3)

    def spec_to_audio(self, spec: Tensor) -> Tensor:
        """Converts a spectrogram to a waveform.

        This version expects the spectrogram to have two channels, one for
        magnitude and one for phase.

        Args:
            spec: Tensor of shape ``(..., 2, num_frames, n_fft // 2 + 1)``.

        Returns:
            Tensor of shape ``(..., num_samples)``, the reconstructed waveform.
        """
        with autocast_tensors(spec, enabled=False) as spec:
            mag, phase = spec.detach().unbind(-3)
            mag = self.denormalize(mag)
            real, imag = mag * phase.cos(), mag * phase.sin()
            spec = torch.complex(real, imag)
            waveform = self.istft(spec)
            return waveform


class AudioMagStftConverter(_Normalizer):
    def __init__(
        self,
        n_fft: int = 1024,
        n_iter: int = 32,
        hop_length: int | None = None,
        win_length: int | None = None,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.n_iter = n_iter
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 4

        self.stft = Spectrogram(self.n_fft, self.win_length, self.hop_length, power=2, normalized=False)
        self.griffin_lim = GriffinLim(self.n_fft, n_iter, self.win_length, self.hop_length, power=2)

    def audio_to_mag_spec(self, waveform: Tensor) -> Tensor:
        """Converts a waveform to a magnitude spectrogram.

        Args:
            waveform: Tensor of shape ``(..., num_samples)``.

        Returns:
            Tensor of shape ``(..., num_frames, n_fft // 2 + 1)``.
        """
        with autocast_tensors(waveform, enabled=False) as waveform:
            mag = self.stft(waveform.detach())
            log_mag = torch.log(mag + 1e-6)
            log_mag = self.normalize(log_mag)
            return log_mag

    def mag_spec_to_audio(self, mag: Tensor) -> Tensor:
        """Converts a magnitude spectrogram to a waveform.

        Args:
            mag: Tensor of shape ``(..., num_frames, n_fft // 2 + 1)``.

        Returns:
            Tensor of shape ``(..., num_samples)``, the reconstructed waveform.
        """
        with autocast_tensors(mag, enabled=False) as mag:
            log_mag = self.denormalize(mag.detach())
            mag = (torch.exp(log_mag) - 1e-6).clamp_min_(1e-8)
            waveform = self.griffin_lim(mag)
            return waveform


class WorldFeatures(NamedTuple):
    sp: Tensor
    f0: Tensor
    ap: Tensor


class AudioPyworldConverter:
    """Defines a class for converting waveforms to PyWorld features and back.

    This function also normalizes the features to have zero mean and unit
    variance using statistics over time.

    Parameters:
        sample_rate: Sample rate of the audio.
        dim: Dimension of the PyWorld features.
        frame_period: Frame period in milliseconds.
        f0_floor: Minimum F0 value.
        f0_ceil: Maximum F0 value.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        dim: int = 24,
        frame_period: float = 5.0,
        f0_floor: float = 71.0,
        f0_ceil: float = 800.0,
    ) -> None:
        super().__init__()

        assert pyworld is not None, "PyWorld is not installed; please install it with `pip install pyworld`."

        self.sampling_rate = sample_rate
        self.dim = dim
        self.frame_period = frame_period
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil

        # Normalization parameters.
        self.running_mean = 0.0
        self.running_std = 0.0
        self.count = 0

    def normalize(self, sp: np.ndarray) -> np.ndarray:
        mean, std = sp.mean(), sp.std()
        alpha, beta = self.count / (self.count + 1), 1 / (self.count + 1)
        self.running_mean = self.running_mean * alpha + mean * beta
        self.running_std = self.running_std * alpha + std * beta
        self.count += 1
        return (sp - self.running_mean) / self.running_std

    def denormalize(self, sp: np.ndarray) -> np.ndarray:
        return sp * self.running_std + self.running_mean

    def audio_to_features(self, waveform: np.ndarray) -> WorldFeatures:
        assert pyworld is not None

        waveform = waveform.astype(np.float64)
        f0, timeaxis = pyworld.harvest(  # F0 estimation
            waveform,
            self.sampling_rate,
            frame_period=self.frame_period,
            f0_floor=self.f0_floor,
            f0_ceil=self.f0_ceil,
        )
        sp = pyworld.cheaptrick(waveform, f0, timeaxis, self.sampling_rate)  # Smoothed spectrogram
        ap = pyworld.d4c(waveform, f0, timeaxis, self.sampling_rate)  # Harmonics spectral envelope
        coded_sp = pyworld.code_spectral_envelope(sp, self.sampling_rate, self.dim)  # Mel-cepstral coefficients
        coded_sp = self.normalize(coded_sp)
        return WorldFeatures(sp=torch.from_numpy(coded_sp), f0=torch.from_numpy(f0), ap=torch.from_numpy(ap))

    def features_to_audio(self, features: WorldFeatures | tuple[Array, Array, Array]) -> np.ndarray:
        assert pyworld is not None

        coded_sp, f0, ap = (as_numpy_array(f) for f in features)
        coded_sp = self.denormalize(coded_sp)
        fftlen = pyworld.get_cheaptrick_fft_size(self.sampling_rate)  # Obtaining FFT size from the sampling rate
        decoded_sp = pyworld.decode_spectral_envelope(coded_sp, self.sampling_rate, fftlen)  # Decoding the spectrum
        wav = pyworld.synthesize(f0, decoded_sp, ap, self.sampling_rate, self.frame_period)  # Synthesizing the waveform
        return wav


class SpectrogramToMFCCs(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 128,
        n_mfcc: int = 40,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_stft: int = 201,
        norm: str | None = None,
        mel_scale: str = "htk",
        dct_norm: str = "ortho",
    ) -> None:
        super().__init__()

        # Convert raw spectrogram to MFCCs. This is differentiable since
        # the transformations are just matrix multiplications.
        self.mel_scale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min, f_max, n_stft, norm, mel_scale)
        dct_mat = A.create_dct(n_mfcc, n_mels, dct_norm)
        self.register_buffer("dct_mat", dct_mat)

    dct_mat: Tensor

    def forward(self, spec: Tensor) -> Tensor:
        x = self.mel_scale(spec)
        x = torch.einsum("...ij,ik->...kj", x, self.dct_mat)
        return x


def test_audio_adhoc() -> None:
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["mfcc", "stft", "mag-stft", "pyworld"], help="Mode to test.")
    parser.add_argument("audio_file", help="Path to a specific audio file.")
    parser.add_argument("--output-dir", default="out", help="Path to the output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_file = Path(args.audio_file)

    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform[0]  # Only use the first channel.

    if args.mode == "mfcc":
        mfcc_converter = AudioMfccConverter(sample_rate)
        mfcc = mfcc_converter.audio_to_spec(waveform)
        mfcc_waveform = mfcc_converter.spec_to_audio(mfcc)
        torchaudio.save(output_dir / "original.wav", waveform[None], sample_rate)
        torchaudio.save(output_dir / "reconstructed.wav", mfcc_waveform[None], sample_rate)
        return

    if args.mode == "stft":
        stft_converter = AudioStftConverter()
        stft = stft_converter.audio_to_spec(waveform)
        stft_waveform = stft_converter.spec_to_audio(stft)
        torchaudio.save(output_dir / "original.wav", waveform[None], sample_rate)
        torchaudio.save(output_dir / "reconstructed.wav", stft_waveform[None], sample_rate)
        return

    if args.mode == "mag-stft":
        mag_stft_converter = AudioMagStftConverter()
        mag_stft = mag_stft_converter.audio_to_mag_spec(waveform)
        mag_stft_waveform = mag_stft_converter.mag_spec_to_audio(mag_stft)
        torchaudio.save(output_dir / "original.wav", waveform[None], sample_rate)
        torchaudio.save(output_dir / "reconstructed.wav", mag_stft_waveform[None], sample_rate)
        return

    if args.mode == "pyworld":
        pyworld_converter = AudioPyworldConverter(sample_rate)
        coded_sp = pyworld_converter.audio_to_features(waveform.numpy())
        pyworld_waveform = pyworld_converter.features_to_audio(coded_sp)
        pyworld_waveform_tensor = torch.from_numpy(pyworld_waveform).to(torch.float32)
        torchaudio.save(output_dir / "original.wav", waveform[None], sample_rate)
        torchaudio.save(output_dir / "reconstructed.wav", pyworld_waveform_tensor[None], sample_rate)
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    # python -m ml.utils.spectrogram
    test_audio_adhoc()
