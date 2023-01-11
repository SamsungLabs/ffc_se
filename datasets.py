import os
import random

import librosa
import numpy as np
import torch.distributions
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

import utils

datasets = utils.ClassRegistry()
loaders = utils.ClassRegistry()


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
    use_full_spec=False,
    return_mel_and_spec=False,
):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    result = spectral_normalize_torch(spec)

    if not use_full_spec:
        mel = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
        mel = spectral_normalize_torch(mel)
        result = mel.squeeze()

        if return_mel_and_spec:
            spec = spectral_normalize_torch(spec)
            return result, spec
    return result


@loaders.add_to_registry("infinite", ("train", "val", "test"))
class InfiniteLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        *args,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        infinite=True,
        device=None,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        **kwargs
    ):
        super().__init__(
            *args,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="fork" if num_workers > 0 else None,
            **kwargs
        )
        self.infinite = infinite
        self.device = device
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.dataset_iterator)
        except StopIteration:
            if self.infinite:
                self.dataset_iterator = super().__iter__()
                x = next(self.dataset_iterator)
            else:
                raise
        if self.device is not None:
            x = utils.move_to_device(x, self.device)
        return x


def split_audios(audios, segment_size, split):
    audios = [torch.FloatTensor(audio).unsqueeze(0) for audio in audios]
    if split:
        if audios[0].size(1) >= segment_size:
            max_audio_start = audios[0].size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            audios = [
                audio[:, audio_start : audio_start + segment_size] for audio in audios
            ]
        else:
            audios = [
                torch.nn.functional.pad(
                    audio,
                    (0, segment_size - audio.size(1)),
                    "constant",
                )
                for audio in audios
            ]
    audios = [audio.squeeze(0).numpy() for audio in audios]
    return audios


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        noisy_wavs_dir,
        clean_wavs_dir=None,
        path_prefix=None,
        segment_size=8192,
        sampling_rate=16000,
        split=True,
        shuffle=False,
        device=None,
        input_freq=None,
    ):
        if path_prefix:
            if clean_wavs_dir:
                clean_wavs_dir = os.path.join(path_prefix, clean_wavs_dir)
            noisy_wavs_dir = os.path.join(path_prefix, noisy_wavs_dir)

        if clean_wavs_dir:
            self.audio_files = self.read_files_list(clean_wavs_dir, noisy_wavs_dir)
        else:
            self.audio_files = self.read_noisy_list(noisy_wavs_dir)

        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)

        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.input_freq = input_freq

    @staticmethod
    def read_files_list(clean_wavs_dir, noisy_wavs_dir):
        fn_lst_clean = os.listdir(clean_wavs_dir)
        fn_lst_noisy = os.listdir(noisy_wavs_dir)
        assert set(fn_lst_clean) == set(fn_lst_noisy)
        return sorted(fn_lst_clean)

    @staticmethod
    def read_noisy_list(noisy_wavs_dir):
        fn_lst_noisy = os.listdir(noisy_wavs_dir)
        return sorted(fn_lst_noisy)

    def make_input(
        self, clean_audio: np.ndarray, noisy_audio: np.ndarray
    ) -> np.ndarray:
        """
        Input arguments have the same length
        """
        raise NotImplementedError()

    def split_audios(self, audios):
        return split_audios(audios, self.segment_size, self.split)

    def __getitem__(self, index):
        fn = self.audio_files[index]

        if self.clean_wavs_dir:
            clean_audio = librosa.load(
                os.path.join(self.clean_wavs_dir, fn),
                sr=self.sampling_rate,
                res_type="polyphase",
            )[0]
        else:
            clean_audio = None

        noisy_audio = librosa.load(
            os.path.join(self.noisy_wavs_dir, fn),
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]

        if clean_audio is not None:
            clean_audio, noisy_audio = self.split_audios([clean_audio, noisy_audio])
        else:
            noisy_audio = self.split_audios([noisy_audio])[0]

        input_audio = self.make_input(clean_audio, noisy_audio)

        if clean_audio is not None:
            assert input_audio.shape[1] == clean_audio.size
            audio = torch.FloatTensor(normalize(clean_audio) * 0.95)
            audio = audio.unsqueeze(0)
        else:
            audio = torch.Tensor()

        input_audio = torch.FloatTensor(input_audio)

        return input_audio, audio

    def __len__(self):
        return len(self.audio_files)


@datasets.add_to_registry("inference_1ch", ("val", "test"))
class Inference1ChannelDataset(InferenceDataset):
    def make_input(self, clean_audio, noisy_audio):
        return normalize(noisy_audio)[None] * 0.95
