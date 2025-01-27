from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from pretty_midi import PrettyMIDI
import librosa
import torch
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tensorflow as tf
import sys
sys.path.append('./audiotools_mir/audiotools/basic-pitch')
from basic_pitch.inference import predict

from ..core import AudioSignal
from ..core import util

class AudioLoader:
    """Loads audio endlessly from a list of audio sources
    containing paths to audio files. Audio sources can be
    folders full of audio files (which are found via file
    extension) or by providing a CSV file which contains paths
    to audio files.

    Parameters
    ----------
    sources : List[str], optional
        Sources containing folders, or CSVs with
        paths to audio files, by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    relative_path : str, optional
        Path audio should be loaded relative to, by default ""
    transform : Callable, optional
        Transform to instantiate alongside audio sample,
        by default None
    ext : List[str]
        List of extensions to find audio within each source by. Can
        also be a file name (e.g. "vocals.wav"). by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    shuffle: bool
        Whether to shuffle the files within the dataloader. Defaults to True.
    shuffle_state: int
        State to use to seed the shuffle of the files.
    """

    def __init__(
            self,
            sources: List[str] = None,
            weights: List[float] = None,
            transform: Callable = None,
            relative_path: str = "",
            ext: List[str] = util.AUDIO_EXTENSIONS,
            shuffle: bool = True,
            shuffle_state: int = 0,
            normalise_audio: bool = False,
            n_instruments: int = 1,
            noisy_labels: bool = False,
            n_notes: int = 88,
            mel_params: dict = None,
            codec_rate: int = None,
    ):
        sources = [sources] if isinstance(sources, str) else sources
        self.audio_lists = util.read_sources(
            sources, relative_path=relative_path, ext=ext
        )

        self.audio_indices = [
            (src_idx, item_idx)
            for src_idx, src in enumerate(self.audio_lists)
            for item_idx in range(len(src))
        ]
        if shuffle:
            state = util.random_state(shuffle_state)
            state.shuffle(self.audio_indices)

        self.sources = sources
        self.weights = weights
        self.transform = transform
        self.normalise_audio = normalise_audio
        self.n_instruments = n_instruments
        self.noisy_labels = noisy_labels
        self.n_notes = n_notes
        self.mel_params = mel_params
        self.codec_rate = codec_rate
        self.midi_offset = 21 if self.n_notes == 88 else 0
        self.max_freq_idx = 87 if self.n_notes == 88 else 127

    def __call__(
            self,
            state,
            sample_rate: int,
            duration: float,
            loudness_cutoff: float = -40,
            num_channels: int = 1,
            offset: float = None,
            source_idx: int = None,
            item_idx: int = None,
            global_idx: int = None
    ):
        if source_idx is not None and item_idx is not None:
            try:
                audio_info = self.audio_lists[source_idx][item_idx]
            except:
                audio_info = {"path": "none"}
        elif global_idx is not None:
            source_idx, item_idx = self.audio_indices[
                global_idx % len(self.audio_indices)
                ]
            audio_info = self.audio_lists[source_idx][item_idx]
        else:
            audio_info, source_idx, item_idx = util.choose_from_list_of_lists(
                state, self.audio_lists, p=self.weights
            )

        path = audio_info["path"]
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)

        if path != "none":
            if offset is None:
                signal = AudioSignal.salient_excerpt(
                    path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
            else:
                signal = AudioSignal(
                    path,
                    offset=offset,
                    duration=duration,
                )

        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)

        if signal.duration < duration:
            signal = signal.zero_pad_to(int(duration * sample_rate))

        if self.normalise_audio:
            signal.normalize(-24) # normalise to -24 dB
            signal.ensure_max_of_audio(1.0) # make sure there is no clipping
        for k, v in audio_info.items():
            signal.metadata[k] = v

        # calculate mel filterbank
        if self.mel_params:
            assert self.codec_rate == None, "Please choose either a mel filterbank or a codec to generate labels for."
            mel = signal.mel_filterbank(**self.mel_params)
            n_frames = mel.shape[1] # 1, 128, 128 -> C, N_FRAMES, BINS
            hop_size_samples = int(self.mel_params['frame_shift'] * 10**-3 * signal.sample_rate)
            dt = hop_size_samples / signal.sample_rate
            item = {"mel": mel, "path": path}
            item["label"] = self.get_noisy_label(signal, n_frames=n_frames, dt=dt).to(
                torch.float32) if self.noisy_labels else self.get_midi_label(n_frames=n_frames, dt=dt,
                                                                             path=item["path"]).to(torch.float32)
        else:
            item = {
                "signal": signal,
                "source_idx": source_idx,
                "item_idx": item_idx,
                "source": str(self.sources[source_idx]),
                "path": str(path),
                "offset": signal.metadata["offset"]
            }
            item["label"] = self.get_noisy_label_for_codec(signal.audio_data, sample_rate=signal.sample_rate, duration=signal.duration, codec_rate=self.codec_rate).to(
                torch.float32) if self.noisy_labels else self.get_midi_label_for_codec(sample_rate=signal.sample_rate, offset=item['offset'], duration=signal.duration, path=item["path"], codec_rate=self.codec_rate).to(torch.float32)

        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        return item


    def get_midi_path(self, path):
        """Function to infer MIDI path corresponding to an audio file based on dataset."""
        if 'slakh' in path.lower():
            label_path = Path(path).parent / 'all_src.mid'
        elif 'maestro' in path.lower():
            label_path = Path(path).parent / f'{Path(path).stem}.midi'
        elif 'musicnet' in path.lower():
            label_path = Path(path).parents[1] / f"{(Path(path).parents[0].name).split('_')[0]}_labels" / f"{Path(path).stem}.mid"
        else:
            raise ValueError(f'Dataset not supported, {path}')
        assert label_path.exists(), f'Label path {label_path} does not exist!'
        return label_path


    def get_midi_label_for_codec(self, offset, duration, path, codec_rate):
        "Function to return ground truth label for codec."
        # todo: add support for multi-instrument roll
        num_samples = duration * codec_rate
        start_time = offset
        end_time = start_time + duration
        if self.n_instruments > 1:
            from ..data.vocabulary import program_to_index, program_to_name
            label = torch.zeros(num_samples, self.n_notes, self.n_instruments, dtype=torch.int32)
        else:
            label = torch.zeros(num_samples, self.n_notes, dtype=torch.int32)

        label_path = self.get_midi_path(path)
        midi_data = PrettyMIDI(str(label_path))

        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start >= start_time:
                        note_start = librosa.time_to_samples(note.start - start_time, sr=codec_rate)
                        pitch_index = note.pitch - self.midi_offset
                        assert pitch_index >= 0, f'Pitch index is negative: {pitch_index}'

                        if note.end <= end_time:
                            note_end = librosa.time_to_samples(note.end - start_time, sr=codec_rate)
                            label[note_start:note_end, pitch_index] = 1
                        else:
                            label[note_start:, pitch_index] = 1
        return label


    def get_midi_label(self, n_frames, dt, path):
        "Function to return ground truth label for spectrogram"
        # dt = frame shift in seconds

        if self.n_instruments > 1:
            from ..data.vocabulary import program_to_index, program_to_name
            label = torch.zeros(n_frames, self.n_notes, self.n_instruments, dtype=torch.int32)
        else:
            label = torch.zeros(n_frames, self.n_notes, dtype=torch.int32)

        label_path = self.get_midi_path(path)
        print('Audio path:', path)
        print('MIDI path:', label_path)
        print()
        midi_data = PrettyMIDI(str(label_path))

        for instrument in midi_data.instruments:
            # only consider non-percussive instruments
            if not instrument.is_drum:
                for note in instrument.notes:
                    frame_start = int(np.round(note.start / dt))
                    frame_end = int(np.round(note.end / dt))
                    pitch_index = note.pitch - self.midi_offset

                    # even if the event was too short, always produce a label!
                    if frame_start == frame_end:
                        frame_end += 1

                    if pitch_index <= self.max_freq_idx:
                        if self.n_instruments > 1:
                            try:
                                label[frame_start:frame_end, pitch_index, program_to_index[instrument.program]] = 1
                            except:
                                pass
                                #print(f'Warning: MIDI instrument {instrument.program} not in vocabulary, {path}')
                        else:
                            label[frame_start:frame_end, pitch_index] = 1
                    else:
                        pass
                        #print(f'Warning: Pitch index {pitch_index} out of range, {path}')
        return label


    def get_noisy_label(self, signal, n_frames, dt):
        "Function to return a noisy label for spectrogram"
        # todo: add option to generate label for audio codec
        # dt = frame shift in seconds

        with tf.device('/cpu:0'):
            _, midi_data, _ = predict(signal.audio_data.squeeze().squeeze().detach().cpu().numpy(), sample_rate=signal.sample_rate)
        label = torch.zeros(n_frames, self.n_notes, dtype=torch.int32)

        for instrument in midi_data.instruments:
            # only consider non-percussive instruments
            if not instrument.is_drum:
                for note in instrument.notes:
                    frame_start = int(np.round(note.start / dt))
                    frame_end = int(np.round(note.end / dt))
                    pitch_index = note.pitch - self.midi_offset

                    # even if the event was too short, always produce a label!
                    if frame_start == frame_end:
                        frame_end += 1

                    if pitch_index <= self.max_freq_idx:
                        label[frame_start:frame_end, pitch_index] = 1
                    else:
                        print(f'Warning: Pitch index {pitch_index} out of range, {item["path"]}')
        return label


    def get_noisy_label_for_codec(self, signal, sample_rate, duration, codec_rate):
        """Function to generate noisy label for audio codec using the pretrained Basic Pitch model.

        Parameters
        ----------
        signal : torch.Tensor
            Audio signal tensor
        sample_rate : int
            Sample rate of the audio signal
        duration : float
            Duration of the audio signal in seconds
        codec_rate : int
            Sample rate of the codec algorithm
        """
        num_samples = int(duration * codec_rate)
        label = torch.zeros(num_samples, self.n_notes, dtype=torch.float32)

        with tf.device('/cpu:0'):
            _, midi_data, _ = predict(signal.squeeze().squeeze().detach().cpu().numpy(), sample_rate=sample_rate)

        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note_start = librosa.time_to_samples(note.start, sr=codec_rate)
                    note_end = librosa.time_to_samples(note.end, sr=codec_rate)
                    pitch_index = note.pitch - self.midi_offset
                    assert pitch_index >= 0, f'Pitch index is negative: {note.pitch}'

                    assert note_end >= note_start, "End sample must be later than start sample!"
                    if note_start == note_end:
                        note_end = note_end + 1
                    label[note_start:note_end, pitch_index] = 1

        return label

def default_matcher(x, y):
    return Path(x).parent == Path(y).parent

def align_lists(lists, matcher: Callable = default_matcher):
    longest_list = lists[np.argmax([len(l) for l in lists])]
    for i, x in enumerate(longest_list):
        for l in lists:
            if i >= len(l):
                l.append({"path": "none"})
            elif not matcher(l[i]["path"], x["path"]):
                l.insert(i, {"path": "none"})
    return lists


class AudioDataset:
    """Loads audio from multiple loaders (with associated transforms)
    for a specified number of samples. Excerpts are drawn randomly
    of the specified duration, above a specified loudness threshold
    and are resampled on the fly to the desired sample rate
    (if it is different from the audio source sample rate).

    This takes either a single AudioLoader object,
    a dictionary of AudioLoader objects, or a dictionary of AudioLoader
    objects. Each AudioLoader is called by the dataset, and the
    result is placed in the output dictionary. A transform can also be
    specified for the entire dataset, rather than for each specific
    loader. This transform can be applied to the output of all the
    loaders if desired.

    AudioLoader objects can be specified as aligned, which means the
    loaders correspond to multitrack audio (e.g. a vocals, bass,
    drums, and other loader for multitrack music mixtures).


    Parameters
    ----------
    loaders : Union[AudioLoader, List[AudioLoader], Dict[str, AudioLoader]]
        AudioLoaders to sample audio from.
    sample_rate : int
        Desired sample rate.
    n_examples : int, optional
        Number of examples (length of dataset), by default 1000
    duration : float, optional
        Duration of audio samples, by default 0.5
    loudness_cutoff : float, optional
        Loudness cutoff threshold for audio samples, by default -40
    num_channels : int, optional
        Number of channels in output audio, by default 1
    transform : Callable, optional
        Transform to instantiate alongside each dataset item, by default None
    aligned : bool, optional
        Whether the loaders should be sampled in an aligned manner (e.g. same
        offset, duration, and matched file name), by default False
    shuffle_loaders : bool, optional
        Whether to shuffle the loaders before sampling from them, by default False
    matcher : Callable
        How to match files from adjacent audio lists (e.g. for a multitrack audio loader),
        by default uses the parent directory of each file.
    without_replacement : bool
        Whether to choose files with or without replacement, by default True.


    Examples
    --------
    >>> from audiotools.data.datasets import AudioLoader
    >>> from audiotools.data.datasets import AudioDataset
    >>> from audiotools import transforms as tfm
    >>> import numpy as np
    >>>
    >>> loaders = [
    >>>     AudioLoader(
    >>>         sources=[f"tests/audio/spk"],
    >>>         transform=tfm.Equalizer(),
    >>>         ext=["wav"],
    >>>     )
    >>>     for i in range(5)
    >>> ]
    >>>
    >>> dataset = AudioDataset(
    >>>     loaders = loaders,
    >>>     sample_rate = 44100,
    >>>     duration = 1.0,
    >>>     transform = tfm.RescaleAudio(),
    >>> )
    >>>
    >>> item = dataset[np.random.randint(len(dataset))]
    >>>
    >>> for i in range(len(loaders)):
    >>>     item[i]["signal"] = loaders[i].transform(
    >>>         item[i]["signal"], **item[i]["transform_args"]
    >>>     )
    >>>     item[i]["signal"].widget(i)
    >>>
    >>> mix = sum([item[i]["signal"] for i in range(len(loaders))])
    >>> mix = dataset.transform(mix, **item["transform_args"])
    >>> mix.widget("mix")

    Below is an example of how one could load MUSDB multitrack data:

    >>> import audiotools as at
    >>> from pathlib import Path
    >>> from audiotools import transforms as tfm
    >>> import numpy as np
    >>> import torch
    >>>
    >>> def build_dataset(
    >>>     sample_rate: int = 44100,
    >>>     duration: float = 5.0,
    >>>     musdb_path: str = "~/.data/musdb/",
    >>> ):
    >>>     musdb_path = Path(musdb_path).expanduser()
    >>>     loaders = {
    >>>         src: at.datasets.AudioLoader(
    >>>             sources=[musdb_path],
    >>>             transform=tfm.Compose(
    >>>                 tfm.VolumeNorm(("uniform", -20, -10)),
    >>>                 tfm.Silence(prob=0.1),
    >>>             ),
    >>>             ext=[f"{src}.wav"],
    >>>         )
    >>>         for src in ["vocals", "bass", "drums", "other"]
    >>>     }
    >>>
    >>>     dataset = at.datasets.AudioDataset(
    >>>         loaders=loaders,
    >>>         sample_rate=sample_rate,
    >>>         duration=duration,
    >>>         num_channels=1,
    >>>         aligned=True,
    >>>         transform=tfm.RescaleAudio(),
    >>>         shuffle_loaders=True,
    >>>     )
    >>>     return dataset, list(loaders.keys())
    >>>
    >>> train_data, sources = build_dataset()
    >>> dataloader = torch.utils.data.DataLoader(
    >>>     train_data,
    >>>     batch_size=16,
    >>>     num_workers=0,
    >>>     collate_fn=train_data.collate,
    >>> )
    >>> batch = next(iter(dataloader))
    >>>
    >>> for k in sources:
    >>>     src = batch[k]
    >>>     src["transformed"] = train_data.loaders[k].transform(
    >>>         src["signal"].clone(), **src["transform_args"]
    >>>     )
    >>>
    >>> mixture = sum(batch[k]["transformed"] for k in sources)
    >>> mixture = train_data.transform(mixture, **batch["transform_args"])
    >>>
    >>> # Say a model takes the mix and gives back (n_batch, n_src, n_time).
    >>> # Construct the targets:
    >>> targets = at.AudioSignal.batch([batch[k]["transformed"] for k in sources], dim=1)

    Similarly, here's example code for loading Slakh data:

    >>> import audiotools as at
    >>> from pathlib import Path
    >>> from audiotools import transforms as tfm
    >>> import numpy as np
    >>> import torch
    >>> import glob
    >>>
    >>> def build_dataset(
    >>>     sample_rate: int = 16000,
    >>>     duration: float = 10.0,
    >>>     slakh_path: str = "~/.data/slakh/",
    >>> ):
    >>>     slakh_path = Path(slakh_path).expanduser()
    >>>
    >>>     # Find the max number of sources in Slakh
    >>>     src_names = [x.name for x in list(slakh_path.glob("**/*.wav"))  if "S" in str(x.name)]
    >>>     n_sources = len(list(set(src_names)))
    >>>
    >>>     loaders = {
    >>>         f"S{i:02d}": at.datasets.AudioLoader(
    >>>             sources=[slakh_path],
    >>>             transform=tfm.Compose(
    >>>                 tfm.VolumeNorm(("uniform", -20, -10)),
    >>>                 tfm.Silence(prob=0.1),
    >>>             ),
    >>>             ext=[f"S{i:02d}.wav"],
    >>>         )
    >>>         for i in range(n_sources)
    >>>     }
    >>>     dataset = at.datasets.AudioDataset(
    >>>         loaders=loaders,
    >>>         sample_rate=sample_rate,
    >>>         duration=duration,
    >>>         num_channels=1,
    >>>         aligned=True,
    >>>         transform=tfm.RescaleAudio(),
    >>>         shuffle_loaders=False,
    >>>     )
    >>>
    >>>     return dataset, list(loaders.keys())
    >>>
    >>> train_data, sources = build_dataset()
    >>> dataloader = torch.utils.data.DataLoader(
    >>>     train_data,
    >>>     batch_size=16,
    >>>     num_workers=0,
    >>>     collate_fn=train_data.collate,
    >>> )
    >>> batch = next(iter(dataloader))
    >>>
    >>> for k in sources:
    >>>     src = batch[k]
    >>>     src["transformed"] = train_data.loaders[k].transform(
    >>>         src["signal"].clone(), **src["transform_args"]
    >>>     )
    >>>
    >>> mixture = sum(batch[k]["transformed"] for k in sources)
    >>> mixture = train_data.transform(mixture, **batch["transform_args"])

    """

    def __init__(
            self,
            loaders: Union[AudioLoader, List[AudioLoader], Dict[str, AudioLoader]],
            sample_rate: int,
            n_examples: int = 1000,
            duration: float = 0.5,
            offset: float = None,
            loudness_cutoff: float = -40,
            num_channels: int = 1,
            transform: Callable = None,
            aligned: bool = False,
            shuffle_loaders: bool = False,
            matcher: Callable = default_matcher,
            without_replacement: bool = True,
    ):
        # Internally we convert loaders to a dictionary
        if isinstance(loaders, list):
            loaders = {i: l for i, l in enumerate(loaders)}
        elif isinstance(loaders, AudioLoader):
            loaders = {0: loaders}

        self.loaders = loaders
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels

        self.length = n_examples
        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
        self.offset = offset
        self.aligned = aligned
        self.shuffle_loaders = shuffle_loaders
        self.without_replacement = without_replacement

        if aligned:
            loaders_list = list(loaders.values())
            for i in range(len(loaders_list[0].audio_lists)):
                input_lists = [l.audio_lists[i] for l in loaders_list]
                # Alignment happens in-place
                align_lists(input_lists, matcher)

    def __getitem__(self, idx):
        state = util.random_state(idx)
        offset = None if self.offset is None else self.offset
        item = {}

        keys = list(self.loaders.keys())
        if self.shuffle_loaders:
            state.shuffle(keys)

        loader_kwargs = {
            "state": state,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "loudness_cutoff": self.loudness_cutoff,
            "num_channels": self.num_channels,
            "global_idx": idx if self.without_replacement else None,
        }

        # Draw item from first loader
        loader = self.loaders[keys[0]]
        item[keys[0]] = loader(**loader_kwargs)

        for key in keys[1:]:
            loader = self.loaders[key]
            if self.aligned:
                # Path mapper takes the current loader + everything
                # returned by the first loader.
                offset = item[keys[0]]["signal"].metadata["offset"]
                loader_kwargs.update(
                    {
                        "offset": offset,
                        "source_idx": item[keys[0]]["source_idx"],
                        "item_idx": item[keys[0]]["item_idx"],
                    }
                )
            item[key] = loader(**loader_kwargs)

        # Sort dictionary back into original order
        keys = list(self.loaders.keys())
        item = {k: item[k] for k in keys}

        item["idx"] = idx
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(
                state=state, signal=item[keys[0]]["signal"]
            )

        # If there's only one loader, pop it up
        # to the main dictionary, instead of keeping it
        # nested.
        if len(keys) == 1:
            item.update(item.pop(keys[0]))

        return item

    def __len__(self):
        return self.length

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        """Collates items drawn from this dataset. Uses
        :py:func:`audiotools.core.util.collate`.

        Parameters
        ----------
        list_of_dicts : typing.Union[list, dict]
            Data drawn from each item.
        n_splits : int
            Number of splits to make when creating the batches (split into
            sub-batches). Useful for things like gradient accumulation.

        Returns
        -------
        dict
            Dictionary of batched data.
        """
        return util.collate(list_of_dicts, n_splits=n_splits)


class ConcatDataset(AudioDataset):
    def __init__(self, datasets: list):
        self.datasets = datasets

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        dataset = self.datasets[idx % len(self.datasets)]
        return dataset[idx // len(self.datasets)]


class ResumableDistributedSampler(DistributedSampler):  # pragma: no cover
    """Distributed sampler that can be resumed from a given start index."""

    def __init__(self, dataset, start_idx: int = None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx // self.num_replicas if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch


class ResumableSequentialSampler(SequentialSampler):  # pragma: no cover
    """Sequential sampler that can be resumed from a given start index."""

    def __init__(self, dataset, start_idx: int = None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch
