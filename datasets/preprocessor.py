import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

import numpy as np
from datasets import audio


def find_files(directory, pattern='**/*.WAV'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


def build_from_path(hparams, input_dirs, mel_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a given input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dirs: input directory that contains the files to prerocess
        - mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    for input_dir in input_dirs:
        print(input_dir)
        files = find_files(os.path.join(input_dir))
        print('len of files: ',len(files))
        for wav_path in files:
            # print('wav_path: ',wav_path)
            text_path = os.path.splitext(wav_path)[0]
            text_parts = text_path.split('/')
#             print('text_parts: ',text_parts)
#             text_parts[0] = text_parts[0][:-10]
#             text_parts[3] = text_parts[3].split('_')[0]
#             print('text_parts[3]: ',text_parts[3])
#             if text_parts[3].find('AURORA') > 0 or text_parts[3].find('NOISEX') > 0:
#                 text_parts[3] = text_parts[3].split(']')[-1]
            text_path = '/'.join(text_parts) + '.PHN'
#             text_path = os.path.join(os.path.split(wav_path)[0].split('_')[0] + '_clean' +
#                                      os.path.split(wav_path)[0].split('-20')[1],
#                                      os.path.split(wav_path)[1].split(']')[1].split('_SNR')[0] + '.PHN')
            # print('.PHN: ',text_path)
            with open(text_path, encoding='utf-8') as f:
                lines = f.readlines()
                start = int(lines[1].split(' ')[0]) 
                end = int(lines[-2].split(' ')[1]) 
                futures.append(executor.submit(partial(_process_utterance, mel_dir, index, wav_path, start, end, hparams)))
                index += 1
            # break
        # break

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, index, wav_path, start, end, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - start, end: start, end points of speech
        - hparams: hyper parameters

    Returns:
        - A tuple: (wav_path, mel_filename, time_steps, mel_frames, start, end)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
        return None

    start += int(1 * hparams.sample_rate)
    end += int(1 * hparams.sample_rate)

    #rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    #M-AILABS extra silence specific
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)

    #[-1, 1]
    out = wav
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(out_dtype)
    mel_frames = mel_spectrogram.shape[1]

    # Ensure time resolution adjustement between audio and mel-spectrogram
    pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))

    # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
    out = np.pad(out, (0, pad), mode='reflect')
    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)

    start = round(start/int(time_steps / mel_frames))
    end = round(end/int(time_steps / mel_frames))

    # Write the spectrogram and audio to disk
    mel_filename = 'mel-{}.npy'.format(index)
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return (wav_path, mel_filename, time_steps, mel_frames, start, end)


def build_from_path_libri(hparams, input_dirs, mel_dir, label_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess
        - mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
        - label: output directory of the preprocessed speech label dataset
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    for input_dir in input_dirs:
        files = find_files(os.path.join(input_dir), pattern='**/*.WAV')
        for wav_path in files:
            # text_path = os.path.splitext(wav_path)[0]
            # text_parts = text_path.split('/')
            label_path = os.path.join(os.path.split(wav_path)[0].split('_')[0] + '_clean',
                                      os.path.split(wav_path)[1].split(']')[1].split('_')[0] + '.label')
            with open(label_path, encoding='utf-8') as f:
                lines = f.readlines()

            label = np.asarray([int(line.strip('\n')) for line in lines])

            futures.append(executor.submit(partial(_process_utterance_libri, mel_dir, label_dir, index, wav_path, label, hparams)))
            index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance_libri(mel_dir, label_dir, index, wav_path, label, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - label_dir: the directory to write the label into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - label: time steps spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (wav_path, mel_filename, time_steps, mel_frames, label_filename)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
        return None

    #rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    #M-AILABS extra silence specific
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)

    #[-1, 1]
    out = wav
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(out_dtype)
    mel_spectrogram = mel_spectrogram[:, -len(label):]
    mel_frames = mel_spectrogram.shape[1]
    time_steps = len(out)

    # Write the spectrogram and audio to disk
    mel_filename = 'mel-{}.npy'.format(index)
    label_filename = 'label-{}.npy'.format(index)
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(label_dir, label_filename), label, allow_pickle=False)

    # Return a tuple describing this training example
    return (wav_path, mel_filename, time_steps, mel_frames, label_filename)


def build_from_path_ispl(hparams, input_dirs, mel_dir, label_dir, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dirs: input directory that contains the files to prerocess
        - mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
        - label_dir: the directory to write the label into
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    futures = []
    index = 1
    for input_dir in input_dirs:
        files = find_files(os.path.join(input_dir))
        for wav_path in files:
            file_name = wav_path.split("\\")[-1]
            if int(file_name.split('.')[0]) <= 10:
                label_path = wav_path.split("\\")[0] + '/label.txt'
                with open(label_path, encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    if file_name in line:
                        labels = line.replace('[', '').replace(']', '').split(':')[1].replace(',\n', '').split(',')
                        start = []
                        end = []
                        for idx in range(0, len(labels), 2):
                            start.append(int(labels[idx]))
                            end.append(int(labels[idx+1]))

            try:
                # Load the audio as numpy array
                wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
            except FileNotFoundError:  # catch missing wav exception
                print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
                return None

            # rescale wav
            if hparams.rescale:
                wav = wav / np.abs(wav).max() * hparams.rescaling_max

            # M-AILABS extra silence specific
            if hparams.trim_silence:
                wav = audio.trim_silence(wav, hparams)

            # [-1, 1]
            out = wav
            out_dtype = np.float32

            if int(file_name.split('.')[0]) <= 10:
                label = np.zeros_like(out)
                for idx in range(len(start)):
                    start[idx] = int(start[idx] / 1000 * hparams.sample_rate)
                    end[idx] = int(end[idx] / 1000 * hparams.sample_rate)
                    label[start[idx]:end[idx]] = 1.
            else:
                label = wav_path.split('.')[0] + '.label'
                with open(label, encoding='utf-8') as f:
                    lines = f.readlines()
                label = np.asarray([int(line.strip('\n')) for line in lines])

            # Compute the mel scale spectrogram from the wav
            mel_spectrogram = audio.melspectrogram(wav, hparams).astype(out_dtype)
            mel_spectrogram = mel_spectrogram[:, -len(label):]
            mel_frames = mel_spectrogram.shape[1]

            # Ensure time resolution adjustement between audio and mel-spectrogram
            pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))

            if int(file_name.split('.')[0]) <= 10:
                # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
                out = np.pad(out, (0, pad), mode='reflect')
                label = np.pad(label, (0, pad), mode='reflect')
                assert len(out) >= mel_frames * audio.get_hop_size(hparams)

                # time resolution adjustement
                # ensure length of raw audio is multiple of hop size so that we can use
                # transposed convolution to upsample
                out = out[:mel_frames * audio.get_hop_size(hparams)]
                label = label[:mel_frames * audio.get_hop_size(hparams)]
                assert len(out) % audio.get_hop_size(hparams) == 0
                label = label[::audio.get_hop_size(hparams)]

                time_steps = len(out)
            else:
                time_steps = len(out)

            # Write the spectrogram and audio to disk
            mel_filename = 'mel-{}.npy'.format(index)
            label_filename = 'label-{}.npy'.format(index)
            np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
            np.save(os.path.join(label_dir, label_filename), label, allow_pickle=False)
            futures.append((wav_path, mel_filename, time_steps, mel_frames, label_filename))
            index += 1

    return [future for future in tqdm(futures)]