# -*- coding: utf-8 -*-
import numpy as np
import random
import librosa
import os
from scipy.io import wavfile
from shutil import copyfile
from preprocessor import find_files

clean_files = find_files(os.path.join('/home/jupyter/VAD/TIMIT/TRAIN_augmented'), pattern='**/*.WAV')
# babble car airport exhibition restaurant street subway train
#noise_files = find_files('./../../../Audio_augmentation/Official_Noise/NOISEX')
noise_files = find_files(os.path.join('/home/jupyter/VAD/AURORA'),pattern='**/*.wav')
snr_ = [-5, -10, -15, 0]
# 16 kHz
sr = 16000


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    return clean_rms / (10**a)


def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    amplitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amplitude


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))


def main():
    for clean_file in clean_files:
        print(clean_file)
        if clean_file.find('NOISEX') > 0 or clean_file.find('AURORA') > 0:
            continue

        file_title_ = clean_file.split("/")[-1].split('.')[0]
        for noise_file in noise_files:
            noise_title_ = noise_file.split("/")[-1].split('.')[0]
            for snr in snr_:
                clean_amp = load_wav(clean_file, sr=sr)
                noise_amp = load_wav(noise_file, sr=sr)

                if len(clean_amp) > len(noise_amp):
                    noise_amp = np.pad(noise_amp, (0, (len(clean_amp)-len(noise_amp)) * 5), mode='wrap')
                start = random.randint(0, len(noise_amp)-len(clean_amp))
                clean_rms = cal_rms(clean_amp)
                split_noise_amp = noise_amp[start: start + len(clean_amp)]
                noise_rms = cal_rms(split_noise_amp)

                adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

                adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)
                mixed_amp = (clean_amp + adjusted_noise_amp)

                if mixed_amp.max(axis=0) > 32767:
                    mixed_amp = mixed_amp * (32767/mixed_amp.max(axis=0))
                    clean_amp = clean_amp * (32767/mixed_amp.max(axis=0))
                    adjusted_noise_amp = adjusted_noise_amp * (32767/mixed_amp.max(axis=0))

                parts = clean_file.split('/')[:-1]
                output_dir = '/'.join(parts)
                os.makedirs(output_dir, exist_ok=True)
                #output_noisy_file = output_dir + '/' + '[NOISEX]' + file_title_ + '_SNR(' + '{:02d}'.format(snr) + ')_' + noise_title_ + '.WAV'
                output_noisy_file = output_dir+'/'+'[AURORA]'+file_title_+'_SNR('+'{:02d}'.format(snr)+')_'+noise_title_+'.WAV'
#                 print('output_noisy_file: ',output_noisy_file)
                save_wav(mixed_amp, output_noisy_file, sr)
                
                
                # copy .PHN files
                parts_phn = clean_file.split('/')[:-1]
                input_phn = '/'.join(parts_phn) + '/' + file_title_ + '.PHN'
                output_phn = output_dir+'/'+'[AURORA]'+file_title_+'_SNR('+'{:02d}'.format(snr)+')_'+noise_title_+'.PHN'
#                 print('input_phn: ',input_phn)
#                 print('output_phn: ',output_phn)
                copyfile(input_phn, output_phn)
                
            
        


if __name__ == '__main__':
    main()
