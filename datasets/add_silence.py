# -*- coding: utf-8 -*-
import numpy as np
import random
import librosa
import os
from shutil import copyfile
from scipy.io import wavfile

from preprocessor import find_files

clean_files = find_files('/home/jupyter/VAD/TIMIT/TRAIN')
# 16 kHz
sr = 16000
# silence length
sil_length = 1
small_num = 0.00005

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def main():
    for clean_file in clean_files:
        print(clean_file)
        file_title_ = clean_file.split("/")[-1].split('.')[0]
        clean_amp = load_wav(clean_file, sr=sr)
        rand_num = np.random.rand(sr*sil_length)*2-1
        clean_add_sil_amp = np.concatenate((rand_num*small_num, clean_amp, rand_num*small_num))
        parts = clean_file.split('/')[:-1]
        parts[5] += '_augmented1'
        output_dir = '/'.join(parts)
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir+'/'+file_title_+'_add_sil'+'.WAV'
        save_wav(clean_add_sil_amp, output_file, sr)
        
        # copy .PHN files
        parts_phn = clean_file.split('/')[:-1]
        input_phn = '/'.join(parts_phn) + '/' + file_title_ + '.PHN'
        output_phn = output_dir+'/'+file_title_+'_add_sil'+'.PHN'
        copyfile(input_phn, output_phn)
        


if __name__ == '__main__':
    main()
