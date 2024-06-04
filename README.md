# Spectro-Temporal Attention Based Voice Activity Detection (pytorch)

![Language](https://img.shields.io/badge/language-Python-orange.svg)&nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)&nbsp;

Pytorch implementation of ["Spectro-Temporal Attention-based Voice Activity Detection"](https://ieeexplore.ieee.org/document/8933025)

My implementation of STAM provides slightly better performance compared to the original tensorflow one:

Tensorflow: Global AUC: 99.86, F1-score: 98.15, DCF: 1.32, accuracy: 97.90, precision: 99.10

Pytorch: Global AUC: 99.87, F1-score: 98.31, DCF: 1.18, accuracy: 98.07, precision: 99.06

## Training data
TIMIT training data + NOISEX (SNR: -10, -5, 0, 5, 10dB)
## Testing data
TIMIT testing data + AURORA (SNR: -5, 0, 5, 10dB)
## Reference
https://github.com/ByeonghakKim/ispl-speech/tree/05b5a175edf2c312db398bad82a3b3cc56da4756/Voice%20activity%20detection
