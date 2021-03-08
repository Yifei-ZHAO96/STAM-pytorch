# spectro-temporal attention based voice activity detection (pytorch)
Pytorch implementation of "spectro-temporal attention-based voice activity detection"

My implementation of STAM provides slightly bettern performance compared to the original tensorflow one:

Tensorflow: Global AUC: 99.86, F1-score: 98.15, DCF: 1.32, acc: 97.90, precision: 99.10
Pytorch: Global AUC: 99.87, F1-score: 98.31, DCF: 1.18, acc: 98.07, precision: 99.06

## Training data
TIMIT training data + NOISEX (SNR: -10, -5, 0, 5, 10dB)
## Testing data
TIMIT testing data + AURORA (SNR: -5, 0, 5, 10dB)
## Reference
https://github.com/ByeonghakKim/ispl-speech/tree/05b5a175edf2c312db398bad82a3b3cc56da4756/Voice%20activity%20detection
