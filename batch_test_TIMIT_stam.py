import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn.functional as F
import os, gc

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
from VAD_module import VAD
from utils import SpeechDataset, HParams, bdnn_prediction

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def main():
    DEVICE = torch.device('cuda')
    CPU = torch.device('cpu')
    input_path = './testing_TIMIT_noisy/train.txt'
    print('input_path: {}'.format(input_path))
    path_checkpoint = "./checkpoint/STAM_weights_4_1.0.pth"  # 断点路径
    print('loading checkpoint: {}'.format(path_checkpoint))
    with open(input_path, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]

    noises = ['airport', 'babble', 'car', 'exhibition', 'restaurant', 'street', 'subway', 'train']
    snrs = ['SNR(-5)_', 'SNR(00)_', 'SNR(05)_', 'SNR(10)_']
    # snrs = ['SNR(-10)_', 'SNR(-15)_']
    # snrs = ['SNR(-20)_', 'SNR(-25)_']
    aucs = []
    features = []
    F1 = []
    DCF = []
    ACC = []
    PRECISION = []
    for snr in snrs:
        for noise in noises:
            features.append(snr + noise)

    gc.collect()
    torch.cuda.empty_cache()
    hparams = HParams()
    model = VAD(hparams).to(DEVICE)
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        model.eval()
        TN = 0
        TP = 0
        FP = 0
        FN = 0

        for feature in tqdm(features):
            labels = []
            vads = []
            count = 0
            test_meta = [meta for meta in metadata if feature in meta[0]]
            test_dataset = SpeechDataset(input_path, test_meta, hparams)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

            for train_data, train_label in test_loader:
                count += 1
                train_data = train_data[0]
                train_label = train_label[0]
                train_data = torch.as_tensor(train_data, dtype=torch.float32).to(DEVICE)
                label = train_label[:, int(np.floor(int(2 * (hparams.w - 1) / hparams.u + 3) / 2))]
                midnet_output, postnet_output, alpha = model(train_data)
                _, vad = bdnn_prediction(F.sigmoid(postnet_output).cpu().detach().numpy(), w=hparams.w, u=hparams.u)

                vads = np.concatenate((vads, vad[:, 0]), axis=None)
                labels = np.concatenate((labels, label), axis=None)

            fpr, tpr, _ = metrics.roc_curve(labels, vads, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)
            A = confusion_matrix(np.int8(labels), np.int8(vads.round()))
            TN += A[0][0]
            FN += A[1][0]
            FP += A[0][1]
            TP += A[1][1]
            f1 = 2 * A[1][1] / (2 * A[1][1] + A[0][1] + A[1][0])
            dcf = (0.75 * A[1][0] + 0.25 * A[0][1]) / sum(sum(A))
            acc = (A[0][0] + A[1][1]) / sum(sum(A))
            precision = A[1][1] / (A[1][1] + A[0][1])
            ACC.append(acc)
            PRECISION.append(precision)
            F1.append(f1)
            DCF.append(dcf)
            print('[{}]: AUC: {:.2f}, F1-score: {:.2f}, DCF: {:.2f}, acc: {:.2f}, precision: {:.2f}'.format(
                feature, auc * 100, f1 * 100, dcf * 100, acc * 100, precision * 100))
            gc.collect()
            torch.cuda.empty_cache()

    for i in range(len(features)):
        print('[{}]: AUC: {:.2f}, F1-score: {:.2f}, DCF: {:.2f}, acc: {:.2f}, precision: {:.2f}'.format(
            features[i], aucs[i]*100, F1[i]*100, DCF[i]*100, ACC[i]*100, PRECISION[i]*100))

    print('Global AUC: {:.2f}, F1-score: {:.2f}, DCF: {:.2f}, acc: {:.2f}, precision: {:.2f}'.format(
            np.mean(aucs)*100, (2*TP / (2*TP+FN+FP))*100, ((0.75*FN+0.25*FP) / (TN+TP+FN+FP))*100,
        (TP+TN)/(TN+TP+FN+FP)*100, TP/(TP+FP)*100))

if __name__ == '__main__':
    main()