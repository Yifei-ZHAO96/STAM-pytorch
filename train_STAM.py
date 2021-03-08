import torch
import numpy as np
import torch.nn as nn
import os, gc
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
from VAD_module import VAD, Scheduler, add_loss
from utils import SpeechDataset, HParams, prediction, train_valid_split, get_parameter_number, ValueWindow
from torch.utils.data import DataLoader
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm


train_on_gpu=torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')
_format = '%Y-%m-%d %H:%M:%S.%f'


def train_epoch(model, train_loader, loss_fn, optimizer, scheduler, batch_size, epoch, start_stpe):
    model.train()
    count = 0
    total_loss = 0
    n = batch_size
    step = start_stpe
    examples = []
    total_loss_window = ValueWindow(100)
    post_loss_window = ValueWindow(100)
    post_acc_window = ValueWindow(100)

    for x, y in train_loader:
        count += 1
        examples.append([x[0], y[0]])

        if count % 8 == 0:
            examples.sort(key=lambda x: len(x[-1]))
            examples = (np.vstack([ex[0] for ex in examples]), np.vstack([ex[1] for ex in examples]))
            batches = [(examples[0][i: i + n], examples[1][i: i + n]) for i in range(0, len(examples[-1]) + 1 - n, n)]

            if len(examples[-1]) % n != 0:
                batches.append((np.vstack((examples[0][-(len(examples[-1]) % n):],
                                          examples[0][:n-(len(examples[0]) % n)])),
                                np.vstack((examples[1][-(len(examples[-1]) % n):],
                                          examples[1][:n-(len(examples[-1]) % n)]))))

            for batch in batches:  # mini batch
                # train_data(?, 7, 80), train_label(?, 7)
                step += 1
                train_data = torch.as_tensor(batch[0], dtype=torch.float32).to(DEVICE)
                train_label = torch.as_tensor(batch[1], dtype=torch.float32).to(DEVICE)

                optimizer.zero_grad(True)
                midnet_output, postnet_output, alpha = model(train_data)
                postnet_accuracy, pipenet_accuracy = prediction(train_label, midnet_output, postnet_output)
                loss, postnet_loss, pipenet_loss, attention_loss = loss_fn(model, train_label, postnet_output, midnet_output, alpha)
                total_loss += loss.detach().item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
                optimizer.step()
                scheduler.step()
                lr = scheduler._rate

                total_loss_window.append(loss.detach().item())
                post_loss_window.append(postnet_loss.detach().item())
                post_acc_window.append(postnet_accuracy)
                if step % 10 == 0:
                    print('{}  Epoch: {}, Step: {}, overall loss: {:.5f}, postnet loss: {:.5f}, ' 
                          'postnet acc: {:.4f}, lr :{:.5f}'.format(
                            datetime.now().strftime(_format)[:-3], epoch, step, total_loss_window.average, post_loss_window.average, post_acc_window.average, lr))
                if step % 50_000 == 0:
                    print('{} save checkpoint.'.format(datetime.now().strftime(_format)[:-3]))
                    checkpoint = {
                        "model": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,
                        'step': step,
                        'scheduler_lr': scheduler._rate,
                        'scheduler_step': scheduler._step
                    }
                    if not os.path.isdir("./checkpoint"):
                        os.mkdir("./checkpoint")
                    torch.save(checkpoint, './checkpoint/STAM_weights_%s_%s.pth' % (str(epoch), str(step / 1_000_000)))
                    gc.collect()
                    torch.cuda.empty_cache()
            del batches, examples
            examples = []

    gc.collect()
    torch.cuda.empty_cache()
    return total_loss/(step-start_stpe), step


def test_epoch(model, test_loader, loss_fn):
    print('{}  Validation Begins...'.format(datetime.now().strftime(_format)[:-3]))
    model.eval()
    counter = 0
    total_loss = 0
    post_acc = 0
    pipe_acc = 0

    for train_data, train_label in tqdm(test_loader):
        # train_data(?, 7, 80), train_label(?, 7)
        train_data = train_data[0]
        train_label = train_label[0]
        train_data = torch.as_tensor(train_data, dtype=torch.float32).to(DEVICE)
        train_label = torch.as_tensor(train_label, dtype=torch.float32).to(DEVICE)
        counter += 1
        midnet_output, postnet_output, alpha = model(train_data)
        postnet_accuracy, pipenet_accuracy = prediction(train_label, midnet_output, postnet_output)
        post_acc += float(postnet_accuracy)
        pipe_acc += float(pipenet_accuracy)
        loss, _, _, _ = loss_fn(model, train_label, postnet_output, midnet_output, alpha)
        total_loss += loss.detach().item()

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    print('{}  Validation Finished...'.format(datetime.now().strftime(_format)[:-3]))
    print('Loss: {:.5f}, post acc: {:.4f}, pipe acc: {:.4f}'.format(
            total_loss/counter, post_acc/counter, pipe_acc/counter))
    return total_loss/counter, post_acc/counter


def train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, batch_size, start_epoch=0,
          epochs=20, start_step=0):
    train_losses = []
    test_losses = []
    for e in range(start_epoch, epochs):
        train_loss, start_step = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, batch_size, e, start_step)
        train_losses.append(train_loss)
        with torch.no_grad():
            test_loss, test_acc = test_epoch(model, test_loader, loss_fn)

        test_losses.append(test_loss)
        # clear cache
        torch.cuda.empty_cache()
        gc.collect()
        print("{}  Epoch: {}/{}...".format(datetime.now().strftime(_format)[:-3], e+1, epochs),
                      "Train Loss: {:.6f}...".format(train_loss),
                      "Test Loss: {:.6f}".format(test_loss))

        checkpoint = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": e+1,
            'step': start_step,
            'scheduler_lr': scheduler._rate,
            'scheduler_step': scheduler._step
        }
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(checkpoint, './checkpoint/weights_{}_acc_{:.2f}.pth'.format(e+1, test_acc*100))
    return train_losses, test_losses


if __name__ == '__main__':
    RESUME = True
    hparams = HParams()
    metadata_filename = './training_data/train.txt'
    metadata, training_idx, validation_idx = train_valid_split(metadata_filename, hparams, test_size=0.05, seed=0)

    train_dataset = SpeechDataset(metadata_filename, list(np.array(metadata)[training_idx]), hparams)
    test_dataset = SpeechDataset(metadata_filename, list(np.array(metadata)[validation_idx]), hparams)
    train_loader = DataLoader(train_dataset, 1, True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, 1, True, num_workers=4, pin_memory=True)

    gc.collect()
    torch.cuda.empty_cache()
    model = VAD(hparams).to(DEVICE)
    get_parameter_number(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = Scheduler(optimizer, init_lr=1e-3, final_lr=1e-5, decay_rate=hparams.vad_decay_rate, start_decay=hparams.vad_start_decay, decay_steps=hparams.vad_decay_steps)
    loss_fn = add_loss

    if not RESUME:
        print('{}  New Training...'.format(datetime.now().strftime(_format)[:-3]))
        train_losses, test_losses = train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, hparams.batch_size, epochs=20)
    else:
        print('{}  Resume Training...'.format(datetime.now().strftime(_format)[:-3]))
        path_checkpoint = "./checkpoint/STAM_weights_4_1.0.pth"
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
        scheduler._rate = checkpoint['scheduler_lr']
        scheduler._step = checkpoint['scheduler_step']
        train_losses, test_losses = train(model, train_loader, test_loader, loss_fn, optimizer, scheduler,
                                          batch_size=hparams.batch_size, start_epoch=start_epoch, epochs=20, start_step=step)

    f = plt.figure()
    plt.grid()
    plt.plot(test_losses, label='valid')
    plt.plot(train_losses, label='train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    plt.savefig('loss.png')
    plt.show()
    print("Min test loss: {:.6f}, min train loss: {:.6f}".format(min(test_losses), min(train_losses)))