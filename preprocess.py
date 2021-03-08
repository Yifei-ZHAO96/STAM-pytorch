import argparse
import os
from multiprocessing import cpu_count
from utils import HParams
from datasets import preprocessor
from tqdm import tqdm


def preprocess(args, input_folders, out_dir, hparams):
	mel_dir = os.path.join(out_dir, 'mels')
	os.makedirs(mel_dir, exist_ok=True)
	metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(metadata, out_dir, hparams)


def write_metadata(metadata, out_dir, hparams):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	timesteps = sum([int(m[2]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} audio timesteps, ({:.2f} hours)'.format(len(metadata), timesteps, hours))
	print('Max audio timesteps length: {:.2f} secs'.format((max(m[2] for m in metadata)) / sr, ))



def run_preprocess(args, hparams):
	input_folders = ['E:\Dataset\TIMIT\TRAIN_clean', 'E:\Dataset\TIMIT\TRAIN_noisy']
	print('input_folders:',input_folders)
	output_folder = os.path.join(args.base_dir, args.output)
	preprocess(args, input_folders, output_folder, hparams)


def main():
	print('initializing preprocessing..')
	hparams = HParams()
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
						help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--dataset', default='TIMIT')
	parser.add_argument('--output', default='testing_TIMIT-noisy')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	run_preprocess(args, hparams)


if __name__ == '__main__':
	main()
