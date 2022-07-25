import os
import sys
import h5py
import torch
import numpy as np
from utils import Hps
from solver import Solver
from scipy.io import wavfile
from torch.autograd import Variable
from preprocess.tacotron.norm_utils import spectrogram2wav

def sp2wav(sp): 
	exp_sp = sp
	wav_data = spectrogram2wav(exp_sp)
	return wav_data

def convert_sp(sp, solver, c1, c2=None):
	c1_var = Variable(torch.from_numpy(np.array([c1]))).cuda()
	if c2 is not None:
		c2_var = Variable(torch.from_numpy(np.array([c2]))).cuda()
	else:
		c2_var = None
	sp_tensor = torch.from_numpy(np.expand_dims(sp, axis = 0))
	sp_tensor = sp_tensor.type(torch.FloatTensor)
	converted_sp = solver.test_step(sp_tensor, c1_var, c2_var)
	converted_sp = converted_sp.squeeze(axis = 0).transpose((1, 0))
	return converted_sp

def get_model(hps_path, model_path):
	hps = Hps()
	hps.load(hps_path)
	hps_tuple = hps.get_tuple()
	solver = Solver(hps_tuple, None, None)
	solver.load_model(model_path)
	return solver

def convert_one_sp(h5_path, src_speaker, tar_speaker, utt_id, solver, dir_path,
				   dset='test', tar_speaker_2=None,
				   speaker_used_path='./hps/en_speaker_used.txt'):
	# read speaker id file
	with open(speaker_used_path) as f:
		speakers = [line.strip() for line in f]
		speaker2id = {speaker: i for i, speaker in enumerate(speakers)}

	with h5py.File(h5_path, 'r') as f_h5:
		sp = f_h5[f'{dset}/{src_speaker}/{utt_id}/lin'][()]
		c1 = speaker2id[tar_speaker]
		c2 = speaker2id[tar_speaker_2] if tar_speaker_2 else None
		converted_sp = convert_sp(sp, solver, c1, c2)
		wav_data = sp2wav(converted_sp)

		if tar_speaker_2 is None:
			fn = f'{src_speaker}_{tar_speaker}_{utt_id}.wav'
		else:
			fn = f'{src_speaker}_{tar_speaker}_{utt_id}_inter.wav'
		wav_path = os.path.join(dir_path, fn)
		wavfile.write(wav_path, 16000, wav_data)

# # do max_n times voice conversion
# def convert_all_sp(h5_path, src_speaker, tar_speaker, solver, dir_path,
# 					dset = 'test', max_n = 5,
# 					speaker_used_path = './hps/en_speaker_used.txt'):
# 	# read speaker id file
# 	with open(speaker_used_path) as f:
# 		speakers = [line.strip() for line in f]
# 		speaker2id = {speaker:i for i, speaker in enumerate(speakers)}
#
# 	with h5py.File(h5_path, 'r') as f_h5:
# 		c = 0 # counter
# 		for utt_id in f_h5[f'{dset}/{src_speaker}']:
# 			sp = f_h5[f'{dset}/{src_speaker}/{utt_id}/lin'][()]
# 			c1 = speaker2id[tar_speaker]
# 			converted_sp = convert_sp(sp, solver, c1)
# 			wav_data = sp2wav(converted_sp)
# 			wav_path = os.path.join(dir_path, f'{src_speaker}_{tar_speaker}_{utt_id}.wav')
# 			wavfile.write(wav_path, 16000, wav_data)
# 			c += 1
# 			if c >= max_n:
# 				break

if __name__ == '__main__':
	h5_path = './vctk.h5'
	root_dir = './results'
	model_path = './pkl/model.pkl'
	hps_path = './hps/vctk.json'
	solver = get_model(hps_path = hps_path, model_path = model_path)
	speakers = ['1', '2']
	# max_n = 5
	if len(sys.argv) == 3:
		if sys.argv[1] == '1to2':
			speaker_A = speakers[0]
			speaker_B = speakers[1]
			tar_speaker_2 = None
		elif sys.argv[1] == '2to1':
			speaker_A = speakers[1]
			speaker_B = speakers[0]
			tar_speaker_2 = None
		elif sys.argv[1] == '1and2':
			speakers_A = speakers[0]
			speakers_A = speakers[1]
			tar_speaker_2 = speakers[0]
		elif sys.argv[1] == '2and1':
			speaker_A = speakers[1]
			speaker_B = speakers[0]
			tar_speaker_2 = speakers[1]
		else:
			print("usage:"
				  "1. convert.py 1to2"
				  "2. convert.py 2to1"
				  "3. convert.py 1and2"
				  "4. convert.py 2and1")
			exit(0)
		utt_id = sys.argv[2]
		dset = 'test'
		speaker_used_path = './hps/en_speaker_used.txt'
		dir_path = os.path.join(root_dir, f'p{speaker_A}_p{speaker_B}')
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)
		convert_one_sp(h5_path, speaker_A, speaker_B, utt_id,
						solver, dir_path, dset, tar_speaker_2,
					   speaker_used_path)

