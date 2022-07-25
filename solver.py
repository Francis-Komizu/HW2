import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from utils import myDataset
from model import Encoder
from model import Decoder
from model import SpeakerClassifier
from model import WeakSpeakerClassifier
from model import PatchDiscriminator
from model import CBHG
import os
from utils import Hps
from utils import Logger
from utils import DataLoader
from utils import to_var
from utils import reset_grad
from utils import multiply_grad
from utils import grad_clip
from utils import cal_acc
from utils import cc
from utils import calculate_gradients_penalty
from utils import gen_noise

class Solver(object):
    def __init__(self, hps, data_loader, log_dir='./log/'):
        self.hps = hps
        self.data_loader = data_loader
        self.model_kept = []
        self.max_keep = 100
        self.build_model()
        self.logger = Logger(log_dir) if log_dir != None else None

    def build_model(self):
        hps = self.hps
        ns = self.hps.ns
        emb_size = self.hps.emb_size
        self.Encoder = cc(Encoder(ns=ns, dp=hps.enc_dp))
        self.Decoder = cc(Decoder(ns=ns, c_a=hps.n_speakers, emb_size=emb_size))
        betas = (0.5, 0.9)
        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        self.ae_opt = optim.Adam(params, lr=self.hps.lr, betas=betas)

    def save_model(self, model_path, iteration):
        all_model = {
            'encoder': self.Encoder.state_dict(),
            'decoder': self.Decoder.state_dict(),
        }
        new_model_path = '{}-{}'.format(model_path, iteration)
        with open(new_model_path, 'wb') as f_out:
            torch.save(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def load_model(self, model_path, enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            self.Decoder.load_state_dict(all_model['decoder'])

    def set_eval(self):
        self.Encoder.eval()
        self.Decoder.eval()

    def test_step(self, x, c1, c2=None):
        self.set_eval()
        x = to_var(x).permute(0, 2, 1)
        enc = self.Encoder(x)
        x_tilde = self.Decoder(enc, c1, c2)
        return x_tilde.data.cpu().numpy()

    def permute_data(self, data):
        C = to_var(data[0], requires_grad=False)
        X = to_var(data[1]).permute(0, 2, 1)
        return C, X

    def sample_c(self, size):
        n_speakers = self.hps.n_speakers
        c_sample = Variable(
                torch.multinomial(torch.ones(n_speakers), num_samples=size, replacement=True),  
                requires_grad=False)
        c_sample = c_sample.cuda() if torch.cuda.is_available() else c_sample
        return c_sample

    def encode_step(self, x):
        enc = self.Encoder(x)
        return enc

    def decode_step(self, enc, c1, c2=None):
        x_tilde = self.Decoder(enc, c1, c2)
        return x_tilde

    def cal_loss(self, logits, y_true):
        # calculate loss 
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_true)
        return loss

    def train(self, model_path, flag='train'):
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
            os.chmod(model_path, 0o755)
        model_path = os.path.join(model_path, 'model.pkl')
        # load hyperparams
        hps = self.hps
        for iteration in range(hps.enc_pretrain_iters):
                data = next(self.data_loader)
                c, x = self.permute_data(data)
                # encode
                enc = self.encode_step(x)
                x_tilde = self.decode_step(enc, c)
                loss_rec = torch.mean(torch.abs(x_tilde - x))
                reset_grad([self.Encoder, self.Decoder])
                loss_rec.backward()
                grad_clip([self.Encoder, self.Decoder], self.hps.max_grad_norm)
                self.ae_opt.step()
                # tb info
                info = {
                    f'{flag}/ae_loss_rec': loss_rec.item(),
                }
                slot_value = (iteration + 1, hps.enc_pretrain_iters) + tuple([value for value in info.values()])
                log = 'tr_ae:[%06d/%06d], loss_rec=%.3f'
                print(log % slot_value)
                if iteration % 100 == 0:
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, iteration + 1)


