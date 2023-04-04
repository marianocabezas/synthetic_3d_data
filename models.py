import time
import torch
from torch import nn
import torch.nn.functional as F
from base import BaseModel, ResConv3dBlock
from base import Autoencoder
from utils import time_to_string
from criteria import tp_binary_loss, tn_binary_loss, dsc_binary_loss, accuracy


def norm_f(n_f):
    return nn.GroupNorm(n_f // 4, n_f)


def print_batch(pi, n_patches, i, n_cases, t_in, t_case_in):
    init_c = '\033[38;5;238m'
    percent = 25 * (pi + 1) // n_patches
    progress_s = ''.join(['█'] * percent)
    remainder_s = ''.join([' '] * (25 - percent))

    t_out = time.time() - t_in
    t_case_out = time.time() - t_case_in
    time_s = time_to_string(t_out)

    t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
    eta_s = time_to_string(t_eta)
    pre_s = '{:}Case {:03d}/{:03d} ({:03d}/{:03d} - {:06.2f}%) [{:}{:}]' \
            ' {:} ETA: {:}'
    batch_s = pre_s.format(
        init_c, i + 1, n_cases, pi + 1, n_patches, 100 * (pi + 1) / n_patches,
        progress_s, remainder_s, time_s, eta_s + '\033[0m'
    )
    print('\033[K', end='', flush=True)
    print(batch_s, end='\r', flush=True)


class SimpleResNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        self.init = True
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.extractor = Autoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.extractor.to(device)
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_filters[-1], self.conv_filters[-1] // 2),
            nn.ReLU(),
            norm_f(self.conv_filters[-1] // 2),
            # nn.Linear(self.conv_filters[-1] // 2, self.conv_filters[-1] // 4),
            # nn.ReLU(),
            # norm_f(self.conv_filters[-1] // 4),
            # nn.Linear(self.conv_filters[-1] // 4, 1)
            nn.Linear(self.conv_filters[-1] // 2, 1)
        )
        self.classifier.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': F.binary_cross_entropy
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 0,
                'f': F.binary_cross_entropy
            },
            {
                'name': 'fn',
                'weight': 0.5,
                'f': tp_binary_loss
            },
            {
                'name': 'fp',
                'weight': 0.5,
                'f': tn_binary_loss
            },
            {
                'name': 'acc',
                'weight': 0,
                'f': lambda p, t: 1 - accuracy(
                    (p > 0.5).type_as(p), t.type_as(p)
                )
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=1e-4)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=1e-4)

    def forward(self, data):
        _, features = self.extractor.encode(data)
        # final_features = torch.mean(features.flatten(2), dim=2)
        final_features = torch.max(features.flatten(2), dim=2)[0]
        logits = self.classifier(final_features)
        return torch.sigmoid(logits).flatten()
