import time
import itertools
from functools import partial
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import time_to_string


class BaseModel(nn.Module):
    """"
    This is the baseline model to be used for any of my networks. The idea
    of this model is to create a basic framework that works similarly to
    keras, but flexible enough.
    For that reason, I have "embedded" the typical pytorch main loop into a
    fit function and I have defined some intermediate functions and callbacks
    to alter the main loop. By itself, this model can train any "normal"
    network with different losses and scores for training and validation.
    It can be easily extended to create adversarial networks (which I have done
    in other repositories) and probably to other more complex problems.
    The network also includes some print functions to check the current status.
    """
    def __init__(self):
        """
        Main init. By default some parameters are set, but they should be
        redefined on networks inheriting that model.
        """
        super().__init__()
        # Init values
        self.device = None
        self.init = True
        self.optimizer_alg = None
        self.current_task = -1
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.dropout = 0
        self.final_dropout = 0
        self.ann_rate = 0
        self.best_loss_tr = np.inf
        self.best_loss_val = np.inf
        self.last_state = None
        self.best_state = None
        self.best_opt = None
        self.train_functions = [
            {'name': 'train', 'weight': 1, 'f': None},
        ]
        self.train_log = {}
        self.val_functions = [
            {'name': 'val', 'weight': 1, 'f': None},
        ]
        self.val_log = {}
        self.acc_functions = {}
        self.acc = None

    def update_logs(self):
        self.train_log = {
            f['name']: []
            for f in self.train_functions
        }
        self.val_log = {
            f['name']: []
            for f in self.val_functions
        }

    def forward(self, *inputs):
        """

        :param inputs: Inputs to the forward function. We are passing the
         contents by reference, so if there are more than one input, they
         will be separated.
        :return: Nothing. This has to be reimplemented for any class.
        """
        return None

    def observe(self, x, y):
        # First, we do a forward pass through the network.
        if isinstance(x, list) or isinstance(x, tuple):
            x_cuda = tuple(x_i.to(self.device) for x_i in x)
            pred_labels = self(*x_cuda)
        else:
            x_cuda = x.to(self.device)
            pred_labels = self(x_cuda)
        if isinstance(y, list) or isinstance(y, tuple):
            y_cuda = tuple(y_i.to(self.device) for y_i in y)
        else:
            y_cuda = y.to(self.device)

        return pred_labels, x_cuda, y_cuda

    def mini_batch_loop(
            self, data, train=True
    ):
        """
        This is the main loop. It's "generic" enough to account for multiple
        types of data (target and input) and it differentiates between
        training and testing. While inherently all networks have a training
        state to check, here the difference is applied to the kind of data
        being used (is it the validation data or the training data?). Why am
        I doing this? Because there might be different metrics for each type
        of data. There is also the fact that for training, I really don't care
        about the values of the losses, since I only want to see how the global
        value updates, while I want both (the losses and the global one) for
        validation.
        :param data: Dataloader for the network.
        :param train: Whether to use the training dataloader or the validation
         one.
        :return:
        """
        losses = list()
        mid_losses = list()
        accs = list()
        n_batches = len(data)
        for batch_i, (x, y) in enumerate(data):
            # In case we are training the the gradient to zero.
            if self.training:
                self.optimizer_alg.zero_grad()

            # First, we do a forward pass through the network.
            pred_labels, x_cuda, y_cuda = self.observe(x, y)

            # After that, we can compute the relevant losses.
            if train:
                # Training losses (applied to the training data)
                batch_losses = [
                    l_f['weight'] * l_f['f'](pred_labels, y_cuda)
                    for l_f in self.train_functions
                ]
                batch_loss = sum(batch_losses)
                if self.training:
                    for l_f, v in zip(self.train_functions, batch_losses):
                        if isinstance(v, torch.Tensor):
                            self.train_log[l_f['name']].append(
                                v.detach().cpu().numpy().tolist()
                            )
                        else:
                            self.train_log[l_f['name']].append(v)
                    try:
                        batch_loss.backward()
                        self.prebatch_update(batch_i, len(data), x_cuda, y_cuda)
                        self.optimizer_alg.step()
                        self.batch_update(batch_i, len(data), x_cuda, y_cuda)
                    except RuntimeError:
                        self.prebatch_update(batch_i, len(data), x_cuda, y_cuda)

            else:
                # Validation losses (applied to the validation data)
                batch_losses = [
                    l_f['f'](pred_labels, y_cuda)
                    for l_f in self.val_functions
                ]
                for l_f, v in zip(self.val_functions, batch_losses):
                    if isinstance(v, torch.Tensor):
                        self.val_log[l_f['name']].append(
                            v.detach().cpu().numpy().tolist()
                        )
                    else:
                        self.val_log[l_f['name']].append(v)
                batch_loss = sum([
                    l_f['weight'] * l
                    for l_f, l in zip(self.val_functions, batch_losses)
                ])
                mid_losses.append([l.tolist() for l in batch_losses])
                batch_accs = [
                    l_f['f'](pred_labels, y_cuda)
                    for l_f in self.acc_functions
                ]
                accs.append([a.tolist() for a in batch_accs])

            # It's important to compute the global loss in both cases.
            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            # Curriculum dropout / Adaptive dropout
            # Here we could modify dropout to be updated for each batch.
            # (1 - rho) * exp(- gamma * t) + rho, gamma > 0

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses)
            )
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Mean loss of the global loss (we don't need the loss for each batch).
        mean_loss = np.mean(losses)

        if train:
            return mean_loss
        else:
            # If using the validation data, we actually need to compute the
            # mean of each different loss.
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            np_accs = np.array(list(zip(*accs)))
            mean_accs = np.mean(np_accs, axis=1) if np_accs.size > 0 else []
            return mean_loss, mean_losses, mean_accs

    def fit(
            self,
            train_loader,
            val_loader,
            epochs=50,
            patience=5,
            verbose=True
    ):
        # Init
        best_e = 0
        no_improv_e = 0
        l_names = ['train', ' val '] + [
            '{:^6s}'.format(l_f['name']) for l_f in self.val_functions
        ]
        acc_names = [
            '{:^6s}'.format(a_f['name']) for a_f in self.acc_functions
        ]
        l_bars = '--|--'.join(
            ['-' * 5] * 2 +
            ['-' * 6] * (len(l_names[2:]) + len(acc_names)) +
            ['-' * 3]
        )
        l_hdr = '  |  '.join(l_names + acc_names + ['drp'])
        # Since we haven't trained the network yet, we'll assume that the
        # initial values are the best ones.
        self.best_state = deepcopy(self.state_dict())
        self.best_opt = deepcopy(self.optimizer_alg.state_dict())
        t_start = time.time()

        # Initial losses
        # This might seem like an unnecessary step (and it actually often is)
        # since it wastes some time checking the output with the initial
        # weights. However, it's good to check that the network doesn't get
        # worse than a random one (which can happen sometimes).
        if self.init:
            # We are looking for the output, without training, so no need to
            # use grad.
            with torch.no_grad():
                self.t_val = time.time()
                # We set the network to eval, for the same reason.
                self.eval()
                # Training losses.
                self.best_loss_tr = self.mini_batch_loop(train_loader)
                # Validation losses.
                self.best_loss_val, best_loss, best_acc = self.mini_batch_loop(
                    val_loader, False
                )
                # Doing this also helps setting an initial best loss for all
                # the necessary losses.
                if verbose:
                    # This is just the print for each epoch, but including the
                    # header.
                    # Mid losses check
                    epoch_s = '\033[32mInit     \033[0m'
                    tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(
                        self.best_loss_tr
                    )
                    loss_s = '\033[32m{:7.4f}\033[0m'.format(
                        self.best_loss_val
                    )
                    losses_s = [
                        '\033[36m{:8.4f}\033[0m'.format(l) for l in best_loss
                    ]
                    # Acc check
                    acc_s = [
                        '\033[36m{:8.4f}\033[0m'.format(a) for a in best_acc
                    ]
                    t_out = time.time() - self.t_val
                    t_s = time_to_string(t_out)

                    drop_s = '{:5.3f}'.format(self.dropout)

                    print('\033[K', end='')
                    whites = ' '.join([''] * 12)
                    print('{:}Epoch num |  {:}  |'.format(whites, l_hdr))
                    print('{:}----------|--{:}--|'.format(whites, l_bars))
                    final_s = whites + ' | '.join(
                        [epoch_s, tr_loss_s, loss_s] +
                        losses_s + acc_s + [drop_s, t_s]
                    )
                    print(final_s)
        else:
            # If we don't initialise the losses, we'll just take the maximum
            # ones (inf, -inf) and print just the header.
            print('\033[K', end='')
            whites = ' '.join([''] * 12)
            print('{:}Epoch num |  {:}  |'.format(whites, l_hdr))
            print('{:}----------|--{:}--|'.format(whites, l_bars))
            best_loss = [np.inf] * len(self.val_functions)
            best_acc = [-np.inf] * len(self.acc_functions)

        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            self.train()
            # First we train and check if there has been an improvement.
            loss_tr = self.mini_batch_loop(train_loader)
            improvement_tr = self.best_loss_tr > loss_tr
            if improvement_tr:
                self.best_loss_tr = loss_tr
                tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(loss_tr)
            else:
                tr_loss_s = '{:7.4f}'.format(loss_tr)

            # Then we validate and check all the losses
            with torch.no_grad():
                self.t_val = time.time()
                self.eval()
                loss_val, mid_losses, acc = self.mini_batch_loop(
                    val_loader, False
                )

            # Mid losses check
            losses_s = [
                '\033[36m{:8.4f}\033[0m'.format(l) if bl > l
                else '{:8.4f}'.format(l) for bl, l in zip(
                    best_loss, mid_losses
                )
            ]
            best_loss = [
                l if bl > l else bl for bl, l in zip(
                    best_loss, mid_losses
                )
            ]
            # Acc check
            acc_s = [
                '\033[36m{:8.4f}\033[0m'.format(a) if ba < a
                else '{:8.4f}'.format(a) for ba, a in zip(
                    best_acc, acc
                )
            ]
            best_acc = [
                a if ba < a else ba for ba, a in zip(
                    best_acc, acc
                )
            ]

            # Patience check
            # We check the patience to stop early if the network is not
            # improving. Otherwise we are wasting resources and time.
            improvement_val = self.best_loss_val > loss_val
            loss_s = '{:7.4f}'.format(loss_val)
            if improvement_val:
                self.best_loss_val = loss_val
                epoch_s = '\033[32mEpoch {:03d}\033[0m'.format(self.epoch)
                loss_s = '\033[32m{:}\033[0m'.format(loss_s)
                best_e = self.epoch
                self.best_state = deepcopy(self.state_dict())
                self.best_opt = deepcopy(self.optimizer_alg.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch {:03d}'.format(self.epoch)
                no_improv_e += 1

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            drop_s = '{:5.3f}'.format(self.dropout)
            self.dropout_update()

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                final_s = whites + ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] +
                    losses_s + acc_s + [drop_s, t_s]
                )
                print(final_s)

            self.epoch_update(epochs, train_loader)

            if no_improv_e == int(patience / (1 - self.dropout)):
                break

        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                    'Training finished in {:} epochs ({:}) '
                    'with minimum loss = {:f} (epoch {:d})'.format(
                        self.epoch + 1, t_end_s, self.best_loss_val, best_e
                    )
            )

        self.last_state = deepcopy(self.state_dict())
        self.epoch = best_e
        self.load_state_dict(self.best_state)

    def inference(self, data, nonbatched=True):
        with torch.no_grad():
            if isinstance(data, list) or isinstance(data, tuple):
                x_cuda = tuple(
                    torch.from_numpy(x_i).to(self.device)
                    for x_i in data
                )
                if nonbatched:
                    x_cuda = tuple(
                        x_i.unsqueeze(0) for x_i in x_cuda
                    )

                output = self(*x_cuda)
            else:
                x_cuda = torch.from_numpy(data).to(self.device)
                if nonbatched:
                    x_cuda = x_cuda.unsqueeze(0)
                output = self(x_cuda)
            torch.cuda.empty_cache()

            if nonbatched:
                np_output = output[0, 0].cpu().numpy()
            else:
                np_output = output.cpu().numpy()

        return np_output

    def reset_optimiser(self, model_params=None):
        """
        Abstract function to rest the optimizer.
        :return: Nothing.
        """
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.best_loss_tr = np.inf
        self.best_loss_val = np.inf
        self.best_state = None
        self.best_opt = None
        return None

    def epoch_update(self, epochs, loader):
        """
        Callback function to update something on the model after the epoch
        is finished. To be reimplemented if necessary.
        :param epochs: Maximum number of epochs
        :param loader: Dataloader used for training
        :return: Nothing.
        """
        return None

    def prebatch_update(self, batch, batches, x, y):
        """
        Callback function to update something on the model before the batch
        update is applied. To be reimplemented if necessary.
        :param batches: Maximum number of epochs
        :param x: Training data
        :param y: Training target
        :return: Nothing.
        """
        return None

    def batch_update(self, batch, batches, x, y):
        """
        Callback function to update something on the model after the batch
        is finished. To be reimplemented if necessary.
        :param batches: Maximum number of epochs
        :param x: Training data
        :param y: Training target
        :return: Nothing.
        """
        return None

    def dropout_update(self):
        """
        Callback function to update the dropout. To be reimplemented
        if necessary. However, the main method already has some basic
        scheduling
        :param epochs: Maximum number of epochs
        :return: Nothing.
        """
        if self.final_dropout <= self.dropout:
            self.dropout = max(
                self.final_dropout, self.dropout - self.ann_rate
            )

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss):
        """
        Function to print the progress of a batch. It takes into account
        whether we are training or validating and uses different colors to
        show that. It's based on Keras arrow progress bar, but it only shows
        the current (and current mean) training loss, elapsed time and ETA.
        :param batch_i: Current batch number.
        :param n_batches: Total number of batches.
        :param b_loss: Current loss.
        :param mean_loss: Current mean loss.
        :return: None.
        """
        init_c = '\033[0m' if self.training else '\033[38;5;238m'
        percent = 25 * (batch_i + 1) // n_batches
        progress_s = ''.join(['█'] * percent)
        remainder_s = ''.join([' '] * (25 - percent))
        loss_name = 'train_loss' if self.training else 'val_loss'

        if self.training:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)
        epoch_hdr = '{:}Epoch {:03} ({:03d}/{:03d} - {:05.2f}%) [{:}] '
        loss_s = '{:} {:f} ({:f}) {:} / ETA {:}'
        batch_s = (epoch_hdr + loss_s).format(
            init_c, self.epoch, batch_i + 1, n_batches,
            100 * (batch_i + 1) / n_batches, progress_s + remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s + '\033[0m'
        )
        print(' '.join([' '] * 300), end='\r')
        print('\033[K' + batch_s, end='\r', flush=True)

    @staticmethod
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
            init_c, i + 1, n_cases, pi + 1, n_patches,
            100 * (pi + 1) / n_patches,
            progress_s, remainder_s, time_s, eta_s + '\033[0m'
        )
        print(' '.join([' '] * 300), end='\r')
        print('\033[K' + batch_s, end='\r', flush=True)

    def freeze(self):
        """
        Method to freeze all the network parameters.
        :return: None
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Method to unfreeze all the network parameters.
        :return: None
        """
        for param in self.parameters():
            param.requires_grad = True

    def set_last_state(self):
        if self.last_state is not None:
            self.load_state_dict(self.last_state)

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(
            torch.load(net_name, map_location=self.device)
        )


class Autoencoder(BaseModel):
    """
    Main autoencoder class. This class can actually be parameterised on init
    to have different "main blocks", normalisation layers and activation
    functions.
    """
    def __init__(
            self,
            conv_filters,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            n_inputs=1,
            kernel=3,
            norm=None,
            activation=None,
            block=None,
    ):
        """
        Constructor of the class. It's heavily parameterisable to allow for
        different autoencoder setups (residual blocks, double convolutions,
        different normalisation and activations).
        :param conv_filters: Filters for both the encoder and decoder. The
         decoder mirrors the filters of the encoder.
        :param device: Device where the model is stored (default is the first
         cuda device).
        :param n_inputs: Number of input channels.
        :param kernel: Kernel width for the main block.
        :param norm: Normalisation block (it has to be a pointer to a valid
         normalisation Module).
        :param activation: Activation block (it has to be a pointer to a valid
         activation Module).
        :param block: Main block. It has to be a pointer to a valid block from
         this python file (otherwise it will fail when trying to create a
         partial of it).
        :param dropout: Dropout value.
        """
        super().__init__()
        # Init
        if norm is None:
            norm = partial(lambda ch_in: nn.Sequential())
        if block is None:
            block = ResConv3dBlock
        block_partial = partial(
            block, kernel=kernel, norm=norm, activation=activation
        )
        self.device = device
        self.filters = conv_filters

        conv_in, conv_out, deconv_in, deconv_out = block.compute_filters(
            n_inputs, conv_filters
        )

        # Down path
        # We'll use the partial and fill it with the channels for input and
        # output for each level.
        self.down = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                conv_in, conv_out
            )
        ])

        # Bottleneck
        self.u = block_partial(conv_filters[-2], conv_filters[-1])

        # Up path
        # Now we'll do the same we did on the down path, but mirrored. We also
        # need to account for the skip connections, that's why we sum the
        # channels for both outputs. That basically means that we are
        # concatenating with the skip connection, and not suming.
        self.up = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                deconv_in, deconv_out
            )
        ])

    def encode(self, input_s):
        # We need to keep track of the convolutional outputs, for the skip
        # connections.
        down_inputs = []
        for c in self.down:
            c.to(self.device)
            input_s = c(input_s)
            down_inputs.append(input_s)
            # Remember that pooling is optional
            input_s = F.max_pool3d(input_s, 2)

        self.u.to(self.device)
        input_s = self.u(input_s)

        return down_inputs, input_s

    def decode(self, input_s, down_inputs):
        for d, i in zip(self.up, down_inputs[::-1]):
            d.to(self.device)
            # Remember that pooling is optional
            input_s = d(
                torch.cat(
                    (F.interpolate(input_s, size=i.size()[2:]), i),
                    dim=1
                )
            )

        return input_s

    def forward(self, input_s, keepfeat=False):
        down_inputs, input_s = self.encode(input_s)

        features = down_inputs + [input_s] if keepfeat else []

        input_s = self.decode(input_s, down_inputs)

        output = (input_s, features) if keepfeat else input_s

        return output


class BaseConv3dBlock(BaseModel):
    def __init__(self, filters_in, filters_out, kernel):
        super().__init__()
        self.conv = partial(
            nn.Conv3d, kernel_size=kernel, padding=kernel // 2
        )

    def forward(self, inputs, *args, **kwargs):
        return self.conv(inputs)

    @staticmethod
    def default_activation(n_filters):
        return nn.ReLU()

    @staticmethod
    def compute_filters(n_inputs, conv_filters):
        conv_in = [n_inputs] + conv_filters[:-2]
        conv_out = conv_filters[:-1]
        down_out = conv_filters[-2::-1]
        up_out = conv_filters[:0:-1]
        deconv_in = list(map(sum, zip(down_out, up_out)))
        deconv_out = down_out
        return conv_in, conv_out, deconv_in, deconv_out


class ResConv3dBlock(BaseConv3dBlock):
    def __init__(
            self, filters_in, filters_out,
            kernel=3, norm=None, activation=None
    ):
        super().__init__(filters_in, filters_out, kernel)
        if activation is None:
            activation = self.default_activation
        conv = nn.Conv3d

        self.conv = self.conv(filters_in, filters_out)

        if filters_in != filters_out:
            self.res = conv(
                filters_in, filters_out, 1,
            )
        else:
            self.res = None

        self.end_seq = nn.Sequential(
            activation(filters_out),
            norm(filters_out)
        )

    def forward(self, inputs, return_linear=False, *args, **kwargs):
        res = inputs if self.res is None else self.res(inputs)
        data = self.conv(inputs) + res
        if return_linear:
            return self.end_seq(data), data
        else:
            return self.end_seq(data)


class AttentionBlock(BaseModel):
    def __init__(
            self, filters_in, filters_out, filters_att,
            kernel=1, norm=None, activation=None
    ):
        super().__init__()
        if activation is None:
            activation = partial(lambda filters: nn.ReLU())
        conv = nn.Conv3d

        self.conv_q = conv(
            filters_in, filters_att, 1,
        )
        self.conv_k = conv(
            filters_in, filters_att, 1,
        )
        self.conv_v = conv(
            filters_in, filters_out, kernel,
        )

        self.end_seq = nn.Sequential(
            activation(filters_out),
            norm(filters_out)
        )

    def forward(self, source, target):
        query = F.instance_norm(self.conv_q(source))
        key = F.instance_norm(self.conv_k(target))
        value = self.conv_v(target - source)
        alpha = torch.abs(
            torch.mean(query * key, dim=1, keepdim=True)
        )
        features = torch.clamp(1 - alpha, 0, 1) * value
        return self.end_seq(features)


class AttentionGate3D(nn.Module):
    """
    Attention gade block based on
    Jo Schlemper, Ozann Oktay, Michiel Schaap, Mattias Heinrich, Bernhard
    Kainz, Ben Glocker, Daniel Rueckert. "Attention gated networks: Learning
    to leverage salient regions in medical images"
    https://doi.org/10.1016/j.media.2019.01.012
    """

    def __init__(
            self, x_features, g_features, int_features, sigma2=torch.sigmoid
    ):
        super().__init__()
        self.conv_g = nn.Conv3d(g_features, int_features, 1)
        self.conv_x = nn.Conv3d(x_features, int_features, 1)
        self.conv_phi = nn.Conv3d(int_features, 1, 1)
        self.sigma2 = sigma2

    def forward(self, x, g, attention=False):
        g_emb = self.conv_g(g)
        x_emb = F.interpolate(
            self.conv_x(x), size=g_emb.size()[2:]
        )
        phi_emb = self.conv_phi(F.relu(g_emb + x_emb))
        alpha = F.interpolate(
            self.sigma2(phi_emb), size=x.size()[2:]
        )

        if attention:
            return x * alpha, alpha
        else:
            return x * alpha


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim, heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Linear(embed_dim, mlp_dim)

    def forward(self, x_in, q_in=None):
        x = self.ln1(x_in)
        if q_in is not None:
            q = self.ln1(q_in)
            x = torch.cat([q, x], dim=1)
            x, _ = self.attention(
                query=q, key=x, value=x, need_weights=False
            )
        else:
            x, _ = self.attention(
                query=x, key=x, value=x, need_weights=False
            )
        x = x + x_in

        y = self.ln2(x)
        y = self.mlp(y)

        return x + y
