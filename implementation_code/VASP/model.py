import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module) :
    def __init__(self, input_dim, latent_dim, hidden_dim, num_enc=7):
        super(Encoder, self).__init__()

        ls = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim)]

        for _ in range(num_enc-1):
            ls.append(nn.Linear(hidden_dim, hidden_dim))
            ls.append(nn.LayerNorm(hidden_dim))

        self.encoder = nn.ModuleList(ls)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        prev_sum = 0
        prev_output = x

        for idx in range(0, len(self.encoder), 2):
            fc = self.encoder[idx]
            ln = self.encoder[idx+1]

            output = ln(self.swish(fc(prev_output) + prev_sum))
            prev_output = output
            prev_sum += output
        z_mean = self.fc_mu(prev_output)
        z_logvar = self.fc_logvar(prev_output)

        return z_mean, z_logvar

    @staticmethod
    def swish(x):
        return x.mul(torch.sigmoid(x))


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_dec=5):
        super(Decoder, self).__init__()

        ls = [nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim)]

        for _ in range(num_dec-1):
            ls.append(nn.Linear(hidden_dim, hidden_dim))
            ls.append(nn.LayerNorm(hidden_dim))

        self.decoder = nn.ModuleList(ls)
        self.decoder_resnet = nn.Linear(hidden_dim, input_dim)
        self.decoder_latent = nn.Linear(latent_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        prev_sum = 0
        prev_output = x

        for idx in range(0, len(self.decoder), 2):
            fc = self.decoder[idx]
            ln = self.decoder[idx+1]

            output = ln(self.swish(fc(prev_output) + prev_sum))
            prev_output = output
            prev_sum += output

            # print(output.shape)

        dr = self.sigmoid(self.decoder_resnet(prev_output))
        # print(dr.shape)
        dl = self.sigmoid(self.decoder_latent(x))
        # print(dr.shape, dl.shape)

        return dr * dl

    @staticmethod
    def swish(x):
        return x.mul(torch.sigmoid(x))


class FLVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_enc=7, num_dec=5):
        super(FLVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(input_dim, latent_dim, hidden_dim, num_enc)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim, num_dec)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)

        return self.decoder(z), z_mean, z_logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


class EASE(nn.Module):
    def __init__(self, input_dim, device):
        super(EASE, self).__init__()
        self.encoder = nn.Linear(input_dim, input_dim, bias=False)

        # constraint diagonal zero
        self.const_eye_zero = torch.ones((input_dim, input_dim), device=device)
        self.diag = torch.eye(input_dim, dtype=torch.bool, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # setting diagonal weight to zero
        self._set_diag_zero()

        output = self.encoder(x)

        return self.sigmoid(output)

    def _set_diag_zero(self):
        self.encoder.weight.data[self.diag] = 0.


class VASP(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout, device, num_enc=7, num_dec=5):
        super(VASP, self).__init__()
        self.ease = EASE(input_dim=input_dim, device=device)
        self.flvae = FLVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim,
                           num_enc=num_enc, num_dec=num_dec)

    def forward(self, x):
        ease_y = self.ease(x)
        flvae_y, mu, logvar = self.flvae(x)
        output = torch.mul(flvae_y, ease_y)

        return output, mu, logvar


def loss_function_vasp(recon_x, x, mu, logvar, alpha=0.25, gamma=2.0):
    # mll = (F.log_softmax(x_pred, dim=-1) * user_ratings)
    # F_loss = torch.pow(1 - torch.exp(mll), r) * mll
    mll = (F.log_softmax(recon_x, 1) * x)
    F_loss = alpha * torch.pow(1 - torch.exp(mll), gamma) * mll
    BCE = -F_loss.sum(dim=-1).mean()
    # BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + KLD, BCE, KLD

