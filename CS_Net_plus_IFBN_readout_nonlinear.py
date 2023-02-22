import torch
import torch.nn as nn
import numpy as np
from blocks3 import PHI, PHI_inv, reconblcok, make_sparse, ConvrelBlock
from gaussian_kernel import SpatialGaussianKernel


class AttributionBottleneck(nn.Module):

    @staticmethod
    def _sample_z(mu, log_noise_var):
        """ return mu with additive noise """
        log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = (log_noise_var / 2).exp()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, log_var) -> torch.Tensor:
        # KL[Q(z|x)||P(z)]
        # 0.5 * (tr(noise_cov) + mu ^ T mu - k  -  log det(noise)
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())


class PerSampleBottleneck(AttributionBottleneck):
    """
    The Attribution Bottleneck.
    Is inserted in a existing model to suppress information, parametrized by a suppression mask alpha.
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray, sigma, device=None, relu=False):
        """
        :param mean: The empirical mean of the activations of the layer
        :param std: The empirical standard deviation of the activations of the layer
        :param sigma: The standard deviation of the gaussian kernel to smooth the mask, or None for no smoothing
        :param device: GPU/CPU
        :param relu: True if output should be clamped at 0, to imitate a post-ReLU distribution
        """
        super().__init__()
        self.device = device
        self.relu = relu
        self.initial_value = 5.0
        self.std = torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha = nn.Parameter(torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None  # Filled on forward pass, used for loss

        if sigma is not None and sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            channels = self.mean.shape[0]
            self.smooth = SpatialGaussianKernel(kernel_size, sigma, channels, device=self.device)
        else:
            self.smooth = None

        self.reset_alpha()

    def reset_alpha(self):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, r):
        """ Remove information from r by performing a sampling step, parametrized by the mask alpha """
        # Smoothen and expand a on batch dimension
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(r.shape[0], r.shape[1], -1, -1)
        lamb = self.smooth(lamb) if self.smooth is not None else lamb

        # We normalize r to simplify the computation of the KL-divergence
        #
        # The equation in the paper is:
        # Z = λ * R + (1 - λ) * ε)
        # where ε ~ N(μ_r, σ_r**2)
        #  and given R the distribution of Z ~ N(λ * R, ((1 - λ) σ_r)**2)
        #
        # In the code μ_r = self.mean and σ_r = self.std.
        #
        # To simplify the computation of the KL-divergence we normalize:
        #   R_norm = (R - μ_r) / σ_r
        #   ε ~ N(0, 1)
        #   Z_norm ~ N(λ * R_norm, (1 - λ))**2)
        #   Z =  σ_r * Z_norm + μ_r
        #
        # We compute KL[ N(λ * R_norm, (1 - λ))**2) || N(0, 1) ].
        #
        # The KL-divergence is invariant to scaling, see:
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties

        r_norm = (r - self.mean) / self.std

        # Get sampling parameters
        noise_var = (1-lamb)**2
        scaled_signal = r_norm * lamb
        noise_log_var = torch.log(noise_var)

        # Sample new output values from p(z|r)
        z_norm = self._sample_z(scaled_signal, noise_log_var)
        self.buffer_capacity = self._calc_capacity(scaled_signal, noise_log_var)

        # Denormalize z to match magnitude of r
        z = z_norm * self.std + self.mean

        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        return z


class ZBottleneck(AttributionBottleneck):
    """
    The Attribution Bottleneck.
    Is inserted in a existing model to suppress information, parametrized by a suppression mask alpha.
    """
    def __init__(self, patch_size, max_CS_ratio):
        """
        :param mean: The empirical mean of the activations of the layer
        :param std: The empirical standard deviation of the activations of the layer
        :param sigma: The standard deviation of the gaussian kernel to smooth the mask, or None for no smoothing
        :param device: GPU/CPU
        :param relu: True if output should be clamped at 0, to imitate a post-ReLU distribution
        """
        super().__init__()
        r_ratio = 2
        phi_size = int(patch_size * patch_size * max_CS_ratio)

        self.con_I1 = ConvrelBlock(1, phi_size * 8, kernel_size=patch_size, stride=patch_size, padding=0, bias=True, act_type='prelu').cuda()
        self.con_I2 = ConvrelBlock(phi_size * 8, phi_size * 4 , kernel_size=1, stride=1, padding=0, bias=True,act_type='prelu').cuda()
        self.con_I3 = nn.Conv2d(phi_size * 4, phi_size, kernel_size=1, stride=1, padding=0, bias=True).cuda()

        self.con_Z = ConvrelBlock(phi_size * 2, phi_size * 2 , kernel_size=3, stride=3, padding=0, bias=True,act_type='prelu').cuda()
        self.con_Z1 = ConvrelBlock(phi_size * 2 , phi_size , kernel_size=1, stride=1, padding=0, bias=True,act_type='prelu').cuda()
        #self.con_Z2 = ConvrelBlock(phi_size, phi_size, kernel_size=1, stride=1, padding=0, bias=True,act_type='prelu').cuda()
        self.con_Z3 = nn.Conv2d(phi_size , phi_size, kernel_size=1, stride=1, padding=0, bias=True).cuda()
        self.sigmoid = nn.Sigmoid()
        self.lamb = torch.tensor([0.5,0.5])
        self.buffer_capacity = None  # Filled on forward pass, used for loss
        sigma = 0
        if sigma is not None and sigma > 0:
            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            channels = phi_size
            self.smooth = SpatialGaussianKernel(kernel_size, sigma, channels, device='cuda:0')
        else:
            self.smooth = None

    def forward(self, HR, r):
        """ Remove information from r by performing a sampling step, parametrized by the mask alpha """
        # Smoothen and expand a on batch dimension
        I = self.con_I1(HR)
        I = self.con_I2(I)
        I = self.con_I3(I)

        cat_HR_r = torch.cat((I,r),dim=1)

        self.alpha = self.con_Z(cat_HR_r)
        self.alpha = self.con_Z1(self.alpha)
        #self.alpha = self.con_Z2(self.alpha)
        self.alpha = self.con_Z3(self.alpha)
        self.lamb = self.sigmoid(self.alpha)
        self.lamb = self.lamb.expand(r.shape[0], r.shape[1], -1, -1)
        self.lamb = self.smooth(self.lamb) if self.smooth is not None else self.lamb

        # We normalize r to simplify the computation of the KL-divergence
        #
        # The equation in the paper is:
        # Z = λ * R + (1 - λ) * ε)
        # where ε ~ N(μ_r, σ_r**2)
        #  and given R the distribution of Z ~ N(λ * R, ((1 - λ) σ_r)**2)
        #
        # In the code μ_r = self.mean and σ_r = self.std.
        #
        # To simplify the computation of the KL-divergence we normalize:
        #   R_norm = (R - μ_r) / σ_r
        #   ε ~ N(0, 1)
        #   Z_norm ~ N(λ * R_norm, (1 - λ))**2)
        #   Z =  σ_r * Z_norm + μ_r
        #
        # We compute KL[ N(λ * R_norm, (1 - λ))**2) || N(0, 1) ].
        #
        # The KL-divergence is invariant to scaling, see:
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Properties
        std, mean = torch.std_mean(r,dim=[1,2,3],keepdim=True)
        r_norm = (r - mean) / std

        # Get sampling parameters
        noise_var = (1-self.lamb)**2 + 1e-9
        scaled_signal = r_norm * self.lamb
        noise_log_var = torch.log(noise_var)

        # Sample new output values from p(z|r)
        z_norm = self._sample_z(scaled_signal, noise_log_var)
        self.buffer_capacity = self._calc_capacity(scaled_signal, noise_log_var)
        # if self.buffer_capacity.mean() > 5000:
        #     print(noise_log_var)

        # Denormalize z to match magnitude of r
        z = z_norm * std + mean

        return z


class Discriminator(nn.Module):
    def __init__(self, patch_size, max_CS_ratio):
        super(Discriminator, self).__init__()

        phi_size = int(patch_size * patch_size * max_CS_ratio)
        # # Discriminator
        self.con_D1 = nn.Conv2d(phi_size, 1, kernel_size=3, stride=1, padding=0, bias=False).cuda()
        #self.relu = nn.ReLU()
        #self.lin_D2 = nn.Linear(phi_size, phi_size//2)
        #self.lin_D2 = nn.Linear(phi_size, 1)
        # self.lin_D3 = nn.Linear(phi_size//2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, y):

        out = self.con_D1(y)
        #out = self.relu(out)
        #out = self.lin_D2(out.view(out.shape[0],out.shape[1]))
        # out = self.relu(out)
        # out = self.lin_D3(out)
        out = self.sigmoid(out.view(out.shape[0],out.shape[1]))

        return out

class Encoder(nn.Module):
    def __init__(self, patch_size, max_CS_ratio):
        super(Encoder, self).__init__()
        phi_size = int(patch_size * patch_size * max_CS_ratio)
        # Generator
        self.con_E1 = nn.Conv2d(1, phi_size, kernel_size=patch_size, stride=patch_size, padding=0, bias=False).cuda()

    def forward(self, x, C):

        out = self.con_E1(x)
        out[:, C:, :, :] = torch.zeros_like(out[:, C:, :, :]).cuda()

        return out


class CS_reconstruction(nn.Module):
    def __init__(self, num_features, CS_ratio, patch_size, phase, G, Z, max_CS_ratio, act_type='relu', norm_type=None):
        super(CS_reconstruction, self).__init__()

        self.patch_size = patch_size
        self.F = num_features
        kernel_size1 = 3
        act_type = 'relu'
        self.l = 1
        self.n = 5
        self.phi_size = int(self.patch_size * self.patch_size * max_CS_ratio)

        # Sampling
        self.phi = G
        #self.phi = nn.Conv2d(1, self.phi_size, kernel_size=patch_size, stride=patch_size, padding=0, bias=False).cuda()

        # Information bottleneck
        self.IFBN = Z

        # Initial Reconstruction
        self.phi_inv = nn.Conv2d(self.phi_size, self.patch_size*self.patch_size*self.l, kernel_size=1, stride=1, padding=0, bias=False).cuda()

        # Deep Reconstruction
        self.D_e = ConvrelBlock(self.l, self.F, kernel_size1, stride=1, bias=True, pad_type='zero', act_type=act_type)

        self.D_m1 = nn.ModuleList()
        self.D_m2 = nn.ModuleList()
        for i in range(self.n):
            self.D_m1.append(ConvrelBlock(self.F, self.F, kernel_size1, stride=1, bias=True, pad_type='zero', act_type=act_type))
            self.D_m2.append(ConvrelBlock(self.F, self.F, kernel_size1, stride=1, bias=True, pad_type='zero', act_type=act_type))

        self.D_a = ConvrelBlock(self.F, self.l, kernel_size1, stride=1, bias=True, pad_type='zero', act_type=act_type)



    def forward(self, HR, CS_ratio, IFBN_s):

        self.b = HR.shape[0]
        self.c = HR.shape[1]
        self.h = HR.shape[2]
        self.w = HR.shape[3]

        C = int(self.patch_size * self.patch_size * CS_ratio)

        # input normalization
        std_i, mean_i = torch.std_mean(HR, dim=[1,2,3], keepdim=True)

        #HR = (HR - mean_i) / std_i

        # Sampling
        y = self.phi(HR, C)

        if IFBN_s == 1:

            y = self.IFBN(HR, y)
        else :
            y = y


        # y[:,C:,:,:] = torch.zeros_like(y[:,C:,:,:]).cuda()

        # Initial Reconstruction
        x_init = self.phi_inv(y)
        x_init = self.Reshape_Concat(x_init, HR.shape)

        # inverse normalization
        #x_init = x_init * std_i + mean_i

        # Deep Reconstruction
        x_f = self.D_e(x_init)

        for i in range(self.n):
            x_f1 = self.D_m1[i](x_f) + x_f
            x_f = self.D_m2[i](x_f1)

        x_f = self.D_a(x_f) + x_init


        return x_f, y[:,:C,:,:], x_init


    def Reshape_Concat(self,x_, shape):
        z = torch.zeros(shape).cuda()

        for i in range(shape[3]//self.patch_size):
            for j in range(shape[3]//self.patch_size):
                z[:,:,i*self.patch_size:(i+1)*self.patch_size,j*self.patch_size:(j+1)*self.patch_size] = torch.reshape(x_[:,:,i,j],[self.b,self.c,self.patch_size,self.patch_size])

        return z

    def _reset_state(self):
        self.block.reset_state()