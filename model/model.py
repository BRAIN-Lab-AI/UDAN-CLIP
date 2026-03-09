# import logging
# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import os
# import model.networks as networks
# from .base_model import BaseModel
# from torchvision.models import vgg16
# import torch.nn.functional as F
# import numpy as np
# from tensorboardX import SummaryWriter
# import core.metrics as Metrics
# from metrics_util import uiqm, uciqe, compute_cpbd

# logger = logging.getLogger('base')

# class VGGPerceptualLoss(nn.Module):
#     def __init__(self, device):
#         super(VGGPerceptualLoss, self).__init__()
#         vgg = vgg16(pretrained=True).features.to(device).eval()
#         self.blocks = nn.ModuleList([
#             nn.Sequential(*list(vgg.children())[:4]),
#             nn.Sequential(*list(vgg.children())[4:9]),
#             nn.Sequential(*list(vgg.children())[9:16]),
#             nn.Sequential(*list(vgg.children())[16:23])
#         ])
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x, y):
#         loss = 0
#         for block in self.blocks:
#             x = block(x)
#             y = block(y)
#             loss += F.l1_loss(x, y)
#         return loss

# class SSIMLoss(nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIMLoss, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 3
#         self.window = self.create_window(window_size, self.channel)

#     def gaussian(self, window_size, sigma):
#         gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(sigma**2)) for x in range(window_size)])
#         return gauss/gauss.sum()

#     def create_window(self, window_size, channel):
#         _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
#         _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#         window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#         return window

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#         window = self.window.to(img1.device)

#         mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
#         mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1*mu2

#         sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
#         sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
#         sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

#         C1 = 0.01**2
#         C2 = 0.03**2

#         ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

#         if self.size_average:
#             return 1 - ssim_map.mean()
#         else:
#             return 1 - ssim_map.mean(1).mean(1).mean(1)

# class DDPM(BaseModel):
#     def __init__(self, opt):
#         super(DDPM, self).__init__(opt)
#         # define network and load pretrained models
#         self.netG = self.set_device(networks.define_G(opt))
#         self.schedule_phase = None

#         # set loss and load resume state
#         self.set_loss()
#         self.set_new_noise_schedule(
#             opt['model']['beta_schedule']['train'], schedule_phase='train')
        
#         # Add perceptual and SSIM loss
#         self.perceptual_loss = VGGPerceptualLoss(self.device)
#         self.ssim_loss = SSIMLoss().to(self.device)
        
#         # TensorBoardX writer
#         self.writer = SummaryWriter(logdir=opt['path']['tb_logger'])
#         self.global_step = 0
        
#         # Initialize log_dict for both training and inference
#         self.log_dict = OrderedDict()
        
#         if self.opt['phase'] == 'train':
#             self.netG.train()
#             # find the parameters to optimize
#             if opt['model']['finetune_norm']:
#                 optim_params = []
#                 for k, v in self.netG.named_parameters():
#                     v.requires_grad = False
#                     if k.find('transformer') >= 0:
#                         v.requires_grad = True
#                         v.data.zero_()
#                         optim_params.append(v)
#                         logger.info(
#                             'Params [{:s}] initialized to 0 and will optimize.'.format(k))
#             else:
#                 optim_params = list(self.netG.parameters())

#             self.optG = torch.optim.Adam(
#                 optim_params, lr=opt['train']["optimizer"]["lr"])
#         self.load_network()
#         self.print_network()

#     def feed_data(self, data):
#         self.data = self.set_device(data)

#     def optimize_parameters(self):
#         # Check if data is properly loaded
#         if not isinstance(self.data, dict) or 'HR' not in self.data:
#             logger.error(f"Invalid data format. Expected dict with 'HR' key. Got: {type(self.data)}")
#             return
            
#         # Generate SR image and compute diffusion loss
#         try:
#             total_loss, self.SR = self.netG(self.data)
            
#             # Ensure SR has proper shape
#             if self.SR.shape != self.data['HR'].shape:
#                 logger.error(f"Size mismatch: SR {self.SR.shape} vs HR {self.data['HR'].shape}")
#                 return
                
#             # Normalize SR to [0,1] range for perceptual and SSIM losses
#             self.SR = (torch.clamp(self.SR, -1, 1) + 1) / 2
#             hr_normalized = (torch.clamp(self.data['HR'], -1, 1) + 1) / 2
            
#             # Compute additional losses
#             l_perceptual = self.perceptual_loss(self.SR, hr_normalized)
#             l_ssim = self.ssim_loss(self.SR, hr_normalized)
            
#             # Combine all losses
#             total_loss = total_loss + 0.1 * l_perceptual + 0.1 * l_ssim
            
#             # Optimize
#             self.optG.zero_grad()
#             total_loss.backward()
#             self.optG.step()
            
#             # Log losses
#             self.log_dict['total_loss'] = total_loss.item()
#             self.log_dict['l_perceptual'] = l_perceptual.item()
#             self.log_dict['l_ssim'] = l_ssim.item()
            
#             # Log to tensorboard
#             if self.writer:
#                 self.writer.add_scalar('Loss/total', total_loss.item(), self.global_step)
#                 self.writer.add_scalar('Loss/perceptual', l_perceptual.item(), self.global_step)
#                 self.writer.add_scalar('Loss/ssim', l_ssim.item(), self.global_step)
#             self.global_step += 1
            
#         except Exception as e:
#             logger.error(f"Error in optimize_parameters: {str(e)}")
#             logger.error(f"SR shape: {self.SR.shape if hasattr(self.SR, 'shape') else 'None'}")
#             logger.error(f"HR shape: {self.data['HR'].shape if 'HR' in self.data else 'None'}")
#             return

#     def test(self, continous=False):
#         self.netG.eval()
#         with torch.no_grad():
#             if isinstance(self.netG, nn.DataParallel):
#                 self.SR = self.netG.module.super_resolution(
#                     self.data['SR'], continous)
#             else:
#                 self.SR = self.netG.super_resolution(
#                     self.data['SR'], continous)
            
#             # Ensure tensors have correct dimensions [B, C, H, W]
#             sr_normalized = (torch.clamp(self.SR[-1], -1, 1) + 1) / 2
#             hr_normalized = (torch.clamp(self.data['HR'], -1, 1) + 1) / 2
            
#             # Add batch and channel dimensions if missing
#             if sr_normalized.dim() == 2:  # [H, W]
#                 sr_normalized = sr_normalized.unsqueeze(0).repeat(1, 3, 1, 1)  # [1, 3, H, W]
#             elif sr_normalized.dim() == 3:  # [C, H, W]
#                 if sr_normalized.size(0) == 1:  # If single channel
#                     sr_normalized = sr_normalized.repeat(3, 1, 1)  # [3, H, W]
#                 sr_normalized = sr_normalized.unsqueeze(0)  # [1, 3, H, W]
            
#             if hr_normalized.dim() == 2:  # [H, W]
#                 hr_normalized = hr_normalized.unsqueeze(0).repeat(1, 3, 1, 1)  # [1, 3, H, W]
#             elif hr_normalized.dim() == 3:  # [C, H, W]
#                 if hr_normalized.size(0) == 1:  # If single channel
#                     hr_normalized = hr_normalized.repeat(3, 1, 1)  # [3, H, W]
#                 hr_normalized = hr_normalized.unsqueeze(0)  # [1, 3, H, W]
            
#             # Compute losses
#             l_perceptual = self.perceptual_loss(sr_normalized, hr_normalized)
#             l_ssim = self.ssim_loss(sr_normalized, hr_normalized)
#             l_pix = F.l1_loss(sr_normalized, hr_normalized)
            
#             # Total loss
#             total_loss = l_pix + 0.1 * l_perceptual + 0.1 * l_ssim
            
#             # Initialize log_dict if it doesn't exist
#             if not hasattr(self, 'log_dict'):
#                 self.log_dict = OrderedDict()
            
#             # Store total validation loss for logging
#             self.log_dict['val_total_loss'] = total_loss.item()
            
#             # Compute metrics
#             # Convert to numpy arrays and ensure same format
#             sr_img = Metrics.tensor2img(sr_normalized.squeeze(0))  # Remove batch dim and convert to numpy
#             hr_img = Metrics.tensor2img(hr_normalized.squeeze(0))  # Remove batch dim and convert to numpy
            
#             # Compute PSNR and SSIM
#             psnr = Metrics.calculate_psnr(sr_img, hr_img)
#             ssim = Metrics.calculate_ssim(sr_img, hr_img)
            
#             # Compute underwater metrics
#             uiqm_val = uiqm(sr_img)
#             uciqe_val = uciqe(sr_img)
#             cpbd_val = compute_cpbd(sr_img)
            
#             # Log metrics
#             self.log_validation_metrics(
#                 val_total_loss=total_loss.item(),
#                 val_pix_loss=l_pix.item(),
#                 val_perceptual_loss=l_perceptual.item(),
#                 val_ssim_loss=l_ssim.item(),
#                 val_psnr=psnr,
#                 val_ssim=ssim,
#                 val_uiqm=uiqm_val,
#                 val_uciqe=uciqe_val,
#                 val_cpbd=cpbd_val
#             )
            
#         self.netG.train()

#     def sample(self, batch_size=1, continous=False):
#         self.netG.eval()
#         with torch.no_grad():
#             if isinstance(self.netG, nn.DataParallel):
#                 self.SR = self.netG.module.sample(batch_size, continous)
#             else:
#                 self.SR = self.netG.sample(batch_size, continous)
#         self.netG.train()

#     def set_loss(self):
#         if isinstance(self.netG, nn.DataParallel):
#             self.netG.module.set_loss(self.device)
#         else:
#             self.netG.set_loss(self.device)

#     def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
#         if self.schedule_phase is None or self.schedule_phase != schedule_phase:
#             self.schedule_phase = schedule_phase
#             if isinstance(self.netG, nn.DataParallel):
#                 self.netG.module.set_new_noise_schedule(
#                     schedule_opt, self.device)
#             else:
#                 self.netG.set_new_noise_schedule(schedule_opt, self.device)

#     def get_current_log(self):
#         return self.log_dict

#     def get_current_visuals(self, need_LR=True, sample=False):
#         out_dict = OrderedDict()
#         if sample:
#             out_dict['SAM'] = self.SR.detach().float().cpu()
#         else:
#             out_dict['SR'] = self.SR.detach().float().cpu()
#             out_dict['INF'] = self.data['SR'].detach().float().cpu()
#             out_dict['HR'] = self.data['HR'].detach().float().cpu()
#             if need_LR and 'LR' in self.data:
#                 out_dict['LR'] = self.data['LR'].detach().float().cpu()
#             else:
#                 out_dict['LR'] = out_dict['INF']
#         return out_dict

#     def print_network(self):
#         s, n = self.get_network_description(self.netG)
#         if isinstance(self.netG, nn.DataParallel):
#             net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
#                                              self.netG.module.__class__.__name__)
#         else:
#             net_struc_str = '{}'.format(self.netG.__class__.__name__)

#         logger.info(
#             'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
#         logger.info(s)

#     def save_network(self, epoch, iter_step):
#         gen_path = os.path.join(
#             self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
#         opt_path = os.path.join(
#             self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
#         # gen
#         network = self.netG
#         if isinstance(self.netG, nn.DataParallel):
#             network = network.module
#         state_dict = network.state_dict()
#         for key, param in state_dict.items():
#             state_dict[key] = param.cpu()
#         torch.save(state_dict, gen_path)
#         # opt
#         opt_state = {'epoch': epoch, 'iter': iter_step,
#                      'scheduler': None, 'optimizer': None}
#         opt_state['optimizer'] = self.optG.state_dict()
#         torch.save(opt_state, opt_path)

#         logger.info(
#             'Saved model in [{:s}] ...'.format(gen_path))

#     def load_network(self):
#         load_path = self.opt['path']['resume_state']
#         if load_path is not None:
#             logger.info(
#                 'Loading pretrained model for G [{:s}] ...'.format(load_path))
#             gen_path = '{}_gen.pth'.format(load_path)
#             opt_path = '{}_opt.pth'.format(load_path)
#             # gen
#             network = self.netG
#             if isinstance(self.netG, nn.DataParallel):
#                 network = network.module
#             network.load_state_dict(torch.load(
#                 gen_path), strict=False)
#             if self.opt['phase'] == 'train':
#                 # optimizer
#                 opt = torch.load(opt_path)
#                 # self.optG.load_state_dict(opt['optimizer'])  # <-- comment this out
#                 self.begin_step = opt['iter']
#                 self.begin_epoch = opt['epoch']

#     def log_validation_loss(self, val_total_loss, val_pix_loss, val_perceptual_loss, val_ssim_loss):
#         self.writer.add_scalar('Loss/val_total', val_total_loss, self.global_step)
#         self.writer.add_scalar('Loss/val_pix', val_pix_loss, self.global_step)
#         self.writer.add_scalar('Loss/val_perceptual', val_perceptual_loss, self.global_step)
#         self.writer.add_scalar('Loss/val_ssim', val_ssim_loss, self.global_step)

#     def log_validation_metrics(self, val_total_loss, val_pix_loss, val_perceptual_loss, val_ssim_loss, val_psnr=None, val_uiqm=None, val_uciqe=None, val_cpbd=None, val_ssim=None):
#         self.writer.add_scalar('Loss/val_total', val_total_loss, self.global_step)
#         self.writer.add_scalar('Loss/val_pix', val_pix_loss, self.global_step)
#         self.writer.add_scalar('Loss/val_perceptual', val_perceptual_loss, self.global_step)
#         self.writer.add_scalar('Loss/val_ssim', val_ssim_loss, self.global_step)
#         if val_psnr is not None:
#             self.writer.add_scalar('Metric/val_psnr', val_psnr, self.global_step)
#         if val_uiqm is not None:
#             self.writer.add_scalar('Metric/val_uiqm', val_uiqm, self.global_step)
#         if val_uciqe is not None:
#             self.writer.add_scalar('Metric/val_uciqe', val_uciqe, self.global_step)
#         if val_cpbd is not None:
#             self.writer.add_scalar('Metric/val_cpbd', val_cpbd, self.global_step)
#         if val_ssim is not None:
#             self.writer.add_scalar('Metric/val_ssim', val_ssim, self.global_step)


#new code for inference


import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from torchvision.models import vgg16
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import core.metrics as Metrics
from metrics_util import uiqm, uciqe, compute_cpbd

logger = logging.getLogger('base')

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features.to(device).eval()
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),
            nn.Sequential(*list(vgg.children())[4:9]),
            nn.Sequential(*list(vgg.children())[9:16]),
            nn.Sequential(*list(vgg.children())[16:23])
        ])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        
        # Add perceptual and SSIM loss
        self.perceptual_loss = VGGPerceptualLoss(self.device)
        self.ssim_loss = SSIMLoss().to(self.device)
        
        # TensorBoardX writer
        self.writer = SummaryWriter(logdir=opt['path']['tb_logger'])
        self.global_step = 0
        
        # Initialize log_dict for both training and inference
        self.log_dict = OrderedDict()
        
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        # Check if data is properly loaded
        if not isinstance(self.data, dict) or 'HR' not in self.data:
            logger.error(f"Invalid data format. Expected dict with 'HR' key. Got: {type(self.data)}")
            return
            
        # Generate SR image and compute diffusion loss
        try:
            total_loss, self.SR = self.netG(self.data)
            
            # Ensure SR has proper shape
            if self.SR.shape != self.data['HR'].shape:
                logger.error(f"Size mismatch: SR {self.SR.shape} vs HR {self.data['HR'].shape}")
                return
                
            # Normalize SR to [0,1] range for perceptual and SSIM losses
            self.SR = (torch.clamp(self.SR, -1, 1) + 1) / 2
            hr_normalized = (torch.clamp(self.data['HR'], -1, 1) + 1) / 2
            
            # Compute additional losses
            l_perceptual = self.perceptual_loss(self.SR, hr_normalized)
            l_ssim = self.ssim_loss(self.SR, hr_normalized)
            
            # Combine all losses
            total_loss = total_loss + 0.1 * l_perceptual + 0.1 * l_ssim
            
            # Optimize
            self.optG.zero_grad()
            total_loss.backward()
            self.optG.step()
            
            # Log losses
            self.log_dict['total_loss'] = total_loss.item()
            self.log_dict['l_perceptual'] = l_perceptual.item()
            self.log_dict['l_ssim'] = l_ssim.item()
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/total', total_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/perceptual', l_perceptual.item(), self.global_step)
                self.writer.add_scalar('Loss/ssim', l_ssim.item(), self.global_step)
            self.global_step += 1
            
        except Exception as e:
            logger.error(f"Error in optimize_parameters: {str(e)}")
            logger.error(f"SR shape: {self.SR.shape if hasattr(self.SR, 'shape') else 'None'}")
            logger.error(f"HR shape: {self.data['HR'].shape if 'HR' in self.data else 'None'}")
            return

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
            
            # Ensure tensors have correct dimensions [B, C, H, W]
            sr_normalized = (torch.clamp(self.SR[-1], -1, 1) + 1) / 2
            hr_normalized = (torch.clamp(self.data['HR'], -1, 1) + 1) / 2
            
            # Add batch and channel dimensions if missing
            if sr_normalized.dim() == 2:  # [H, W]
                sr_normalized = sr_normalized.unsqueeze(0).repeat(1, 3, 1, 1)  # [1, 3, H, W]
            elif sr_normalized.dim() == 3:  # [C, H, W]
                if sr_normalized.size(0) == 1:  # If single channel
                    sr_normalized = sr_normalized.repeat(3, 1, 1)  # [3, H, W]
                sr_normalized = sr_normalized.unsqueeze(0)  # [1, 3, H, W]
            
            if hr_normalized.dim() == 2:  # [H, W]
                hr_normalized = hr_normalized.unsqueeze(0).repeat(1, 3, 1, 1)  # [1, 3, H, W]
            elif hr_normalized.dim() == 3:  # [C, H, W]
                if hr_normalized.size(0) == 1:  # If single channel
                    hr_normalized = hr_normalized.repeat(3, 1, 1)  # [3, H, W]
                hr_normalized = hr_normalized.unsqueeze(0)  # [1, 3, H, W]
            
            # Compute losses
            l_perceptual = self.perceptual_loss(sr_normalized, hr_normalized)
            l_ssim = self.ssim_loss(sr_normalized, hr_normalized)
            l_pix = F.l1_loss(sr_normalized, hr_normalized)
            
            # Total loss
            total_loss = l_pix + 0.1 * l_perceptual + 0.1 * l_ssim
            
            # Initialize log_dict if it doesn't exist
            if not hasattr(self, 'log_dict'):
                self.log_dict = OrderedDict()
            
            # Store total validation loss for logging
            self.log_dict['val_total_loss'] = total_loss.item()
            
            # Compute metrics
            # Convert to numpy arrays and ensure same format
            sr_img = Metrics.tensor2img(sr_normalized.squeeze(0))  # Remove batch dim and convert to numpy
            hr_img = Metrics.tensor2img(hr_normalized.squeeze(0))  # Remove batch dim and convert to numpy
            
            # Compute PSNR and SSIM
            psnr = Metrics.calculate_psnr(sr_img, hr_img)
            ssim = Metrics.calculate_ssim(sr_img, hr_img)
            
            # Compute underwater metrics
            uiqm_val = uiqm(sr_img)
            uciqe_val = uciqe(sr_img)
            cpbd_val = compute_cpbd(sr_img)
            
            # Log metrics
            self.log_validation_metrics(
                val_total_loss=total_loss.item(),
                val_pix_loss=l_pix.item(),
                val_perceptual_loss=l_perceptual.item(),
                val_ssim_loss=l_ssim.item(),
                val_psnr=psnr,
                val_ssim=ssim,
                val_uiqm=uiqm_val,
                val_uciqe=uciqe_val,
                val_cpbd=cpbd_val
            )
            
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                # self.optG.load_state_dict(opt['optimizer'])  # <-- comment this out
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

    def log_validation_loss(self, val_total_loss, val_pix_loss, val_perceptual_loss, val_ssim_loss):
        self.writer.add_scalar('Loss/val_total', val_total_loss, self.global_step)
        self.writer.add_scalar('Loss/val_pix', val_pix_loss, self.global_step)
        self.writer.add_scalar('Loss/val_perceptual', val_perceptual_loss, self.global_step)
        self.writer.add_scalar('Loss/val_ssim', val_ssim_loss, self.global_step)

    def log_validation_metrics(self, val_total_loss, val_pix_loss, val_perceptual_loss, val_ssim_loss, val_psnr=None, val_uiqm=None, val_uciqe=None, val_cpbd=None, val_ssim=None):
        self.writer.add_scalar('Loss/val_total', val_total_loss, self.global_step)
        self.writer.add_scalar('Loss/val_pix', val_pix_loss, self.global_step)
        self.writer.add_scalar('Loss/val_perceptual', val_perceptual_loss, self.global_step)
        self.writer.add_scalar('Loss/val_ssim', val_ssim_loss, self.global_step)
        if val_psnr is not None:
            self.writer.add_scalar('Metric/val_psnr', val_psnr, self.global_step)
        if val_uiqm is not None:
            self.writer.add_scalar('Metric/val_uiqm', val_uiqm, self.global_step)
        if val_uciqe is not None:
            self.writer.add_scalar('Metric/val_uciqe', val_uciqe, self.global_step)
        if val_cpbd is not None:
            self.writer.add_scalar('Metric/val_cpbd', val_cpbd, self.global_step)
        if val_ssim is not None:
            self.writer.add_scalar('Metric/val_ssim', val_ssim, self.global_step)
