"""

Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

This is a modified version of the original code of the paper:
Mahmoud Afifi, et al., CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks. TPAMI, 2021.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSubnet(nn.Module):
  def __init__(self, block_depth, conv_depth, scale=0.25):
    super(LocalSubnet, self).__init__()
    self.block_depth = block_depth
    self.conv_depth = conv_depth
    self.scale = scale
    self.net = torch.nn.Sequential()
    for i in range(self.block_depth):
      if i != self.block_depth - 1:
        if i == 0:
          conv = torch.nn.Conv2d(3, self.conv_depth, 3, padding=1)
          torch.nn.init.kaiming_normal_(conv.weight)
          torch.nn.init.zeros_(conv.bias)
        else:
          conv = torch.nn.Conv2d(self.conv_depth, self.conv_depth, 3,
                                 padding=1)
          torch.nn.init.kaiming_normal_(conv.weight)
          torch.nn.init.zeros_(conv.bias)
        self.net.add_module('conv%d' % i, conv)
        self.net.add_module('leakyRelu%d' % i, torch.nn.LeakyReLU(
          inplace=False))
      else:
        conv = torch.nn.Conv2d(self.conv_depth, 3, 3, padding=1)
        torch.nn.init.kaiming_normal_(conv.weight)
        torch.nn.init.zeros_(conv.bias)
        self.net.add_module('conv%d' % i, conv)
        self.net.add_module('tanh-out', torch.nn.Tanh())

  def forward(self, x):
    local_layer = self.net(x) * self.scale
    return local_layer


class Flatten(nn.Module):
  def forward(self, x):
    x = x.view(x.size()[0], -1)
    return x


class GlobalSubnet(nn.Module):
  def __init__(self, block_depth, conv_depth, in_img_sz):
    super(GlobalSubnet, self).__init__()
    self.block_depth = block_depth
    self.conv_depth = conv_depth
    self.in_img_sz = in_img_sz
    self.net = torch.nn.Sequential()
    for i in range(self.block_depth):
      if i == 0:
        conv = torch.nn.Conv2d(3, self.conv_depth, 3, padding=1)
        torch.nn.init.kaiming_normal_(conv.weight)
        torch.nn.init.zeros_(conv.bias)

      else:
        conv = torch.nn.Conv2d(self.conv_depth, self.conv_depth, 3,
                               padding=1)
        torch.nn.init.kaiming_normal_(conv.weight)
        torch.nn.init.zeros_(conv.bias)
      self.net.add_module('conv%d' % i, conv)
      self.net.add_module('leakyRelu%d' % i, torch.nn.LeakyReLU(
        inplace=False))
      self.net.add_module('maxpool%d' % i,
                          torch.nn.MaxPool2d(2, stride=2))

    self.net.add_module('flatten', Flatten())
    dummy = torch.zeros(1, 3, self.in_img_sz, self.in_img_sz)
    with torch.no_grad():
      out_dim = self.net(dummy).shape[1]

    self.net.add_module('fc1', nn.Linear(out_dim, 512))
    self.net.add_module('l_relu1', nn.LeakyReLU(inplace=True))
    self.net.add_module('fc2', torch.nn.Linear(512, 256))
    self.net.add_module('l_relu2', nn.LeakyReLU(inplace=True))
    self.net.add_module('fc3', torch.nn.Linear(256, 256))
    self.net.add_module('out', torch.nn.Linear(256, 3 * 6))


  def forward(self, x):
    x_resized = F.interpolate(x, size=(self.in_img_sz, self.in_img_sz), mode="bilinear", align_corners=False)
    m = self.net(x_resized)
    return m


class CIEXYZNet(nn.Module):
  def __init__(self, local_depth=8, local_conv_depth=24,
               global_depth=4, global_conv_depth=24,
               global_in=96, scale=0.25):
    super(CIEXYZNet, self).__init__()
    self.local_depth = local_depth
    self.local_conv_depth = local_conv_depth
    self.global_depth = global_depth
    self.global_conv_depth = global_conv_depth
    self.global_in = global_in
    self.scale = scale
    self.srgb2xyz_local_net = LocalSubnet(
      block_depth=self.local_depth, conv_depth=self.local_conv_depth,
      scale=self.scale)

    self.srgb2xyz_globa_net = GlobalSubnet(block_depth=self.global_depth, conv_depth=self.global_conv_depth,
                                           in_img_sz=self.global_in)

  def forward_local(self, x):
    local_layer = self.srgb2xyz_local_net(x)
    return local_layer

  def forward_global(self, x):
      m_v = self.srgb2xyz_globa_net(x)
      m = m_v.view(x.size(0), 3, 6)
      x_flat = x.view(x.size(0), 3, -1)
      x_proj = self.kernel(x_flat)
      y_flat = torch.bmm(m, x_proj)
      y = y_flat.view(x.size(0), x.size(1), x.size(2), x.size(3))
      return y

  def forward(self, srgb):
    l_xyz = srgb - self.forward_local(srgb)
    xyz = self.forward_global(l_xyz)
    return xyz

  def get_num_of_params(self) -> int:
    """Returns total number of parameters."""
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return total_params


  @staticmethod
  def kernel(x):
    return torch.cat((x, x * x), dim=1)
