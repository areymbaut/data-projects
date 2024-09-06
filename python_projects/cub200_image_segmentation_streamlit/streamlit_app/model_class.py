import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetDown(nn.Module):
    def __init__(self, input_size, output_size):
        super(UnetDown, self).__init__()
        
        model = [nn.BatchNorm2d(input_size),
                 nn.ELU(),
                 nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(output_size),
                 nn.ELU(),
                 nn.MaxPool2d(2),
                 nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):        
        return self.model(x)
      

class UnetUp(nn.Module):
    def __init__(self, input_size, output_size):
        super(UnetUp, self).__init__()

        model = [nn.BatchNorm2d(input_size),
                 nn.ELU(),
                 nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(output_size),
                 nn.ELU(),
                 nn.Upsample(scale_factor=2, mode="nearest"),  # Counterpart of the MaxPool2d
                 nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1)]
          
        self.model = nn.Sequential(*model)
            
    def forward(self, x):
        return self.model(x)
            
         
class Unet(nn.Module):
    def __init__(self, channels_in, channels_out=2):
        super(Unet, self).__init__()
        
        self.conv_in = nn.Conv2d(channels_in, 64, 
                                 kernel_size=3, stride=1, padding=1)  # H x W --> H x W
        
        self.down1 = UnetDown(64, 64)  # H x W --> H/2 x W/2
        self.down2 = UnetDown(64, 128)  # H/2 x W/2 --> H/4 x W/4
        self.down3 = UnetDown(128, 128)  # H/4 x W/4 --> H/8 x W/8
        self.down4 = UnetDown(128, 256)  # H/8 x W/8 --> H/16 x W/16

        # The "*2" below come from the skip-connection concatenations
        self.up4 = UnetUp(256, 128)  # H/16 x W/16 --> H/8 x W/8
        self.up5 = UnetUp(128*2, 128)  # H/8 x W/8 --> H/4 x W/4
        self.up6 = UnetUp(128*2, 64)  # H/4 x W/4 --> H/2 x W/2
        self.up7 = UnetUp(64*2, 64)  # H/2 x W/2 --> H x W
        
        self.conv_out = nn.Conv2d(64*2, channels_out, 
                                  kernel_size=3, stride=1, padding=1)  # H x W --> H x W

    def forward(self, x):
        x0 = self.conv_in(x)  # 64 x H x W
        
        x1 = self.down1(x0)  # 64 x H/2 x W/2
        x2 = self.down2(x1)  # 128 x H/4 x W/4
        x3 = self.down3(x2)  # 128 x H/8 x W/8
        x4 = self.down4(x3)  # 256 x H/16 x W/16

        # Bottleneck --> 256 x H/16 x W/16

        x5 = self.up4(x4)  # 128 x H/8 x W/8
        
        x5_ = torch.cat((x5, x3), 1)  # 256 x H/8 x W/8 (skip connection)
        x6 = self.up5(x5_)  # 128 x H/4 x W/4
        
        x6_ = torch.cat((x6, x2), 1)  # 256 x H/4 x W/4 (skip connection)
        x7 = self.up6(x6_)  # 64 x H/2 x W/2
        
        x7_ = torch.cat((x7, x1), 1)  # 128 x H/2 x W/2 (skip connection)
        x8 = self.up7(x7_)  # 64 x H x W
        
        x8_ = F.elu(torch.cat((x8, x0), 1))  # 128 x H x W        
        return self.conv_out(x8_)  # channels_out x H x W