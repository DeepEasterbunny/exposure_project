from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.enc5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 16)
        )
        self.enc6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.enc7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.enc8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5)
        )
        self.dec2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5)
        )
        self.dec3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5)
        )
        # self.dec4 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(ngf * 8)
        # )
        self.dec4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        self.dec5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        self.dec6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        self.dec7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.dec8 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        e1 = self.enc1(x)

        e2 = self.enc2(e1)

        e3 = self.enc3(e2)
       
        e4 = self.enc4(e3)

        e5 = self.enc5(e4)


        # e6 = self.enc6(e5)
        # print(e6.shape)
        # e7 = self.enc7(e6)
        # print(e6.shape)
        # e8 = self.enc8(e7)

        # Decoder forward pass with skip connections
        # d1 = self.dec1(e8)
        # print(d1.shape)
        # d2 = self.dec2(torch.cat([d1, e7], 1))
        # d3 = self.dec3(torch.cat([d2, e6], 1))

        d4 = self.dec4(e5)
        # print(d3.shape)
        # d4 = self.dec4(torch.cat([d3, e4], 1))
  
        d5 = self.dec5(torch.cat([d4, e4], 1))

        d6 = self.dec6(torch.cat([d5, e3], 1))
 
        d7 = self.dec7(torch.cat([d6, e2], 1))

        d8 = self.dec8(torch.cat([d7, e1], 1))


        return d8
    
class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf, n_layers=3):
        super(Discriminator, self).__init__()
        
        layers = [
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)