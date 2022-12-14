import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReadabilityCNN(nn.Module):
    def __init__(self, in_channel=3):
        super(ReadabilityCNN, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.1))

            return layers

        self.inputAndHiddenLayers = nn.Sequential(
            *discriminator_block(in_channel, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
        )
        self.outputLayer = nn.Conv2d(256, 1, 7, padding=1, bias=False)

    def forward(self, generatorOutput):
        hiddenLayersOutput = self.inputAndHiddenLayers(generatorOutput)
        readabilityScore = self.outputLayer(hiddenLayersOutput)

        readabilityScore = readabilityScore.reshape((readabilityScore.shape[0],1))

        return readabilityScore
