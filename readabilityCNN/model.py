import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Some notes on how I changed this from the DiscriminatorWithClassifier

This is a copy of the DiscriminatorWithClassifier except all of the attribute
prediction code has been cut. Also, forward no longer takes an input of img_A
and img_B becuase we only care about the readability of the generator's ouput
not the ground truth font. 
'''
class readabilityCNN(nn.Module):
    def __init__(self, in_channel=3):
        super(readabilityCNN, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.1))

            return layers

        self.inputAndHiddenLayers = nn.Sequential(
            *discriminator_block(in_channel*2, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
        )
        self.outputLayer = nn.Conv2d(256, 1, 4, padding=1, bias=False)

    def forward(self, generatorOutput):
        hiddenLayersOutput = self.inputAndHiddenLayers(generatorOutput)
        readabilityScore = self.outputLayer(hiddenLayersOutput)

        return readabilityScore
