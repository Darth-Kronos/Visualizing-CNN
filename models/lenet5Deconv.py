from torch import nn
import torch

def transpose_conv(input, kernel, stride, padding):
    trans_kernel = torch.transpose(kernel, 0, 1)
    output_image = nn.functional.conv2d(input, trans_kernel, stride=stride, padding=padding)
    return output_image

class lenet5Deconv(nn.Module):
    def __init__(self, hidden_layers: list, weights: dict) -> None:
        super().__init__()

        # Block 2
        self.second_maxUnPool = nn.MaxUnpool2d(2, 2)
        self.second_ReLU = nn.ReLU()
        # self.second_convT2d = nn.ConvTranspose2d(hidden_layers[1], hidden_layers[0],5,1,0)
        self.second_conv2d_weights = weights['second_conv2d.weight']

        # Block 1
        self.first_maxUnPool = nn.MaxUnpool2d(2, 2)
        self.first_ReLU = nn.ReLU()
        # self.first_convT2d = nn.ConvTranspose2d(hidden_layers[0], 1, 5, 1, 2)
        self.first_conv2d_weights = weights['first_conv2d.weight']

    def forward(self, x, layer, switch):
        if layer == 'second_maxPool2d':
            x = self.second_maxUnPool(x, switch['second_maxPool2d'])
            x = self.second_ReLU(x)
            x = transpose_conv(x, self.second_conv2d_weights, stride=1, padding=4)

        x = self.first_maxUnPool(x, switch['first_maxPool2d'])
        x = self.first_ReLU(x)
        x = transpose_conv(x, self.first_conv2d_weights, stride=1, padding=2)
        return x


