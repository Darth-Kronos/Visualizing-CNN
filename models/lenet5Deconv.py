from torch import nn

class lenet5Deconv(nn.Module):
    def __init__(self, hidden_layers) -> None:
        super().__init__()

        # Block 2
        self.second_maxUnPool = nn.MaxUnpool2d(2, 2)
        self.second_sigmod = nn.Sigmoid()
        self.second_convT2d = nn.ConvTranspose2d(hidden_layers[1], hidden_layers[0],5,1,0)
        
        # Block 1
        self.first_maxUnPool = nn.MaxUnpool2d(2, 2)
        self.first_sigmod = nn.Sigmoid()
        self.first_convT2d = nn.ConvTranspose2d(hidden_layers[0], 1, 5, 1, 2)

    def forward(self, x, layer, switch):
        if layer == 2:
            x = self.second_convT2d(x)
            x = self.first_maxUnPool(x, switch['first_maxPool2d'])
            x = self.first_sigmod(x)
        x = self.first_convT2d(x)
        return x

