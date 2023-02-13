from torch import nn

class lenet5Conv(nn.Module):
    def __init__(self, input_shape: int, hidden_units: list, output_shape:int) -> None:
        super().__init__()
        self.first_conv2d = nn.Conv2d(in_channels=input_shape, out_channels=hidden_units[0], kernel_size=5, stride=1, padding=2)
        self.first_sigmoid = nn.Sigmoid()
        self.first_maxPool2d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.second_conv2d = nn.Conv2d(in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=5, stride=1, padding=0)
        self.second_sigmoid = nn.Sigmoid()
        self.second_maxPool2d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=output_shape),
            nn.LogSoftmax(1)
        )
    def forward(self, x):
        # Block 1
        self.output_first_conv2d = self.first_conv2d(x)
        sigmoid = self.first_sigmoid(self.output_first_conv2d)
        self.output_first_maxPool2d, self.output_first_maxPool2d_loc = self.first_maxPool2d(sigmoid)
        # Block 2
        self.output_second_conv2d = self.second_conv2d(self.output_first_maxPool2d)
        sigmoid = self.second_sigmoid(self.output_second_conv2d)
        self.output_second_maxPool2d, self.output_second_maxPool2d_loc = self.second_maxPool2d(sigmoid)
        # Classifier
        
        self.classifier_output = self.classifier(self.output_second_maxPool2d)

        return self.classifier_output
