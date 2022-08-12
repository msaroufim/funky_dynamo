import torch
import torch.nn as nn
import probtorch

class Encoder(nn.Module):
    def __init__(self, num_pixels=784, num_hidden=50, num_digits=10, num_style=2):
        super(self.__class__, self).__init__()
        self.h = nn.Sequential(
                    nn.Linear(num_pixels, num_hidden),
                    nn.ReLU())
        self.y_log_weights = nn.Linear(num_hidden, num_digits)
        self.z_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.z_log_std = nn.Linear(num_hidden + num_digits, num_style)

    def forward(self, x, y_values=None, num_samples=10):
        q = probtorch.Trace()
        x = x.expand(num_samples, *x.size())
        if y_values is not None:
            y_values = y_values.expand(num_samples, *y_values.size())
        h = self.h(x)
        y = q.concrete(logits=self.y_log_weights(h), temperature=0.66,
                       value=y_values, name='y')
        h2 = torch.cat([y, h], -1)
        z = q.normal(loc=self.z_mean(h2),
                     scale=torch.exp(self.z_log_std(h2)),
                     name='z')
        return q