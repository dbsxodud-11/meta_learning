import torch.nn as nn
import torch


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden=[64, 64],
                 hidden_activation=nn.LeakyReLU(), out_activation=nn.Identity(),
                 bias=True, critic=False):
        super(MLP, self).__init__()

        self.critic = critic
        in_dims = [in_dim] + hidden
        out_dims = hidden + [out_dim]

        self.l = nn.ModuleList()
        self.act = nn.ModuleList()

        for i, o in zip(in_dims, out_dims):
            layer = nn.Linear(i, o, bias=bias)
            nn.init.xavier_uniform_(layer.weight)
            self.l.append(layer)

        for _ in range(len(hidden)):
            self.act.append(hidden_activation)
        self.act.append(out_activation)

    def forward(self, x, a=None):

        if self.critic:
            x = torch.cat([x, a], dim=-1)

        for l, act in zip(self.l, self.act):
            x = l(x)
            x = act(x)

        return x


if __name__ == "__main__":

    model = MLP(in_dim=5, out_dim=4, hidden=[16, 16])
    x = torch.randn(128, 5)
    y = model(x)

    print(" ")
    print("------- MLP test -------")
    print("input shape  : ", x.shape)
    print("output shape : ", y.shape)