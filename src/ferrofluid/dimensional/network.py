import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss1_weight=0.1,
        loss2_weight=0.1,
        droplet_size_idx=0,
        batch_size=None
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.loss1_weight = loss1_weight
        self.lr = lr
        self.n_units = n_units
        self.droplet_size_idx = droplet_size_idx
        self.batch_size = batch_size

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.SiLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.SiLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.SiLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.SiLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def fit(self, X, y):
        Xt = np_to_th(X)
        yt = np_to_th(y)

        dataset = thdat.TensorDataset(Xt, yt)
        if self.batch_size:
            loader = thdat.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            loader = [(Xt, yt)]  # fallback to full-batch SGD

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        tot_losses = []
        data_losses = []
        physics_losses = []

        for ep in range(self.epochs):
            for batch_X, batch_y in loader:
                optimiser.zero_grad()
                outputs = self.forward(batch_X)
                data_loss = self.loss1_weight * self.loss(batch_y, outputs)

                if self.loss2:
                    physics_loss = self.loss2_weight * self.loss2(self)
                    loss = data_loss + physics_loss
                else:
                    loss = data_loss

                loss.backward()
                optimiser.step()
            
            tot_losses.append(loss.cpu().detach().numpy())
            data_losses.append(data_loss.cpu().detach().numpy())
            physics_losses.append(physics_loss.cpu().detach().numpy())

            if (ep+1) % int(self.epochs / 10) == 0 or (ep >= 0 and ep < 10):
                print(f"Epoch {ep+1}/{self.epochs}, data loss: {data_loss}, physics loss: {physics_loss}")
        return data_losses, physics_loss, tot_losses

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))