'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: AutoEncoder.py
@time: 11/27/2019 11:41 PM
'''

import torch
import torch.nn as nn
import torch.utils.data as DATA
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# hyper parameters
epochs = 10
batch_size = 64
lr = 0.005  # learning rate
download_mnist = True
n_test_img = 5

# mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='/mnist/MNIST/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=download_mnist,
)

# plot one example
print(train_data.data.size())  # (60000, 28, 28)
print(train_data.targets.size())  # (6000)

plt.figure()
plt.imshow(train_data.data[2].numpy(), cmap='gray')
plt.title(train_data.targets[2].numpy())
plt.show()

train_loader = DATA.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),  # compress to 3 features which can be visualized in plt
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # compress to a range(0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()
autoencoder.cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(2, n_test_img, figsize=(5, 2))
plt.ion()

# original data (first row) for viewing
view_data = train_data.data[:n_test_img].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
for i in range(n_test_img):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for epoch in range(epochs):
    for step, (x, _) in enumerate(train_data):
        x = x.cuda()
        b_x = x.view(-1, 28 * 28)
        b_y = x.view(-1, 28 * 28)
        # b_label = b_label.cuda()

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)  # mean square error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data.cuda())
            for i in range(n_test_img):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()
