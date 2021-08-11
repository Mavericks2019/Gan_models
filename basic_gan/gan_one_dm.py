import time
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
from torch.utils.data import DataLoader


CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_point")
NETD_PATH = ""  # os.path.join(CHECKPOINT_DIR, "GAN_D1504_D.pkl")
NETG_PATH = ""  # os.path.join(CHECKPOINT_DIR, "GAN_G1504_G.pkl")

TOTAL_WORK_NUM = 60000
BATCH_SIZE = 64
RANDOM_NUM_COUNT = 20
ART_POINT_COUNTS = 100
LEARNING_RATE_G = 0.0001  # learning rate for generator
LEARNING_RATE_D = 0.0001  # learning rate for discriminator
USE_GPU = True
ART_COMPONENTS = 100  # it could be total point G can drew in the canvas
PAINT_POINTS = np.vstack([np.linspace(-3, 3, ART_POINT_COUNTS) for _ in range(TOTAL_WORK_NUM)])


def artist_works(func="sin"):  # painting from the famous artist (real target)
    # a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    r = 0.02 * np.random.randn(1, ART_POINT_COUNTS)
    if func == "sin":
        paintings = np.sin(PAINT_POINTS) + r
    elif func == "tanh":
        paintings = np.tanh(PAINT_POINTS) + r
    elif func == "cos":
        paintings = np.cos(PAINT_POINTS) + r
    else:
        return None
    paintings = torch.from_numpy(paintings).float()
    return paintings


class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        self.g_net = nn.Sequential(
            nn.Linear(RANDOM_NUM_COUNT, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ART_POINT_COUNTS),
        )

    def forward(self, x):
        x = self.g_net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d_net = nn.Sequential(
            nn.Linear(ART_POINT_COUNTS, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.d_net(x)
        return x


if __name__ == '__main__':
    G = Generator()
    D = Discriminator()
    if NETD_PATH:
        D.load_state_dict(torch.load(NETD_PATH, map_location=lambda storage, loc: storage))
    if NETG_PATH:
        G.load_state_dict(torch.load(NETG_PATH, map_location=lambda storage, loc: storage))
    G_optimizer = optim.Adam(G.parameters(), lr=LEARNING_RATE_D)
    D_optimizer = optim.Adam(D.parameters(), lr=LEARNING_RATE_G)
    y_real, y_fake = torch.ones(BATCH_SIZE, 1), torch.zeros(BATCH_SIZE, 1)
    if USE_GPU:
        G.cuda()
        D.cuda()
        BCE_loss = nn.BCELoss().cuda()
        artist_paintings = DataLoader(artist_works("sin").cuda(), batch_size=BATCH_SIZE)
        y_real, y_fake = y_real.cuda(), y_fake.cuda()

    else:
        BCE_loss = nn.BCELoss()
        artist_paintings = DataLoader(artist_works("sin"), batch_size=BATCH_SIZE)

    D.train()

    for epoch in range(1000):
        G.train()
        epoch_start_time = time.time()
        for ite, rx in list(enumerate(artist_paintings)):
            if ite == artist_paintings.dataset.__len__() // BATCH_SIZE:
                break
            random_ideas = torch.randn(BATCH_SIZE, RANDOM_NUM_COUNT)  # random ideas
            if USE_GPU:
                realx = rx.cuda()
                random_ideas = random_ideas.cuda()

            # handle D loss
            D_optimizer.zero_grad()
            d_real = D(rx)
            d_real_loss = BCE_loss(d_real, y_real)
            fake_paintings = G(random_ideas)  # fake painting from G (random ideas)
            d_fake = D(fake_paintings)
            d_fake_loss = BCE_loss(d_fake, y_fake)
            d_total_loss = d_real_loss + d_fake_loss
            d_total_loss.backward()
            D_optimizer.step()

            # handle G loss
            G_optimizer.zero_grad()
            fake_paintings = G(random_ideas)
            d_fake = D(fake_paintings)
            g_loss = BCE_loss(d_fake, y_real)
            g_loss.backward()
            G_optimizer.step()

            plt.cla()
            plt.plot(PAINT_POINTS[0], fake_paintings.data.cpu().numpy()[0], c='#4AD631', lw=3,
                     label='Generated painting', )
            plt.plot(PAINT_POINTS[0], np.sin(PAINT_POINTS[0]), c='#74BCFF', lw=3, label='upper bound')
            plt.text(-1, 0.75, 'D accuracy=%.2f (0.5 for D to converge)' % d_real.data.cpu().numpy().mean(),
                     fontdict={'size': 13})
            plt.text(-1, 0.5, 'D score= %.2f (-1.38 for G to converge)' %
                     -d_total_loss.data.cpu().numpy(), fontdict={'size': 13})
            plt.ylim((-1, 1))
            plt.legend(loc='upper right', fontsize=10)
            plt.draw()
            plt.pause(0.001)

            if ((ite + 1) % 100) == 0:
                print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                      ((epoch + 1), (ite + 1), len(artist_paintings.dataset) // BATCH_SIZE,
                       d_total_loss.item(), g_loss.item()))
        if epoch % 50 == 0:
            torch.save(G.state_dict(), os.path.join(CHECKPOINT_DIR, "GAN_G" + F"{epoch}" + "_50_G.pkl"))
            torch.save(D.state_dict(), os.path.join(CHECKPOINT_DIR, "GAN_D" + F"{epoch}" + "_50_D.pkl"))
