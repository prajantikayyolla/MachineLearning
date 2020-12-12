import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.generator_layer1 = nn.Linear(g_input_dim, 256)
        self.generator_layer2 = nn.Linear(self.generator_layer1.out_features, self.generator_layer1.out_features * 2)
        self.generator_layer3 = nn.Linear(self.generator_layer2.out_features, self.generator_layer2.out_features * 2)
        self.generator_layer4 = nn.Linear(self.generator_layer3.out_features, g_output_dim)

    def forward(self, input):
        generator_layer1_output = functional.leaky_relu(self.generator_layer1(input), 0.2)
        generator_layer2_output = functional.leaky_relu(self.generator_layer2(generator_layer1_output), 0.2)
        generator_layer3_output = functional.leaky_relu(self.generator_layer3(generator_layer2_output), 0.2)
        return torch.tanh(self.generator_layer4(generator_layer3_output))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.discriminator_layer1 = nn.Linear(d_input_dim, 1024)
        self.discriminator_layer2 = nn.Linear(self.discriminator_layer1.out_features, self.discriminator_layer1.out_features // 2)
        self.discriminator_layer3 = nn.Linear(self.discriminator_layer2.out_features, self.discriminator_layer2.out_features // 2)
        self.discriminator_layer4 = nn.Linear(self.discriminator_layer3.out_features, 1)

    def forward(self, input):
        discriminator_layer1_output = functional.leaky_relu(self.discriminator_layer1(input), 0.2)
        discriminator_layer1_output_dropout = functional.dropout(discriminator_layer1_output, 0.3)
        discriminator_layer2_output = functional.leaky_relu(self.discriminator_layer2(discriminator_layer1_output_dropout), 0.2)
        discriminator_layer2_output_dropout = functional.dropout(discriminator_layer2_output, 0.3)
        discriminator_layer3_output = functional.leaky_relu(self.discriminator_layer3(discriminator_layer2_output_dropout), 0.2)
        discriminator_layer3_output_dropout = functional.dropout(discriminator_layer3_output, 0.3)
        return torch.sigmoid(self.discriminator_layer4(discriminator_layer3_output_dropout))


def D_train(x):
    # setting gradients to zero before training
    D.zero_grad()

    # training on real data with positive labels
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
    output = D(x_real)
    real_loss = criterion(output, y_real)

    # training on fake data with zero labels
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))
    output = D(x_fake)
    fake_loss = criterion(output, y_fake)

    loss = real_loss + fake_loss
    loss.backward()
    D_optimizer.step()
    return loss.data.item()


def G_train(x):
    # setting gradients to zero before training
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    output_generator = G(z)
    output_discriminator = D(output_generator)
    loss = criterion(output_discriminator, y)

    loss.backward()
    G_optimizer.step()

    return loss.data.item()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loading data
    bs = 100
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # defining the model
    z_dim = 100
    mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
    G = Generator(z_dim, mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    # defining loss and optimizers
    criterion = nn.BCELoss()
    lr = 0.0002
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    n_epoch = 200
    for epoch in range(1, n_epoch + 1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            D_losses.append(D_train(x))
            G_losses.append(G_train(x))

        print('epoch [%d/%d]: discriminator_loss: %.5f, generator_loss: %.5f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        # taking sample output after every 10 epochs
        if (epoch % 10 == 0):
            with torch.no_grad():
                test_z = Variable(torch.randn(bs, z_dim).to(device))
                generated = G(test_z)

            save_image(generated.view(generated.size(0), 1, 28, 28), './GAN_samples/sample_ep' + str(epoch) + '.png')
