import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn.functional as functional
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        self.encoder_layer1 = nn.Linear(x_dim, h_dim1)
        self.encoder_layer2 = nn.Linear(h_dim1, h_dim2)
        self.encoder_latent1 = nn.Linear(h_dim2, z_dim)
        self.encoder_latent2 = nn.Linear(h_dim2, z_dim)
        self.decoder_layer1 = nn.Linear(z_dim, h_dim2)
        self.decoder_layer2 = nn.Linear(h_dim2, h_dim1)
        self.decoder_layer3 = nn.Linear(h_dim1, x_dim)

    def encoder(self, input):
        encoder_layer1_output = functional.relu(self.encoder_layer1(input))
        encoder_layer2_output = functional.relu(self.encoder_layer2(encoder_layer1_output))
        return self.encoder_latent1(encoder_layer2_output), self.encoder_latent1(encoder_layer2_output)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, latent_sample):
        decoder_layer1_output = functional.relu(self.decoder_layer1(latent_sample))
        decoder_layer2_output = functional.relu(self.decoder_layer2(decoder_layer1_output))
        return functional.sigmoid(self.decoder_layer3(decoder_layer2_output))

    def forward(self, input):
        mu, log_var = self.encoder(input.view(-1, 784))
        latent_sample = self.sampling(mu, log_var)
        return self.decoder(latent_sample), mu, log_var


def loss_function(output, input, mu, log_var):
    BCE = functional.binary_cross_entropy(output, input.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_no, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        output_batch, mu, log_var = vae(data)
        loss = loss_function(output_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch Number: {} Average loss incurred is : {:.5f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for input, _ in test_loader:
            output, mu, log_var = vae(input)
            test_loss += loss_function(output, input, mu, log_var).item()

    test_loss /= len(test_loader.dataset)
    print('Test set loss incurred is : {:.5f}'.format(test_loss))


if __name__ == '__main__':

    bs = 100
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
    optimizer = optim.Adam(vae.parameters())

    for epoch in range(1, 51):
        train(epoch)
        test()

    with torch.no_grad():
        z = torch.randn(64, 2)
        sample = vae.decoder(z)
        save_image(sample.view(64, 1, 28, 28), './VAE_Samples/sample' + '.png')
