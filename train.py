import torch
import torch.optim as optim
import torchvision.transforms as T
import config as cfg

from torch.nn import MSELoss
from torch.utils.data import DataLoader
from conv_network import Network
from data import SlsDataset
from sacred import Experiment
from utils import split_train_val, plot_loss

ex = Experiment("SLS-classifier training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(sls, no_sls, random_no_sls, batch_size, lr, momentum, epochs, model_path, img_size):
    # get data
    sls_train, sls_val = split_train_val(sls, cfg.val_size)
    no_sls_train, no_sls_val = split_train_val(no_sls, cfg.val_size)
    random_train, random_val = split_train_val(random_no_sls, cfg.val_size)
    # transforms
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor()])
    # instantiate datasets
    train_dataset = SlsDataset(
        sls_train, no_sls_train, random_train, transform)
    val_dataset = SlsDataset(sls_val, no_sls_val, random_val, transform)
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True)
    # define network
    net = Network().to(device).float()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = MSELoss()

    train_logs_list, valid_logs_list = [], []
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            opt.zero_grad()
            # Forward Pass
            prediction = net(data)
            # Find the Loss
            loss = criterion(label, prediction)
            loss.backward()
            opt.step()
            # Calculate Loss
            train_loss += loss.item()
        valid_loss = 0.0
        net.eval()     # Optional when not using Model Specific layer
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            # Forward Pass
            prediction = net(data)
            # Find the Loss
            loss = criterion(label, prediction)
            # Calculate Loss
            valid_loss += loss.item()
        valid_score = valid_loss/len(val_loader)
        train_logs_list.append(train_loss/len(train_loader))
        valid_logs_list.append(valid_score)
        # Save model if a better val IoU score is obtained
        if valid_score <= min(valid_logs_list):
            best_score = valid_score
            torch.save(net, model_path)
            print('Model saved!')
        print('Epoch: {} - Loss: {:.6f} - Validation: {:.6f}'.format(epoch + 1,
                                                                     train_loss /
                                                                     len(train_loader),
                                                                     valid_loss/len(val_loader)))

    plot_loss(train_logs_list, valid_logs_list)


train(cfg.target_sls, cfg.target_nosls, cfg.random_dir,
      cfg.batch_size, cfg.lr, cfg.momentum, cfg.epochs, cfg.model_path, cfg.img_size)
