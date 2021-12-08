import torch
import os


def load_model_result(model, train_set, test_set, device):
    x1, e1, batch1 = None, None, None
    for data in train_set:
        data = data.to(device)
        _, x1, e1, batch1 = model(data)
    # mean = torch.mean(x1, dim=0)
    # std = torch.std(x1, dim=0) + 1e-12
    # x1 = (x1 - mean) / std

    x2, e2, batch2 = None, None, None
    for data in test_set:
        data = data.to(device)
        _, x2, e2, batch2 = model(data)
    # x2 = (x2 - mean) / std

    return [x1.detach(), e1.detach(), batch1.detach()], \
           [x2.detach(), e2.detach(), batch2.detach()]


def train_cp(model, optimizer, device, train_set, valid_set, num_epoch, path, m_name):

    for e in range(num_epoch):
        reconstruction_loss = 0
        reconstruction_loss_1 = 0
        for data in train_set:
            optimizer.zero_grad()
            data = data.to(device)
            z, _, _, _ = model(data)

            loss = torch.nn.MSELoss()(z, data.x)
            loss.backward()
            reconstruction_loss += loss.item()
            optimizer.step()

        for data in valid_set:
            data = data.to(device)
            z, _, _, _ = model(data)
            mse_loss = torch.nn.MSELoss()(z, data.x)
            reconstruction_loss_1 += mse_loss.item()

        reconstruction_loss /= len(train_set)
        reconstruction_loss_1 /= len(valid_set)

        print()
        print('Epoch: {:03d}'.format(e))
        print('Training Loss:', reconstruction_loss)
        print('Test Loss:', reconstruction_loss_1)

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path + m_name + ".ckpt")
    print("model saved")


def train_cf(model, optimizer, device, train_set, valid_set, num_epoch, group1, group2):

    for e in range(num_epoch):
        c_loss = 0
        accuracy = 0
        total_num = 0
        for data in train_set:
            optimizer.zero_grad()
            data = data.to(device)
            model.training = True
            c = model(group1[0], group1[1], group1[2])
            label = data.y.long()

            pred = c.argmax(dim=1)
            total_num += label.shape[0]
            accuracy += (pred == label).sum().item()
            c_loss = torch.nn.CrossEntropyLoss()(c, label)
            c_loss.backward()
            optimizer.step()

        accuracy /= total_num
        print()
        print('Epoch: {:03d}'.format(e))
        print('Train Loss:', c_loss.item())
        print('Train Accuracy:', accuracy)

        accuracy = 0
        total_num = 0
        for data in valid_set:
            data = data.to(device)
            model.training = False
            c = model(group2[0], group2[1], group2[2])
            pred = c.argmax(dim=1)
            label = data.y.long()
            total_num += label.shape[0]
            accuracy += (pred == label).sum().item()

        accuracy /= total_num

        print('Test Accuracy:', accuracy)
