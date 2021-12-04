import numpy as np 
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
from PIL import Image

class train_val_Data(Dataset):
    
    def __init__(self, X_input, Y_output, Y_label):
        self.X_input = X_input
        self.Y_output = Y_output
        self.Y_label = Y_label
        
    def __getitem__(self, index):
        return self.X_input[index], self.Y_output[index], self.Y_label[index]
        
    def __len__ (self):
        return len(self.X_input)
    
class AutoEncoder(nn.Module):
    def __init__(self, add_gaussian = False):
        super(AutoEncoder, self).__init__()
        
        self.add_gaussian = add_gaussian
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=2, kernel_size=3)
        
        self.Tconv1 = nn.ConvTranspose2d(in_channels=2, out_channels=20, kernel_size=3)
        self.Tconv2 = nn.ConvTranspose2d(in_channels=20, out_channels=3, kernel_size=3)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        out = self.relu(self.conv1(inputs))
        out = self.relu(self.conv2(out))
        
        if self.add_gaussian == True:
            out = out + (0.01**0.5)*torch.randn(out.shape).to(device) # add Gaussian noise of zero mean

        out = self.relu(self.Tconv1(out))
        out = self.relu(self.Tconv2(out))

        return out

def gen_data_label(loader):
    with torch.no_grad():
        for X_batch, Y_gt, Y_label in loader:
            X_batch, Y_gt = X_batch.to(device), Y_gt.to(device)
            Y_pred = model(X_batch)
            #print(Y_pred.shape) # torch.Size([32, 3, 26, 26])
            Y_pred = Y_pred.reshape((Y_pred.shape[0], 26, 26, 3))
            gen_data_list.append(Y_pred.cpu().numpy())
            gen_label_list.append(Y_label.numpy())

if __name__ == "__main__":
    torch.manual_seed(100)
    BATCH_SIZE = 32
    EPOCH = 80
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.01
    
    data = np.load('./wafer/data.npy')
    data = data.reshape((data.shape[0], 3, 26, 26))
    
    label = np.load('./wafer/label.npy')
    
    whole_data = train_val_Data(data, data, label)
    whole_data_loader = DataLoader(dataset=whole_data, batch_size=BATCH_SIZE, shuffle=False)
    # autoencoder: reconstruct input data
    train_inputs, val_inputs, train_outputs, val_outputs, train_labels, val_labels = train_test_split(data, data, label, random_state=1234, test_size=0.2, shuffle=True)
    
    train_inputs = torch.from_numpy(train_inputs)
    val_inputs = torch.from_numpy(val_inputs)
    train_outputs = torch.from_numpy(train_outputs)
    val_outputs = torch.from_numpy(val_outputs)
    train_labels = torch.from_numpy(train_labels)
    val_labels = torch.from_numpy(val_labels)
    
    train_data = train_val_Data(train_inputs, train_outputs, train_labels)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data = train_val_Data(val_inputs, val_outputs, val_labels)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AutoEncoder(add_gaussian=False)
    model.double()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    criterion = nn.MSELoss(reduction='mean') # mean square error
    
    train_loss_list = []
    min_val_loss = -100000.0
    select_epoch = 0
    for epoch in range(1, EPOCH+1, 1):
        print('EPOCH', epoch)
        model.train()
        epoch_loss_train = 0
        for X_batch, Y_gt, Y_label in train_loader:
            X_batch, Y_gt = X_batch.to(device), Y_gt.to(device)
            optimizer.zero_grad()
            
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_gt)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss_train += loss.item()
            
        train_loss_list.append(epoch_loss_train/len(train_loader))
        
        # validation
        model.eval()
        with torch.no_grad():
            epoch_loss_val = 0
            for X_batch, Y_gt, Y_label in val_loader:
                X_batch, Y_gt = X_batch.to(device), Y_gt.to(device)
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_gt)
                epoch_loss_val += loss.item()
            if min_val_loss < 0 or epoch_loss_val < min_val_loss:
                min_val_loss = epoch_loss_val
                select_epoch = epoch
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                torch.save(checkpoint, './model/best_model.pth')
    print('Select epoch', select_epoch, 'to generate new samples')
    
    # plot training loss
    plt.figure(figsize=(15, 10))
    plt.title("Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Epoch Average Loss")
    plt.plot(np.arange(1, EPOCH+1), train_loss_list, label = 'train loss')
    plt.legend(loc="best")
    plt.savefig('Train_Loss.png')

    # generate new samples by train & val
    model = AutoEncoder(add_gaussian=True)
    model.double()
    model.to(device)
    checkpoint = torch.load('./model/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    gen_data_list, gen_label_list = [], []
    
    for i in range(5):
        gen_data_label(whole_data_loader)
    
    gen_data = gen_data_list[0]
    gen_label = gen_label_list[0]
    for i in range(1, len(gen_label_list), 1):
        gen_data = np.concatenate((gen_data, gen_data_list[i]), axis = 0)
        gen_label = np.concatenate((gen_label, gen_label_list[i]), axis = 0)
        
    np.save('gen_data.npy', np.array(gen_data)) # save
    np.save('gen_label.npy', np.array(gen_label)) # save
    
    for class_type in range(0, 9, 1):
        for data_index in range(0, data.shape[0], 1):
            if class_type == gen_label[data_index]:
                im = Image.fromarray((data[data_index].reshape(26, 26, 3) * 255.).astype(np.uint8))
                im.save('./sample_image/class_' + str(class_type) + '/original.png')
                for i in range(0, 5, 1):
                    im = Image.fromarray((gen_data[data_index + i*data.shape[0]] * 255.).astype(np.uint8))
                    im.save('./sample_image/class_' + str(class_type) + '/Gaussian_' + str(i+1) + '.png')
                break
    