import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyperparamters:
n_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

train_data = torchvision.datasets.MNIST(root = "./data/", train = True, transform = transforms.ToTensor(), download = True)
test_data = torchvision.datasets.MNIST(root = "./data/", train = False, transform = transforms.ToTensor())

# Make the data iteratble and shuffle it as per the batch_size
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

#CNN:
class CovNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(CovNet, self).__init__()
        #sequentially add the layers
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 2), 
                nn.BatchNorm2d(16), 
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2), 
                nn.BatchNorm2d(32), 
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CovNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#training:
total_step = len(train_loader)
for epoch in tqdm(range(n_epochs)):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#testing:
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total+= labels.size(0)
        correct+= (predicted == labels).sum().item()
    print("test accuracy : {}".format(correct/total))

torch.save(model.state_dict(), "model.ckpt")