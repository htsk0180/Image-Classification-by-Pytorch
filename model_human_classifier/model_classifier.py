# KÜTÜPHANELERİN YÜKLENMESİ:
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision 
from torchvision import datasets, transforms, models
from PIL import Image

# VERİNİN YÜKLENMESİ:
data_dir = './veriler'

# VERİNİN ÖN İŞLEMESİ / TEST VE TRAİN OLARAK AYRIMI:
def load_split_train_test(data_dir, valid_size = .2):
    train_transforms = transforms.Compose([
                                       transforms.RandomResizedCrop(224),
                                       transforms.Resize(224),
                                       transforms.ToTensor(),
                                       ])
    test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.Resize(224),
                                          transforms.ToTensor(),
                                      ])
    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=16)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=16)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)
test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                 ])

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels

images, labels = get_random_images(5)
to_pil = transforms.ToPILImage()
fig=plt.figure(figsize=(20,20))
classes=trainloader.dataset.classes
for ii in range(len(images)):
    image = to_pil(images[ii])
    sub = fig.add_subplot(1, len(images), ii+1)
    plt.axis('off')
    plt.imshow(image)
plt.show()

# GPU YA DA CPU'YA KARAR VERME:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODELİN OLUŞTURULMASI:
model = models.resnet50(pretrained=True)
for param in model.parameters():
     param.requires_grad = False
model.fc = nn.Sequential(nn.Linear(2048, 512),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(512, 2),
                               nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)
print('Modeliniz başarılı bir şekilde oluşturuldu, eğitime başlıyoruz...')

epochs = 10
print_every = 5
running_loss = 0
train_losses, test_losses = [], []
steps = 0
for epoch in range(epochs):
   epoch += 1
   for inputs, labels in trainloader:
      steps += 1
      print('Eğitim adımı ', steps)
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      logps = model.forward(inputs)
      loss = criterion(logps, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if steps % print_every == 0:
         test_loss = 0
         accuracy = 0
         model.eval()
         with torch.no_grad():
            for inputs, labels in testloader:
               inputs, labels = inputs.to(device), labels.to(device)
               logps = model.forward(inputs)
               batch_loss = criterion(logps, labels)
               test_loss += batch_loss.item()
               ps = torch.exp(logps)
               top_p, top_class = ps.topk(1, dim=1)
               equals = top_class == labels.view(*top_class.shape)
               accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
         train_losses.append(running_loss/len(trainloader))
         test_losses.append(test_loss/len(testloader))  
         print(f"\n     Epoch {epoch}/{epochs}: "
               f"Eğitim Kaybı: {running_loss/print_every:.3f}.. "
               f"Test Kaybı: {test_loss/len(testloader):.3f}.. "
               f"Test Accuracy: {accuracy/len(testloader):.3f}\n")
         running_loss = 0
         model.train()
         break

# MODELİN KAYDEDİLMESİ:
print(accuracy/len(testloader))
torch.save(model, 'model_classifier.pth')
'''
# MODELİN KULLANILMAK İÇİN YÜKLENMESİ:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('aerialmodel.pth')

# ÖRNEK TAHMİNİ:
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index
to_pil = transforms.ToPILImage()
images, labels = get_random_images(5)
fig=plt.figure(figsize=(20,10))
classes=trainloader.dataset.classes
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()

images, labels = get_random_images(20)
'''