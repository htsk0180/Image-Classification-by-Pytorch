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
data_dir = './testveri'

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
'''
DİKKAT !
Model eğitimini tamamladıktan sonra modelimizi "model_classifier" isminde kaydetmiştik.
"model_classifier" modelimizin şuanda doğruluk oranı: 0.88.
Bu oran epoch ve lr değerleri ve modelde kullanılan algoritmaları değiştirerek modelin başarı (Accuracy) değerini gözlemleyebilirsiniz.
Şimdi ise eğitimde olmayan, internetten rastgele indirdiğimiz küçük veri seti ile eğitilmiş modelimizi çağırarak test edeceğiz.
'''
# MODELİN KULLANILMAK İÇİN YÜKLENMESİ:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('model_classifier.pth')

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
images, labels = get_random_images(2)
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
