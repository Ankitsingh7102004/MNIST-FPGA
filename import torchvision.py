import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),          # converts to [0,1] tensor
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST mean/std
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

print(len(train_dataset), len(test_dataset))  # should print 60000 10000