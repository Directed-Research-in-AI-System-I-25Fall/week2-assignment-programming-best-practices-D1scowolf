import torch
from transformers import AutoImageProcessor, ResNetModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# load model and image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetModel.from_pretrained("microsoft/resnet-50")

# define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# load dataset: download can avoid connection error
mnist_test = datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64)

# projection: the most important part, add a classification head after the last layer
class Classifier(torch.nn.Module):
    def __init__(self, resnet_model):
        super(Classifier, self).__init__()
        self.resnet = resnet_model
        self.classifier = torch.nn.Linear(2048, 10)

    def forward(self, x):
        last_layer = self.resnet(pixel_values=x).last_hidden_state
        # (batch_size, num_channels, height, width) -> (batch_size, num_channels)
        # average pooling
        modified_last_layer = F.adaptive_avg_pool2d(last_layer, (1, 1)).squeeze()
        result = self.classifier(modified_last_layer)
        return result

# get the result
model_with_classifier = Classifier(model)
model_with_classifier.eval()
model_with_classifier.to(torch.device("cuda"))

correct = total = 0

with torch.no_grad():
    for image, label in test_loader:
        image = image.to(torch.device("cuda"))
        label = label.to(torch.device("cuda"))
        output = model_with_classifier(image)
        prediction = torch.argmax(output, dim=-1)
        correct += (prediction == label).sum().item()
        total += label.size(0)

acc = correct / total * 100
print(f"{acc:.2f}%")
