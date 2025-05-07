import torch
from torchvision.models import resnet50

# Initialize model
model = resnet50(weights=None)  # No pre-trained weights initially

# Load state dict
state_dict = torch.load(
    "C:\\Users\\karan\\Desktop\\fakedetection\\models\\best_model.pth", 
    map_location="cpu"
)

# Check keys and load with strict=False if needed
try:
    model.load_state_dict(state_dict, strict=True)
except RuntimeError:
    print("Key mismatch detected. Loading with strict=False")
    model.load_state_dict(state_dict, strict=False)

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
