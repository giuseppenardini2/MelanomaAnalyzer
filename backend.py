import numpy
from PIL import Image
import torch
from torchvision import transforms
from torch import load
from melanoma_cnn import SkinCancerCNN

# Function to analyze photos uploaded by the user
def is_melanoma(file_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = SkinCancerCNN().to(device)
    with open('best_model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f, map_location=device))

    norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(file_name).convert("RGB")
    img_tensor = norm(img).unsqueeze(0).to(device)

    clf.eval()
    with torch.no_grad():
        output = clf(img_tensor)
        pred = torch.argmax(output, dim=1)

    if pred.item() == 0:  # Return 0=Melanoma, 1=NotMelanoma
        return True
    return False