import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import GalaxyCNN


# Galaxy classes — must match the label encoding used during training
CLASS_NAMES = [
    "Smooth",
    "Spiral",
    "Disk_without_spiral"
]

MODEL_PATH = Path("models/best_galaxy_model.pth")

# Use the same preprocessing as validation during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def predict_image(image_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the trained model
    model = GalaxyCNN(num_classes=3)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    # Support either a plain state dictionary or a checkpoint dictionary
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load and preprocess the galaxy image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    print("\nGalaxy Classification")
    print("---------------------")
    print("Image:", image_path)
    print("Predicted class:", CLASS_NAMES[predicted_index])

    print("\nClass probabilities:")
    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{name}: {probability.item() * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print("Error: Image file not found:", image_path)
        sys.exit(1)

    if not MODEL_PATH.is_file():
        print("Error: Trained model not found:", MODEL_PATH)
        sys.exit(1)

    predict_image(image_path)
