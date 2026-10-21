import os
import copy
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

class ProduceResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv_base = nn.Sequential(*list(base.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(base.fc.in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.conv_base(x)
        logits = self.classifier(features)
        return logits


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=10):
    best_loss = float("inf")
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == "train" and scheduler is not None:
                scheduler.step()

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            print(f"{phase.upper():5} Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                print(f"  ↳ [Checkpoint] Best model updated (validation Loss: {best_loss:.4f})")

    print(f"\nTraining Complete. Best validation Accuracy: {best_acc * 100:.2f}%")
    model.load_state_dict(best_weights)
    return model


if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing on: {device}")

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
    validation_dataset = datasets.ImageFolder("data/validation", transform=validation_transforms)

    
    train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=0  
    )

    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names[:5]}...")

    model = ProduceResNet50(num_classes=num_classes).to(device)
    dataloaders = {"train": train_loader, "validation": validation_loader}
    criterion = nn.CrossEntropyLoss()

    # --- STAGE 1: Warmup Classifier Head (3 Epochs) ---
    print("\n>>> Stage 1: Warmup Classifier Head (Backbone Frozen)")
    for param in model.conv_base.parameters():
        param.requires_grad = False

    optimizer_stage1 = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
    model = train_model(model, dataloaders, criterion, optimizer_stage1, scheduler=None, device=device, num_epochs=3)

    # --- STAGE 2: Fine-Tune Layer 4 + Head (8 Epochs) ---
    print("\n>>> Stage 2: Fine-Tuning Layer 4 and Head")
    for param in model.conv_base[7].parameters():
        param.requires_grad = True

    optimizer_stage2 = torch.optim.AdamW([
        {"params": model.conv_base[7].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=8, eta_min=1e-6)
    model = train_model(model, dataloaders, criterion, optimizer_stage2, scheduler=scheduler, device=device, num_epochs=8)

    # Save Checkpoint
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/BRAHMA_CV50_PRODUCTION.pt"
    torch.save({
        "num_classes": num_classes,
        "class_names": class_names,
        "state_dict": model.state_dict()
    }, save_path)
    print(f"\nModel and class dictionary successfully saved to {save_path}")