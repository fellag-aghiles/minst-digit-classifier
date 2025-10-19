
import os
import torch 
from PIL import Image, ImageOps
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from tkinter import Tk, filedialog

EPOCH_NUM = 4

# Get data 
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

full_train = datasets.MNIST(root="data", download=True, train=True, transform=transform)
train_size = len(full_train) * 8 // 10
val_size = len(full_train) * 2 // 10
train_ds, val_ds = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_ds = datasets.MNIST(root="data", download=True, train=False, transform=transform)
test_loader = DataLoader(test_ds, batch_size=32)
#1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    _train = input("Do you want to train the model? (y/n): ") == 'y'
    model_path = "model_state.pt"
    if _train:
        clf.train()
        for epoch in range(EPOCH_NUM): # train for number of EPOCH_NUM
            running_loss = 0.0
            for batch in train_loader: 
                X,y = batch 
                X, y = X.to(device), y.to(device)
                yhat = clf(X) 
                loss = loss_fn(yhat, y) 

                # Apply backprop 
                opt.zero_grad()
                loss.backward() 
                opt.step() 

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            # Validation pass
            clf.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    out = clf(Xv)
                    val_loss += loss_fn(out, yv).item()
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == yv).sum().item()
                    total += yv.size(0)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total

            print(f"Epoch:{epoch} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}")
            clf.train()
        
        # save model
        save(clf.state_dict(), model_path) 
            
    # load model (either freshly trained or existing)
    if os.path.exists(model_path):
        clf.load_state_dict(load(model_path, map_location=device))  
        clf.eval()
    else:
        print("No saved model found. Exiting.")
        exit()

    # evaluate on test set
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt, yt = Xt.to(device), yt.to(device)
            out = clf(Xt)
            test_loss += loss_fn(out, yt).item()
            preds = torch.argmax(out, dim=1)
            correct += (preds == yt).sum().item()
            total += yt.size(0)
    print(f"Test loss: {test_loss/len(test_loader):.4f} Test accuracy: {correct/total:.4f}")

    # Create Tkinter root window (hidden) for a single image prediction
    root = Tk()
    root.withdraw()

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select test image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        exit()

    # Load and preprocess image
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28

    # Invert colors (MNIST has white digits on black background)
    img = ImageOps.invert(img)

    # Convert to tensor and normalize (same transform used for training)
    from torchvision.transforms import ToTensor, Normalize, Compose
    preprocess = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension

    # Run prediction
    with torch.no_grad():
        output = clf(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Display results
    img.show(title=f"Prediction: {prediction}")
    print(f"Model prediction: {prediction}")
