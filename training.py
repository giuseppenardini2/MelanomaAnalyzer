import numpy
import torch 
from PIL import Image
from torch import nn, save, load
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from melanoma_cnn import SkinCancerCNN

# Data loader and image trasformation
if __name__ == "__main__" :
    
    trns = transforms.Compose([
        transforms.Resize((224, 224)),            # Image resize 
        transforms.RandomHorizontalFlip(p=0.5),   # Image mirroring
        transforms.RandomVerticalFlip(p=0.5),     # Vertical image flip 
        transforms.RandomRotation(25),            # Image rotation up to 25 degrees
        transforms.ColorJitter(brightness=0.1, contrast=0.1),   # Small light variation 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Color flattening,(standard ImageNet) 
    ])

    norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder("DermMel/train_sep", transform=trns)
    eval_dataset = datasets.ImageFolder("DermMel/valid", transform=norm) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,  
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6)
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6)
    
    # N.B. Image transformation is important because every time we move from one epoch to another the transformation is reapplied, 
         #this allows us to have a digital database much larger than the real one     

# Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instance of the neural network, loss, optimizer and print check
clf = SkinCancerCNN().to(device)
#print(next(clf.parameters()).device)
opt = Adam(clf.parameters(), lr=1e-3) # If we increase batch size (lr/batch size) should remain constant
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.5) # adapt lr
loss_fn = nn.CrossEntropyLoss()


# Training flow 
if __name__ == "__main__":   
    
    best_eval_loss = 9999
    for epoch in range(35): # Train for tot epochs
        running_loss = 0.0  
        correct = 0
        total = 0

        # Train
        clf.train() 
        for batch in train_loader: 
            X,y = batch 
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            yhat = clf(X) 
            loss = loss_fn(yhat, y)

            # Apply backprop 
            opt.zero_grad() # Gradient reset
            loss.backward() # How much each weight in the model contributed to the loss
            opt.step()      # Weights adjust: weight = weight-learningRate*gradient

            running_loss += loss.item()
            _, predicted = torch.max(yhat.data, 1) # Takes the index of the highest probability
            total += y.size(0)
            correct += (predicted == y).sum().item()

        avg_loss = running_loss / len(train_loader) # Loss: gap between model prediction and reality
        accuracy = 100 * correct / total            

        print(f"Epoch: {epoch+1:02d} | Avg. Epoch Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}%")
        
        # Evaluation: check how the model work on a different dataset w/o it learning from it
        running_loss = 0.0
        correct = 0
        total = 0
        clf.eval()
        with torch.no_grad():
            for batch in eval_loader: 
                X,y = batch 
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                yhat = clf(X)
                loss = loss_fn(yhat, y)
                
                running_loss += loss.item()
                _, predicted = torch.max(yhat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            avg_val_loss = running_loss / len(eval_loader)
            eval_accuracy = 100 * correct / total
            
            print(f"Epoch: {epoch+1:02d} | Avg. Epoch Eval. Loss: {avg_val_loss:.4f} | Eval. Accuracy: {eval_accuracy:.2f}%")
            

            scheduler.step(avg_val_loss)  # if n avg_val_loss > n-1 avg_val_loss, patrience-1, else patience = 3 
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch: {epoch+1:02d} | Learning Rate: {current_lr}")

            print(f"*************************** NEXT EPOCH ***************************  ")

        # Save the best model until now
        if avg_val_loss < best_eval_loss: 
            best_eval_loss  = avg_val_loss
            with open('best_model_state.pt', 'wb') as f: 
                 save(clf.state_dict(), f) 
        
       
