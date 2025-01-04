import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import ast
from tqdm import tqdm
import wandb
import os
from data_loader import FMRIDataGenerator
from sklearn.model_selection import StratifiedKFold
from model import CNNLSTM
from sklearn.metrics import f1_score
import numpy as np
import config

def load_folds_from_csv(csv_path):
    if os.path.exists(csv_path):
        print(f"Loading existing folds from {csv_path}")
        df = pd.read_csv(csv_path)
        folds = []
        for _, row in df.iterrows():
            fold_data = {
                "train_images": ast.literal_eval(row["train_images"]),
                "train_labels": ast.literal_eval(row["train_labels"]),
                "val_images": ast.literal_eval(row["val_images"]),
                "val_labels": ast.literal_eval(row["val_labels"]),
            }
            folds.append(fold_data)
        return folds
    
    else:
        print(f"No existing folds found at {csv_path}. Creating new folds...")
        # Load the original metadata file
        metadata_csv = config.CSV_PATH + config.METADATA_CSV
        file = pd.read_csv(metadata_csv)
        
        # Prepare data for stratification
        list_IDs = file['Image'].tolist()
        labels = dict(zip(file['Image'], file['DX']))
        label_list = [labels[img_id] for img_id in list_IDs]
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
        fold_data = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(list_IDs, label_list)):
            fold_dict = {
                'fold': fold + 1,
                'train_images': [list_IDs[i] for i in train_idx],
                'train_labels': [label_list[i] for i in train_idx],
                'val_images': [list_IDs[i] for i in val_idx],
                'val_labels': [label_list[i] for i in val_idx]
            }
            fold_data.append(fold_dict)
        
        # Save the newly created folds
        fold_df = pd.DataFrame(fold_data)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fold_df.to_csv(csv_path, index=False)
        print(f"Created and saved {config.N_SPLITS} folds to {csv_path}")
        
        return fold_data

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    for _, batch in pbar:
        X, y = batch
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(y.cpu().numpy())

        # Update progress bar
        accuracy = (np.array(predictions) == np.array(true_labels)).mean()
        pbar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = (np.array(predictions) == np.array(true_labels)).mean()
    epoch_f1 = f1_score(true_labels, predictions, average='weighted')
    
    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc * 100,
        "f1": epoch_f1
    }

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    scores = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")
    with torch.no_grad():
        for _, batch in pbar:
            X, y = batch
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            running_loss += loss.item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
            scores.extend(probs.cpu().numpy())

            # Update progress bar
            accuracy = (np.array(predictions) == np.array(true_labels)).mean()
            pbar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy})

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    scores = np.array(scores)

    return {
        "loss": running_loss / len(val_loader),
        "accuracy": (predictions == true_labels).mean() * 100,
        "f1": f1_score(true_labels, predictions, average='weighted'),
        "predictions": predictions,
        "true_labels": true_labels,
        "scores": scores
    }

def save_checkpoint(model, optimizer, fold_idx, epoch, metrics, history, checkpoint_dir="/root/autodl-tmp/CNNLSTM/Project/checkpoints/"):
    """Save model checkpoint with history."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'history': history  # Save the complete training history
    }
    path = os.path.join(checkpoint_dir, f'checkpoint_fold_{fold_idx + 1}_epoch_{epoch + 1}.pth')
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(checkpoint_path, model, optimizer):    
    if os.path.exists(checkpoint_path):  
        checkpoint = torch.load(checkpoint_path)  
        model.load_state_dict(checkpoint['model_state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        start_epoch = checkpoint['epoch']  
        history = checkpoint['history']  
        print(f"Resuming from epoch {start_epoch}")  
        return start_epoch, history  
    return 0, {'train_loss': [], 'val_loss': [], 'val_acc': []} 

def cross_validation_training(folds, resume_from=None, preprocessed_dir=config.PREPROCESSED_DIR, 
                            num_epochs=config.EPOCH, batch_size=config.BATCH_SIZE, 
                            learning_rate=config.LR, dropout_rate=config.DROP_OUT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results = []
    best_accuracy = 0.0

    # Initialize wandb
    wandb.init(
        project='fMRI',
        config={
            'num_classes': 4,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'dropout_rate': dropout_rate
        }
    )

    for fold_idx, fold in enumerate(folds):
        print(f"Starting Fold {fold_idx + 1}/{len(folds)}")

        # Prepare data loaders
        train_images = fold["train_images"]
        train_labels = {img: label for img, label in zip(fold["train_images"], fold["train_labels"])}
        val_images = fold["val_images"]
        val_labels = {img: label for img, label in zip(fold["val_images"], fold["val_labels"])}

        train_dataset = FMRIDataGenerator(train_images, train_labels, preprocessed_dir)
        val_dataset = FMRIDataGenerator(val_images, val_labels, preprocessed_dir)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model with dropout
        model = CNNLSTM(num_classes=config.NUM_CLASSES, time_length=config.TIME_LENGTH, 
                       cnn_feature_dim=config.FEATURE_DIM, dropout=config.DROP_OUT).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if resume_from:
            checkpoint_path = f'/root/autodl-tmp/CNNLSTM/Project/checkpoints/checkpoint_fold_{fold_idx + 1}_epoch_{resume_from}.pth'
            start_epoch, history = load_checkpoint(checkpoint_path, model, optimizer)
        else:
            start_epoch = 0
            history = {
                'train_loss': [],
                'train_acc': [],
                'train_f1': [],
                'val_loss': [],
                'val_acc': [],
                'val_f1': []
            }

        try:
            for epoch in range(start_epoch, num_epochs):
                print(f"\nFold {fold_idx + 1}, Epoch {epoch + 1}/{num_epochs}")
                
                # Training phase
                train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
                
                # Validation phase
                val_metrics = validate(model, val_loader, criterion, device)

                # Update history
                history['train_loss'].append(train_metrics['loss'])
                history['train_acc'].append(train_metrics['accuracy'])
                history['train_f1'].append(train_metrics['f1'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                history['val_f1'].append(val_metrics['f1'])

                # Log metrics
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1']
                })

                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")

                # Save checkpoint if accuracy improved
                if val_metrics['accuracy'] > best_accuracy:
                    best_accuracy = val_metrics['accuracy']
                    save_checkpoint(model, optimizer, fold_idx, epoch, val_metrics, history)

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            save_checkpoint(model, optimizer, fold_idx, epoch, val_metrics, history)
            print("You can resume training later using the saved checkpoint.")
            raise

        # Save final model for this fold
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history
        }, f"model_fold_{fold_idx + 1}.pth")
        
        fold_results.append({
            "fold": fold_idx + 1,
            "val_loss": val_metrics['loss'],
            "val_acc": val_metrics['accuracy'],
            "val_f1": val_metrics['f1']
        })

    print("\nCross-Validation Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Val Loss = {result['val_loss']:.4f}, "
              f"Val Acc = {result['val_acc']:.2f}%, Val F1 = {result['val_f1']:.4f}")

    avg_val_acc = sum([result["val_acc"] for result in fold_results]) / len(fold_results)
    avg_val_f1 = sum([result["val_f1"] for result in fold_results]) / len(fold_results)
    print(f"\nAverage Validation Accuracy: {avg_val_acc:.2f}%")
    print(f"Average Validation F1 Score: {avg_val_f1:.4f}")

    wandb.finish()

if __name__ == "__main__":
    csv_path = config.CSV_PATH + config.FOLD_DATA_CSV
    preprocessed_dir = config.PREPROCESSED_DIR
    folds = load_folds_from_csv(csv_path)

    cross_validation_training(
        folds=folds,
        preprocessed_dir=preprocessed_dir,
        num_epochs=config.EPOCH,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LR
    )