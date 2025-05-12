import os
import torch
from src.pre_processing.pre_processing import prepare_dataloaders
from src.models.model_simple import SimpleNN
# from src.models.model_BN import BNNet  # Décommente si tu veux tester celui-ci
from src.entrainement.train import train_model

def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            predicted = (preds > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")

def main():
    # 1. Charger les données
    file_path = os.path.join("data", "voice.csv")
    train_loader, test_loader = prepare_dataloaders(file_path)

    # 2. Choisir un modèle
    input_dim = next(iter(train_loader))[0].shape[1]
    model = SimpleNN(input_dim)
    # model = BNNet(input_dim)  # ou utiliser le modèle avec BN

    # 3. Entraîner
    train_model(model, train_loader, epochs=30, learning_rate=0.01)

    # 4. Évaluer
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
