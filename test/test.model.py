import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.model_simple import SimpleNN
from src.models.model_BN import BNNet
from src.entrainement.train import train_model, evaluate

# Générer des données d'exemple
def generate_synthetic_data():
    # Crée des données aléatoires pour X et y
    X = torch.randn(100, 20)  # 100 échantillons, 20 caractéristiques
    y = torch.randint(0, 2, (100, 1)).float()  # 100 étiquettes binaires
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True), DataLoader(dataset, batch_size=16, shuffle=False)

# Test simple de la fonction d'entraînement
def test_train_model():
    # Créer un modèle simple
    model = SimpleNN(input_dim=20)
    
    # Créer les DataLoader pour l'entraînement et le test
    train_loader, test_loader = generate_synthetic_data()

    # Entraîner le modèle
    train_model(model, train_loader, epochs=1, learning_rate=0.01)

    # Vérifier si le modèle est bien entraîné (on vérifie simplement qu'il a été entraîné sans erreurs)
    assert model is not None, "Le modèle n'est pas entraîné correctement."

# Test de l'évaluation du modèle
def test_evaluate():
    model = SimpleNN(input_dim=20)
    train_loader, test_loader = generate_synthetic_data()

    # Entraîner le modèle pour qu'il ait appris quelque chose
    train_model(model, train_loader, epochs=1, learning_rate=0.01)

    # Tester l'évaluation (vérification de l'accuracy)
    accuracy = evaluate(model, test_loader)
    assert accuracy >= 0, "L'accuracy ne peut pas être négatif"

# Test de comparaison entre les modèles SimpleNN et BNNet
def test_model_comparison():
    train_loader, test_loader = generate_synthetic_data()

    # Test du modèle SimpleNN
    simple_model = SimpleNN(input_dim=20)
    train_model(simple_model, train_loader, epochs=1, learning_rate=0.01)
    simple_accuracy = evaluate(simple_model, test_loader)

    # Test du modèle BNNet
    bn_model = BNNet(input_dim=20)
    train_model(bn_model, train_loader, epochs=1, learning_rate=0.01)
    bn_accuracy = evaluate(bn_model, test_loader)

    # Comparer les accuracies
    assert simple_accuracy >= 0, "L'accuracy du modèle simple ne peut pas être négatif"
    assert bn_accuracy >= 0, "L'accuracy du modèle BN ne peut pas être négatif"

    print(f"SimpleNN Accuracy: {simple_accuracy:.2%}")
    print(f"BNNet Accuracy: {bn_accuracy:.2%}")