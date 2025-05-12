import os
from src.pre_processing.pre_processing import prepare_dataloaders
from src.model.model_simple import SimpleNN
from src.model.model_BN import BNNet
from src.train.entrainement import train_model
from src.evaluation.evaluate import evaluate


def main():
    # Charger les données
    file_path = os.path.join("data", "voice.csv")
    train_loader, test_loader = prepare_dataloaders(file_path)

    # Obtenir la dimension d'entrée
    input_dim = next(iter(train_loader))[0].shape[1]

    # Liste des modèles à comparer
    models = {"SimpleNN": SimpleNN(input_dim), "BNNet": BNNet(input_dim)}

    results = {}

    for name, model in models.items():
        print(f"\n🔧 Entraînement du modèle : {name}")
        train_model(model, train_loader, epochs=30, learning_rate=0.01)
        acc = evaluate(model, test_loader)
        results[name] = acc
        print(f"✅ Accuracy de {name} : {acc:.2%}")

    # Résumé final
    print("\n📊 Résumé des performances :")
    for name, acc in results.items():
        print(f"- {name}: {acc:.2%}")


if __name__ == "__main__":
    main()
