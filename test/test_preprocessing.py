import torch
import pandas as pd
from src.pre_processing.pre_processing import prepare_dataloaders

def test_data_preprocessing():
    # Création d'un fichier CSV fictif pour les tests
    data = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [5.0, 4.0, 3.0, 2.0, 1.0],
        'label': ['male', 'female', 'male', 'female', 'male']
    }
    df = pd.DataFrame(data)
    df['label'] = df['label'].map({'male': 1, 'female': 0})
    
    # Sauver les données en CSV temporairement
    file_path = 'data/voice.csv'
    df.to_csv(file_path, index=False)

    # Tester la fonction de pré-processing
    train_loader, test_loader = prepare_dataloaders(file_path)

    # Vérifier que les DataLoaders ne sont pas vides
    assert len(train_loader) > 0, "Le DataLoader d'entraînement est vide"
    assert len(test_loader) > 0, "Le DataLoader de test est vide"

    print("Pré-processing des données réussi !")
