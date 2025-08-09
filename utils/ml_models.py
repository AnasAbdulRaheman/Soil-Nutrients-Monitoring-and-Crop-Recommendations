# utils/ml_models.py

import os
import pickle

def load_random_forest_model():
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'random_forest_model.pkl')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("✅ RandomForest model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"❌ RandomForest_reduced.pkl file not found at {model_path}")
    except Exception as e:
        print(f"❌ Error loading RandomForest model: {e}")
    return None