import joblib
import numpy as np

def predict(sample_sequence):
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")

    sample = np.array(sample_sequence).reshape(1, -1)

    pred = model.predict(sample)

    return le.inverse_transform(pred)

# test
if __name__ == "__main__":
    sample = [0,12,2]*5
    print("Predicted vibe:", predict(sample))