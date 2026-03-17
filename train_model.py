from preprocessing import load_and_merge
from sequence_model import create_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train():
    df, le = load_and_merge()

    X, y = create_sequences(df)

    # Flatten for ML model
    X = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print("Accuracy:", model.score(X_test, y_test))

    joblib.dump(model, "model.pkl")
    joblib.dump(le, "label_encoder.pkl")

if __name__ == "__main__":
    train()