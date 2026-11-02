import pytest
import numpy as np
import tensorflow as tf
from app.ml.models import build_lstm_model, build_gru_model, train_single_model, evaluate_model_on_splits
from sklearn.preprocessing import MinMaxScaler

def test_lstm_architecture_and_output_shape():
    model = build_lstm_model(input_shape=(24, 1))
    assert model.name == "PowerPulse_LSTM"
    # Verify layers
    layer_names = [l.name for l in model.layers]
    assert any("lstm" in name.lower() for name in layer_names)
    assert any("dense" in name.lower() for name in layer_names)
    
    # Dummy forward pass
    dummy_input = np.random.normal(size=(4, 24, 1)).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (4, 1)

def test_gru_architecture_and_output_shape():
    model = build_gru_model(input_shape=(24, 1))
    assert model.name == "PowerPulse_GRU"
    layer_names = [l.name for l in model.layers]
    assert any("gru" in name.lower() for name in layer_names)
    assert any("dense" in name.lower() for name in layer_names)
    
    dummy_input = np.random.normal(size=(4, 24, 1)).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (4, 1)

def test_small_end_to_end_training():
    # Fast 1-epoch training on synthetic sequences to verify convergence and evaluation
    np.random.seed(42)
    N = 40
    X = np.random.uniform(0.1, 0.9, size=(N, 24, 1)).astype(np.float32)
    y = np.random.uniform(0.1, 0.9, size=(N, 1)).astype(np.float32)
    
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0.0], [5.0]]))

    splits = {
        "X_train": X[:25],
        "y_train": y[:25],
        "X_val": X[25:32],
        "y_val": y[25:32],
        "X_test": X[32:],
        "y_test": y[32:],
        "ts_test": [f"2025-01-01 {i:02d}:00:00" for i in range(len(X[32:]))]
    }

    model, info = train_single_model(
        "lstm", splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"],
        epochs=1, batch_size=16
    )
    assert info["epochs_trained"] == 1
    assert "loss_history" in info

    metrics = evaluate_model_on_splits(model, "LSTM", scaler, splits, info)
    assert metrics["val_mae"] > 0
    assert metrics["test_mae"] > 0
    assert metrics["test_rmse"] > 0
    assert "inference_latency_ms" in metrics
