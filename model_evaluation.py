import pandas as pd
from pycaret.regression import *

# Load dataset
data = pd.read_csv("insurance.csv")

# Load trained model
model = load_model("insurance_automl_model")

# Recreate PyCaret environment
setup(
    data=data,
    target="charges",
    session_id=42,
    verbose=False
)

# Evaluate model performance (interactive dashboard)
evaluate_model(model)

# Generate predictions
predictions = predict_model(model, data=data)

# Save predictions
predictions.to_csv("insurance_model_predictions.csv", index=False)

# Feature importance
plot_model(model, plot="feature")

# Residual analysis
plot_model(model, plot="residuals")

# Error distribution
plot_model(model, plot="error")