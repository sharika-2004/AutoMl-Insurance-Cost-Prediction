# ===============================
# AutoML Insurance Cost Prediction
# ===============================

# 1. Imports
import pandas as pd
from pycaret.regression import *

# 2. Load dataset
print("Loading dataset...")
df = pd.read_csv("insurance.csv")

# 3. Basic checks
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 4. AutoML Setup
print("\nInitializing PyCaret setup...")

reg_setup = setup(
    data=df,
    target="charges",
    verbose=False,     # replaces silent=True (PyCaret 3.x)
    session_id=42
)

# 5. Compare multiple regression models
print("\nComparing models...")
best_model = compare_models()

# 6. Display best model
print("\nBest Model Selected:")
print(best_model)

# 7. Tune the best model
print("\nTuning the best model...")
tuned_model = tune_model(best_model)

# 8. Finalize the model
print("\nFinalizing model...")
final_model = finalize_model(tuned_model)

# 9. Save the trained model
print("\nSaving model...")
save_model(final_model, "insurance_automl_model")

print("\n✅ AutoML pipeline completed successfully!")