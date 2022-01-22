# #### Import library to read file and load to dataframe
import pandas as pd
# #### Import libraries for machine learning
# Import regression module
from pycaret.regression import *

# TARGET_VARIABLE='G3' #variable to be predicted
# NUMERICAL_FEATURES=["age", "G1", "G2"] #features to be interpreted as numeric
# Default values
INPUT_TRAIN_FILE="train.csv"
INPUT_TEST_FILE="pred.csv"
OUTPUT_PRED_FILE="output.csv"
OUTPUT_PRED="y_pred"
MODEL_FILE="model"

df_train=pd.read_csv(INPUT_TRAIN_FILE)
df_test=pd.read_csv(INPUT_TEST_FILE)

def train(target = 'G3', numeric_features=["age", "G1", "G2"]):
    print("Training process has begun and it might take a while based on the size of input dataset.")
    #intialize the setup
    exp_reg = setup(df_train, target=target, numeric_features=numeric_features, silent=True)#html=False)
    print("Finding the best model ...")
    best_model = compare_models()
    print("Finalising the model ...")
    final_model = finalize_model(best_model)
    save_model(final_model, MODEL_FILE)
    print(f"Saved model file: {MODEL_FILE}")

def pred():
    print(f"Loading model file: {MODEL_FILE}")
    loaded_model = load_model(MODEL_FILE)
    y_pred=loaded_model.predict(df_test)
    df_test[OUTPUT_PRED]=y_pred
    df_test.to_csv(OUTPUT_PRED_FILE)
    print(f"Output is written to the file: {OUTPUT_PRED_FILE}")