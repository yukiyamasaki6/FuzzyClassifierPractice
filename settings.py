import os

# Model parameters
RULES_PER_INPUT = 3                                     # Number of fuzzy rules per input dimension
TUNE_M_V = False                                        # Whether to tune the mean and variance parameters

# Training parameters
SEED = 42                                               # Random seed
LEARNING_RATE = 0.1    
NUM_EPOCHS = 100    
N_SPLITS = 10                                            # Number of KFold splits

# Dataset
USE_SKLEARN_DATASET = True                              # Whether to use sklearn dataset(True) or custom dataset(False)
SKLEARN_DATASET_NAME = "iris"                            # Name of the dataset(iris, wine, or breast_cancer) from sklearn.datasets
LOCAL_DATASET_NAME = ""                        # Name of the dataset from data folder
DATASET_PATH = "data/" + LOCAL_DATASET_NAME + ".csv"    # Path to the dataset if not using sklearn dataset

# Output path
OUTPUT_DIR = f"output/"
if USE_SKLEARN_DATASET:
    OUTPUT_PATH = OUTPUT_DIR + SKLEARN_DATASET_NAME + ".csv"
else:
    OUTPUT_PATH = OUTPUT_DIR + LOCAL_DATASET_NAME + ".csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)