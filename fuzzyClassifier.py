import settings
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score

def write_hyperparameters_to_csv():
    filename = settings.OUTPUT_PATH
    with open(filename.replace(".csv", "_setting.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model Parameters"])
        writer.writerow(["Rules Per Input", settings.RULES_PER_INPUT])
        writer.writerow(["Tune M_V", settings.TUNE_M_V])
        writer.writerow(["Training Parameters"])
        writer.writerow(["Seed", settings.SEED])
        writer.writerow(["Learning Rate", settings.LEARNING_RATE])
        writer.writerow(["Num Epochs", settings.NUM_EPOCHS])
        writer.writerow(["N Splits", settings.N_SPLITS])
        writer.writerow(["Dataset"])
        if settings.USE_SKLEARN_DATASET:
            writer.writerow(["Dataset Name", settings.SKLEARN_DATASET_NAME])
        else:
            writer.writerow(["Dataset Name", settings.LOCAL_DATASET_NAME])

def write_evaluation_to_csv(
        mean_accuracy,
        weights,
        feature_names,
        target_names,
        rules_per_input,
        num_inputs
        ):
    filename = settings.OUTPUT_PATH
    
    with open(filename.replace(".csv", "_result.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Mean Accuracy", mean_accuracy])
        writer.writerow([])

        # Write weights
        writer.writerow(["Weights"])
        if not isinstance(target_names, list):
            target_names = target_names.tolist()
        
        # Header: Rule combinations for each class
        total_rules = rules_per_input ** num_inputs
        writer.writerow(["Rule_Index"] + target_names)
        
        # Write weights for each rule combination
        for rule_idx in range(total_rules):            
            rule_name = f"Rule_{rule_idx}"
            writer.writerow([rule_name] + weights[rule_idx].tolist())
        writer.writerow([])

def load_custom_dataset(filename):
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        data = np.array([row for row in reader], dtype=float)
        np.random.shuffle(data)  # Shuffle the data

        X = data[:, :-1]
        y = data[:, -1].astype(int)

        num_samples = X.shape[0]
        num_features = X.shape[1]
        num_classes = len(np.unique(y))

        return X, y, num_features, num_classes

def get_dataset():
    if settings.USE_SKLEARN_DATASET:
        if settings.SKLEARN_DATASET_NAME == "iris":
            dataset = load_iris()
        elif settings.SKLEARN_DATASET_NAME == "breast_cancer":
            dataset = load_breast_cancer()
        elif settings.SKLEARN_DATASET_NAME == "wine":
            dataset = load_wine()
        else:
            raise ValueError(f"Unknown sklearn dataset: {settings.SKLEARN_DATASET_NAME}")
        X = dataset.data
        y = dataset.target
        feature_names = dataset.feature_names
        target_names = dataset.target_names
        input_dim = X.shape[1]
        classes = len(np.unique(y))
    else:
        X, y, input_dim, classes = load_custom_dataset(settings.DATASET_PATH)
        feature_names = [f"Feature_{i+1}" for i in range(input_dim)]
        target_names = [f"Class_{i}" for i in range(classes)]
    
    return X, y, input_dim, classes, feature_names, target_names

class FuzzyClassifier(nn.Module):
    def __init__(self, num_inputs, num_rules_per_input, num_classes, tune_m_v=False):
        super(FuzzyClassifier, self).__init__()

        # Store dimensions for reference
        self.num_inputs = num_inputs
        self.num_rules_per_input = num_rules_per_input
        self.num_classes = num_classes

        # Parameters for Gaussian membership functions for each input dimension
        self.mean = nn.Parameter(torch.linspace(0, 1, num_rules_per_input), requires_grad=tune_m_v)
        interval = 1.0 / (num_rules_per_input - 1)
        self.variance = nn.Parameter(torch.full((num_rules_per_input,), (interval/2)**2 / (2 * torch.abs(torch.log(torch.tensor(0.5))))), requires_grad=tune_m_v)
        
        # Weights for each combined rule and class
        total_rules = num_rules_per_input ** num_inputs
        self.weights = nn.Parameter(torch.rand(total_rules, num_classes))
    
    def get_membership(self, x):
        """
        各ルールのメンバーシップ値を計算
        各入力次元のルールのメンバーシップ値の積で全体のルールのメンバーシップ値を求める
        
        Args:
            x: [batch_size, num_inputs]
        
        Returns:
            membership: [batch_size, num_rules_per_input**num_inputs]
        """
        batch_size = x.size(0)

        # Calculate membership for each input dimension
        membership_per_dim = []
        for i in range(self.num_inputs):
            x_i = x[:, i].unsqueeze(1)  # [batch_size, 1]
            # Gaussian membership function for dimension i
            gaussian_numerator = -(x_i - self.mean) ** 2  # [batch_size, num_rules_per_input]
            gaussian_denominator = 2 * self.variance  # [num_rules_per_input]
            membership_i = torch.exp(gaussian_numerator / gaussian_denominator)  # [batch_size, num_rules_per_input]
            membership_per_dim.append(membership_i)

        # Compute Cartesian product of memberships across all input dimensions
        # 例: X1={S,M,L}, X2={S,M,L} → 9個のルール: SS, SM, SL, MS, MM, ML, LS, LM, LL
        membership = membership_per_dim[0]  # [batch_size, num_rules_per_input]
        
        for i in range(1, self.num_inputs):
            # membership: [batch_size, current_total_rules]
            # membership_per_dim[i]: [batch_size, num_rules_per_input]
            membership = membership.unsqueeze(2) * membership_per_dim[i].unsqueeze(1)
            # Reshape to flatten the new dimension
            membership = membership.view(batch_size, -1)

        return membership

    def forward(self, x):
        membership = self.get_membership(x)  # [batch_size, num_rules**num_inputs]

        # Compute final output
        output = torch.matmul(membership, self.weights)  # [batch_size, num_classes]

        # Apply Softmax to get class probabilities
        output_softmax = nn.Softmax(dim=1)(output)
        return output_softmax

    def compute_loss(self, predictions, target):
        # Compute cross-entropy loss for classification tasks
        return nn.CrossEntropyLoss()(predictions, target)

def training_FuzzyClassifier(num_epochs, classes, X_train, y_train, model, optimizer, device):
    model.train()
    for epoch in range(num_epochs):
        sum_loss = 0
        for i in range(X_train.size(0)):
            # Step 1: Forward pass - compute model predictions
            x_single = X_train[i].unsqueeze(0)  # Add batch dimension
            y_single = y_train[i].unsqueeze(0)  # Add batch dimension

            predictions = model(x_single)

            # Step 2: Compute loss
            loss = model.compute_loss(predictions, y_single)  # Add batch dimension
            sum_loss += loss.item()
            
            # Step 3: Backward pass - compute gradients
            optimizer.zero_grad()
            loss.backward()

            # Step 4: Update model parameters
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average loss: {sum_loss/X_train.size(0):.4f}')

def compute_accuracy(model, X_test, y_test):
    """
    Accuracy: モデルの予測精度を計算
    """
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        _, predicted_classes = torch.max(test_predictions, 1)
        accuracy = accuracy_score(
            y_test.cpu().numpy(), predicted_classes.cpu().numpy()
        )
    return accuracy

def fuzzyClassifierMain():
    # Get hyperparameters
    rules_per_input = settings.RULES_PER_INPUT
    learning_rate = settings.LEARNING_RATE
    num_epochs = settings.NUM_EPOCHS
    tune_m_v = settings.TUNE_M_V
    n_splits = settings.N_SPLITS
    seed = settings.SEED
    # Load dataset
    X, y, input_dim, classes, feature_names, target_names = get_dataset()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Scale the features to [0, 1]    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if n_splits > 1:
        # K-Fold Cross Validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kf.split(X)
    else:
        # Single Train/Test split (e.g., 80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        splits = [(
            np.arange(len(X_train)),  # train indices
            np.arange(len(X_train), len(X_train) + len(X_test))  # test indices (shifted)
        )]
        # Concatenate back for indexing convenience
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

    accuracies = []

    for fold, (train_index, test_index) in enumerate(splits):
        print(f"Fold: {fold + 1}/{n_splits}")

        # Split the data into training and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)  # Change to long for one-hot encoding
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)  # Change to long for one-hot encoding

        # Initialize the Fuzzy Classifier model
        model = FuzzyClassifier(
            num_inputs=input_dim,
            num_rules_per_input=rules_per_input,
            num_classes=classes,
            tune_m_v=tune_m_v
        ).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        training_FuzzyClassifier(num_epochs, classes, X_train, y_train, model, optimizer, device)

        # Display predictions on training data
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Calculate accuracy
            accuracy = compute_accuracy(model, X_test, y_test)
            print(f'Accuracy: {accuracy:.4f}')
            accuracies.append(accuracy)

    # Calculate mean accuracy
    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f'Mean Accuracy: {mean_accuracy:.4f}')

    # Write evaluation results to CSV
    weights = model.weights.detach().cpu().numpy()
            
    write_evaluation_to_csv(
        mean_accuracy,
        weights,
        feature_names,
        target_names,
        rules_per_input,
        input_dim
    )
                
if __name__ == "__main__":
    # Run fuzzy classifier main
    fuzzyClassifierMain()

    # Write hyperparameters to CSV
    write_hyperparameters_to_csv()
