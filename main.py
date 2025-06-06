import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import sqlalchemy
from sqlalchemy import create_engine
import warnings
import os
warnings.filterwarnings('ignore')

# Check for CUDA availability and set device (cpu works fine too, just takes longer)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Database connection (sql lab database)
# You will have to set this up based on your own database specifications
user = 'arna'
password = input('Enter password: ')
host = '127.0.0.1'
port = 3306
database = 'localhost'
engine = sqlalchemy.create_engine(f"mysql+pymysql://arna:{password}@127.0.0.1:3306/lab_result")

class TestDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class TestDurationPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super(TestDurationPredictor, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def load_and_prepare_data():
    """Load data from database and prepare for training"""

    # Get all data and filter in pandas to avoid MySQL datetime issues (not sure why it gives incorrect value errors without this)
    query = """
            SELECT
                tr.added_date as request_date,
                trd.finalized_date,
                trd.test_type,
                tr.sample_type,
                trd.status
            FROM tbl_test_request tr
                     JOIN tbl_test_request_detail trd ON tr.id = trd.test_request_id
            WHERE trd.status = 2 \
            """

    print("Loading data from database...")
    df = pd.read_sql(query, engine)

    print(f"Initial records loaded: {len(df)}")

    # Filter out invalid dates in pandas (for the null entries)
    df = df[
        (df['finalized_date'].notna()) &
        (df['request_date'].notna()) &
        (df['finalized_date'] != '0000-00-00 00:00:00') &
        (df['request_date'] != '0000-00-00 00:00:00') &
        (df['finalized_date'] != '') &
        (df['request_date'] != '')
        ]

    print(f"Records after filtering invalid dates: {len(df)}")

    # Convert dates with error handling (cleans up model training data)
    try:
        df['request_date'] = pd.to_datetime(df['request_date'], errors='coerce')
        df['finalized_date'] = pd.to_datetime(df['finalized_date'], errors='coerce')
    except Exception as e:
        print(f"Date conversion error: {e}")
        # If conversion fails, try alternative approach
        df['request_date'] = pd.to_datetime(df['request_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['finalized_date'] = pd.to_datetime(df['finalized_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Removr rows where date conversion failed
    df = df.dropna(subset=['request_date', 'finalized_date'])
    print(f"Records after date conversion: {len(df)}")

    # Calculate duration in hours
    df['duration_hours'] = (df['finalized_date'] - df['request_date']).dt.total_seconds() / 3600

    # Remove negative durations and zero durations
    df = df[df['duration_hours'] > 0]
    print(f"Records after removing invalid durations: {len(df)}")

    # Remove extreme outliers (beyond 99th percentile) (it would mess up the data a lil bit)
    if len(df) > 0:
        duration_99th = df['duration_hours'].quantile(0.99)
        df = df[df['duration_hours'] <= duration_99th]
        print(f"Records after removing outliers: {len(df)}")
    else:
        print("No valid records found after filtering!")

    print(f"Loaded {len(df)} records")
    if len(df) > 0:
        print(f"Duration range: {df['duration_hours'].min():.2f} - {df['duration_hours'].max():.2f} hours")
    else:
        print("No valid data available for training!")
        return None

    return df

def prepare_features(df):
    """Prepare features for training"""

    # Create encoders for categorical variables
    test_type_encoder = LabelEncoder()
    sample_type_encoder = LabelEncoder()

    # Fit encoders and transform data
    df['test_type_encoded'] = test_type_encoder.fit_transform(df['test_type'].astype(str))
    df['sample_type_encoded'] = sample_type_encoder.fit_transform(df['sample_type'].astype(str))

    # Create additional time-based features (found these on stack overflow)
    df['hour_of_day'] = df['request_date'].dt.hour
    df['day_of_week'] = df['request_date'].dt.dayofweek
    df['month'] = df['request_date'].dt.month

    # Create interaction features
    df['test_sample_interaction'] = df['test_type_encoded'] * df['sample_type_encoded']

    # Select features
    feature_columns = [
        'test_type_encoded', 'sample_type_encoded', 'hour_of_day',
        'day_of_week', 'month', 'test_sample_interaction'
    ]

    X = df[feature_columns].values
    y = df['duration_hours'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Unique test types: {df['test_type'].nunique()}")
    print(f"Unique sample types: {df['sample_type'].nunique()}")

    return X_scaled, y, scaler, test_type_encoder, sample_type_encoder, df

def train_model(X, y):
    """Train the PyTorch model"""

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create datasets and dataloaders
    train_dataset = TestDataset(X_train, y_train)
    val_dataset = TestDataset(X_val, y_val)
    test_dataset = TestDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Initialize model and move to GPU
    model = TestDurationPredictor(input_size=X.shape[1], hidden_sizes=[256, 128, 64, 32]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # start training teh model
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    print("Starting training...")

    for epoch in range(200):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            # Move data to gpu/cpu
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                # Move data to cpu/gpu
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping (helps to prevent overfitting, and saves training time)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch}/200, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= 25:  # Early stopping
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            # Move data to GPU
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_features).squeeze()
            test_predictions.extend(outputs.cpu().numpy())  # Move back to CPU for numpy
            test_actuals.extend(batch_targets.cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)

    # Calculate metrics
    mae = mean_absolute_error(test_actuals, test_predictions)
    mse = mean_squared_error(test_actuals, test_predictions)
    rmse = np.sqrt(mse)

    print(f"\nFinal Test Results:")
    print(f"Mean Absolute Error: {mae:.2f} hours")
    print(f"Root Mean Square Error: {rmse:.2f} hours")

    # Plot results (only made 3 plots, the training data and residuals on stack overflow)
    plt.figure(figsize=(15, 5))

    # Training history
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Predictions vs Actual
    plt.subplot(1, 3, 2)
    plt.scatter(test_actuals, test_predictions, alpha=0.6, s=20)
    plt.plot([test_actuals.min(), test_actuals.max()],
             [test_actuals.min(), test_actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Duration (hours)')
    plt.ylabel('Predicted Duration (hours)')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)

    # Residuals
    plt.subplot(1, 3, 3)
    residuals = test_predictions - test_actuals
    plt.scatter(test_predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Duration (hours)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model, scaler, test_predictions, test_actuals

def analyze_data(df):
    """Analyze the data patterns"""
    print("\nData Analysis:")
    print("=" * 50)

    # Overall statistics (final results)
    print(f"Duration Statistics:")
    print(f"Mean: {df['duration_hours'].mean():.2f} hours")
    print(f"Median: {df['duration_hours'].median():.2f} hours")
    print(f"Std: {df['duration_hours'].std():.2f} hours")

    # By test type
    print(f"\nAverage Duration by Test Type:")
    test_analysis = df.groupby('test_type')['duration_hours'].agg(['mean', 'std', 'count']).round(2)
    print(test_analysis)

    # By sample type
    print(f"\nAverage Duration by Sample Type:")
    sample_analysis = df.groupby('sample_type')['duration_hours'].agg(['mean', 'std', 'count']).round(2)
    print(sample_analysis)

    # By test type and sample type
    print(f"\nAverage Duration by Test Type and Sample Type:")
    combo_analysis = df.groupby(['test_type', 'sample_type'])['duration_hours'].agg(['mean', 'std', 'count']).round(2)
    print(combo_analysis)

def predict_duration(model, scaler, test_type_encoder, sample_type_encoder,
                     test_type, sample_type, hour_of_day=12, day_of_week=1, month=1):
    """Make prediction for new test"""
    try:
        # Encode categorical variables
        test_type_encoded = test_type_encoder.transform([str(test_type)])[0]
        sample_type_encoded = sample_type_encoder.transform([str(sample_type)])[0]

        # Create features
        features = np.array([[
            test_type_encoded, sample_type_encoded, hour_of_day,
            day_of_week, month, test_type_encoded * sample_type_encoded
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            prediction = model(features_tensor).cpu().item()  # Move back to CPU

        return max(0, prediction)  # Ensure non-negative prediction

    except ValueError as e:
        return f"Error: Unknown test_type or sample_type. {str(e)}"

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_and_prepare_data()

    if df is None or len(df) == 0:
        print("No data available for training. Please check your database.")
        exit(1)

    X, y, scaler, test_type_encoder, sample_type_encoder, df_processed = prepare_features(df)

    # Analyze data
    analyze_data(df_processed)

    # Train model
    model, scaler, test_predictions, test_actuals = train_model(X, y)

    # Example predictions
    print("\nBatch Predictions:")
    print("=" * 50)

    # Get unique values for examples
    unique_test_types = df['test_type'].unique()[:3]
    unique_sample_types = df['sample_type'].unique()[:3]

    for test_type in unique_test_types:
        for sample_type in unique_sample_types:
            pred = predict_duration(model, scaler, test_type_encoder, sample_type_encoder,
                                    test_type, sample_type)
            if isinstance(pred, str):
                print(pred)
            else:
                print(f"Test Type: {test_type}, Sample Type: {sample_type} -> {pred:.2f} hours")

    # Save model and encoders with better error handling
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Save to models directory
        model_path = os.path.join('models', 'test_duration_model.pth')

        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'test_type_encoder': test_type_encoder,
            'sample_type_encoder': sample_type_encoder,
            'model_architecture': {
                'input_size': X.shape[1],
                'hidden_sizes': [256, 128, 64, 32]
            }
        }, model_path)

        print(f"\nModel saved as '{model_path}'")

    except Exception as e:
        print(f"\nError saving model: {e}")
        # Try saving in current directory if it doesnt work (YOU NEED TO SAVE TIHS FOR THE PREDICTION FUNCTION)
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'test_type_encoder': test_type_encoder,
                'sample_type_encoder': sample_type_encoder,
                'model_architecture': {
                    'input_size': X.shape[1],
                    'hidden_sizes': [256, 128, 64, 32]
                }
            }, 'model_backup.pth')
            print("Model saved as 'model_backup.pth' in current directory")
        except Exception as e2:
            print(f"Failed to save model: {e2}")

    print("\nTraining completed successfully!")

print("starting predictions")

# Prediction Function (can use either after using training program and then importing training file, or directly after training function)
def predict_duration_now(test_type, sample_type, hour_of_day=12, day_of_week=1, month=1):
    try:
        # Encode categorical variables
        test_type_encoded = test_type_encoder.transform([str(test_type)])[0]
        sample_type_encoded = sample_type_encoder.transform([str(sample_type)])[0]

        # Create features
        features = np.array([[
            test_type_encoded, sample_type_encoded, hour_of_day,
            day_of_week, month, test_type_encoded * sample_type_encoded
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            prediction = model(features_tensor).cpu().item()

        return max(0, prediction)  # Ensure non-negative prediction

    except ValueError as e:
        return f"Error: {str(e)}"

# Show available options for user reference from sql columns
print("\nAvailable Test Types:")
available_test_types = df_processed['test_type'].unique()
for i, test_type in enumerate(available_test_types):
    print(f"  {i+1}. {test_type}")

print("\nAvailable Sample Types:")
available_sample_types = df_processed['sample_type'].unique()
for i, sample_type in enumerate(available_sample_types):
    print(f"  {i+1}. {sample_type}")

# Prediction function loop with simple ui (so you dont have to run the training each time)
print("\n" + "="*50)
print("PREDICTION MODE")
print("="*50)
print("Enter test details to get duration predictions.")
print("Type 'quit' or 'exit' to stop.")
print("-" * 50)

while True:
    try:
        # Get test type
        test_type = input("\nEnter test type: ").strip()
        if test_type.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        # Get sample type
        sample_type = input("Enter sample type: ").strip()
        if sample_type.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        # Get optional parameters with defaults
        hour_input = input("Enter hour of day (0-23) [default: 12]: ").strip()
        hour_of_day = int(hour_input) if hour_input and hour_input.isdigit() else 12

        day_input = input("Enter day of week (0=Monday, 6=Sunday) [default: 1]: ").strip()
        day_of_week = int(day_input) if day_input and day_input.isdigit() else 1

        month_input = input("Enter month (1-12) [default: 1]: ").strip()
        month = int(month_input) if month_input and month_input.isdigit() else 1

        # Make prediction
        prediction = predict_duration_now(test_type, sample_type, hour_of_day, day_of_week, month)

        print(f"\n{'='*40}")
        if isinstance(prediction, str):
            print(f" {prediction}")
        else:
            print(f" Predicted Duration: {prediction:.2f} hours")
            print(f"   Test Type: {test_type}")
            print(f"   Sample Type: {sample_type}")
            print(f"   Time: Hour {hour_of_day}, Day {day_of_week}, Month {month}")
        print(f"{'='*40}")

        # Ask if user wants to continue
        continue_input = input("\nMake another prediction? (y/n) [default: y]: ").strip().lower()
        if continue_input in ['n', 'no']:
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again.")
