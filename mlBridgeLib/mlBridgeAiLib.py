import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import polars as pl
import pathlib
import time
from collections import defaultdict
import pickle
import os
from pathlib import Path
import logging
import json
from copy import deepcopy
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_to_log_info(*args):
    print_to_log(logging.INFO, *args)

def print_to_log_debug(*args):
    print_to_log(logging.DEBUG, *args)

def print_to_log(level, *args):
    logging.log(level, ' '.join(str(arg) for arg in args))

# --- PyTorch Dataset for Tabular Data ---
class TabularDataset(Dataset):
    def __init__(self, X_categorical, X_continuous, y):
        self.X_categorical = X_categorical
        self.X_continuous = X_continuous
        self.y = y

    def __len__(self):
        if torch.is_tensor(self.y) and self.y.ndim > 0:
            return self.y.size(0)
        elif torch.is_tensor(self.X_categorical) and self.X_categorical.ndim > 0 and self.X_categorical.size(0) > 0:
            return self.X_categorical.size(0)
        elif torch.is_tensor(self.X_continuous) and self.X_continuous.ndim > 0 and self.X_continuous.size(0) > 0:
            return self.X_continuous.size(0)
        return 0

    def __getitem__(self, idx):
        cat_item = self.X_categorical[idx] if torch.is_tensor(self.X_categorical) and self.X_categorical.numel() > 0 and self.X_categorical.size(0) > idx else torch.empty(0, dtype=torch.long)
        cont_item = self.X_continuous[idx] if torch.is_tensor(self.X_continuous) and self.X_continuous.numel() > 0 and self.X_continuous.size(0) > idx else torch.empty(0, dtype=torch.float32)
        label_item = self.y[idx] if torch.is_tensor(self.y) and self.y.numel() > 0 and self.y.size(0) > idx else torch.empty(0, dtype=torch.long)
        
        return {
            'categorical': cat_item,
            'continuous': cont_item,
            'labels': label_item
        }

# --- PyTorch Model Definition ---
class TabularNNModel(nn.Module):
    def __init__(self, embedding_sizes, n_continuous, n_classes, layers, p_dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_embeddings = sum(e.embedding_dim for e in self.embeddings)
        self.n_continuous = n_continuous
        
        all_layers = []
        input_size = n_embeddings + n_continuous
        
        for i, layer_size in enumerate(layers):
            all_layers.append(nn.Linear(input_size, layer_size))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(layer_size))
            all_layers.append(nn.Dropout(p_dropout))
            input_size = layer_size
            
        all_layers.append(nn.Linear(layers[-1], n_classes))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_continuous):
        x_embeddings = []
        for i, e in enumerate(self.embeddings):
            x_embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(x_embeddings, 1)
        
        if self.n_continuous > 0:
            if x_continuous.ndim == 1:
                x_continuous = x_continuous.unsqueeze(0)
            x = torch.cat([x, x_continuous], 1)
            
        x = self.layers(x)
        return x

# Function to calculate the total input size (similar to fastai version)
def calculate_input_size(embedding_sizes, n_continuous):
    total_embedding_size = sum(size for _, size in embedding_sizes)
    return total_embedding_size + n_continuous

# Function to define optimal layer sizes (similar to fastai version)
def define_layer_sizes(input_size, num_layers=3, shrink_factor=2):
    layer_sizes = [input_size]
    for i in range(1, num_layers):
        layer_sizes.append(layer_sizes[-1] // shrink_factor)
    return layer_sizes

# create a test set using date and sample size. current default is 10k samples ge 2024-07-01.
def split_by_date(df, include_dates):
    include_date = datetime.strptime(include_dates, '%Y-%m-%d') # i'm not getting why datetime.datetime.strptime isn't working here but the only thing that works elsewhere?
    date_filter = df['Date'] >= include_date
    return df.filter(~date_filter), df.filter(date_filter)

def get_device():
    """
    Get the best available device (GPU if available, otherwise CPU).
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        print_to_log_info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Log current GPU memory usage
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print_to_log_info(f"GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        device = 'cpu'
        print_to_log_info("No GPU available, using CPU")
    return device

def train_model(df, y_names, cat_names=None, cont_names=None, nsamples=None, 
                procs=None, valid_pct=0.2, bs=1024*10, layers=None, epochs=3, 
                device=None, y_range=(0,1), lr=1e-3, patience=3, min_delta=0.001, seed=42):
    """
    Train a PyTorch tabular model similar to the fastai implementation.
    
    Args:
        df: Polars DataFrame with the data
        y_names: List with single target column name
        cat_names: List of categorical column names (auto-detected if None)
        cont_names: List of continuous column names (auto-detected if None)
        nsamples: Number of samples to use (None for all)
        procs: Processing steps (ignored for PyTorch, kept for compatibility)
        valid_pct: Validation split percentage
        bs: Batch size
        layers: List of layer sizes (auto-calculated if None)
        epochs: Number of training epochs
        device: Device to train on ('cpu', 'cuda', or None for auto-detection)
        y_range: Range for regression targets (ignored for classification)
        lr: Learning rate
        patience: Early stopping patience
        min_delta: Minimum delta for early stopping
    
    Returns:
        Dictionary containing model, artifacts, and training info
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    t = time.time()
    print_to_log_info(f"{y_names=} {cat_names=} {cont_names} {valid_pct=} {bs=} {layers=} {epochs=} {device=}")

    # Validate inputs
    assert isinstance(y_names, list) and len(y_names) == 1, 'Only one target variable is supported.'
    
    print(df.describe())
    unimplemented_dtypes = df.select(pl.exclude(pl.Boolean,pl.Categorical,pl.Int8,pl.Int16,pl.Int32,pl.Int64,pl.Float32,pl.Float64,pl.String,pl.UInt8,pl.Utf8)).columns
    print(f"{unimplemented_dtypes=}")

    # Setup categorical and continuous column names
    if cat_names is None:
        cat_names = list(set(df.select(pl.col([pl.Boolean,pl.Categorical,pl.String])).columns).difference(y_names))
    print(f"{cat_names=}")
    
    if cont_names is None:
        cont_names = list(set(df.columns).difference(cat_names + y_names))
    print(f"{cont_names=}")
    
    assert set(y_names).intersection(cat_names+cont_names) == set(), set(y_names).intersection(cat_names+cont_names)
    assert set(cat_names).intersection(cont_names) == set(), set(cat_names).intersection(cont_names)

    # Sample data if requested. If nsamples is None, use all data.
    if nsamples is None:
        pandas_df = df[y_names+cat_names+cont_names].to_pandas()
    else:
        pandas_df = df[y_names+cat_names+cont_names].sample(nsamples, seed=seed).to_pandas()

    print('y_names[0].dtype:', pandas_df[y_names[0]].dtype.name)
    
    # Determine if this is classification or regression
    is_classification = pandas_df[y_names[0]].dtype.name in ['boolean','category','object','string','uint8']
    
    if is_classification:
        return train_classifier_pytorch(pandas_df, y_names, cat_names, cont_names, 
                                      valid_pct=valid_pct, bs=bs, layers=layers, 
                                      epochs=epochs, device=device, lr=lr, 
                                      patience=patience, min_delta=min_delta)
    else:
        return train_regression_pytorch(pandas_df, y_names, cat_names, cont_names, 
                                      valid_pct=valid_pct, bs=bs, layers=layers, 
                                      epochs=epochs, device=device, lr=lr, 
                                      patience=patience, min_delta=min_delta, y_range=y_range)

def train_classifier_pytorch(df, y_names, cat_names, cont_names, valid_pct=0.2, 
                           bs=1024*5, layers=None, epochs=3, device=None, 
                           lr=1e-3, patience=3, min_delta=0.001):
    """Train a classification model using PyTorch."""
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    t = time.time()
    
    # Preprocessing
    artifacts = {
        'target_encoder': LabelEncoder(),
        'categorical_encoders': {col: LabelEncoder() for col in cat_names},
        'continuous_scalers': {col: StandardScaler() for col in cont_names},
        'na_fills': {},
        'categorical_feature_names': cat_names,
        'continuous_feature_names': cont_names,
        'target_name': y_names[0],
        'model_params': {},
        'training_history': {},
        'is_classification': True
    }

    # Target encoding
    df[y_names[0]] = df[y_names[0]].astype(str)
    artifacts['target_encoder'].fit(df[y_names[0]])
    df[y_names[0]] = artifacts['target_encoder'].transform(df[y_names[0]])
    n_classes = len(artifacts['target_encoder'].classes_)
    print_to_log_info(f"Target '{y_names[0]}' encoded. Number of classes: {n_classes}")

    # Categorical feature encoding and NA handling
    for col in cat_names:
        df[col] = df[col].astype(str)
        fill_val_cat = "MISSING_CAT"
        artifacts['na_fills'][col] = fill_val_cat

        unique_values_for_fit = pd.unique(df[col].fillna(fill_val_cat)).tolist()
        if fill_val_cat not in unique_values_for_fit:
            unique_values_for_fit.append(fill_val_cat)
        
        artifacts['categorical_encoders'][col].fit(unique_values_for_fit)
        df[col] = artifacts['categorical_encoders'][col].transform(df[col].fillna(fill_val_cat))

    # Continuous feature scaling and NA handling
    for col in cont_names:
        fill_val_cont = df[col].median()
        artifacts['na_fills'][col] = fill_val_cont
        df[col] = df[col].fillna(fill_val_cont)
        df[col] = artifacts['continuous_scalers'][col].fit_transform(df[col].values.reshape(-1, 1)).flatten()

    # Calculate embedding sizes
    embedding_sizes = []
    for col in cat_names:
        num_categories = len(artifacts['categorical_encoders'][col].classes_)
        embed_dim = max(4, min(300, int(num_categories / 2)))
        embedding_sizes.append((num_categories, embed_dim))

    artifacts['embedding_sizes'] = embedding_sizes
    n_continuous = len(cont_names)

    # Calculate input size and layer sizes
    input_size = calculate_input_size(embedding_sizes, n_continuous)
    if layers is None:
        layers = define_layer_sizes(input_size)
    print(f"Input size: {input_size}, Layer sizes: {layers}")

    # Prepare data
    X_cat = torch.tensor(df[cat_names].values, dtype=torch.long) if cat_names else torch.empty((len(df), 0), dtype=torch.long)
    X_cont = torch.tensor(df[cont_names].values, dtype=torch.float32) if cont_names else torch.empty((len(df), 0), dtype=torch.float32)
    y = torch.tensor(df[y_names[0]].values, dtype=torch.long)

    # Train/validation split - handle stratification issues
    try:
        # Try stratified split first
        train_idx, val_idx = train_test_split(range(len(df)), test_size=valid_pct, random_state=42, stratify=y)
        print_to_log_info("Using stratified train/validation split")
    except ValueError as e:
        if "least populated class" in str(e):
            # Fall back to regular split if stratification fails
            print_to_log_info("Stratified split failed due to classes with too few samples. Using regular split.")
            train_idx, val_idx = train_test_split(range(len(df)), test_size=valid_pct, random_state=42)
        else:
            raise e
    
    train_dataset = TabularDataset(X_cat[train_idx], X_cont[train_idx], y[train_idx])
    val_dataset = TabularDataset(X_cat[val_idx], X_cont[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    # Create model
    model = TabularNNModel(embedding_sizes, n_continuous, n_classes, layers)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch['categorical'].to(device), batch['continuous'].to(device))
            loss = criterion(outputs, batch['labels'].to(device))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch['labels'].size(0)
            train_correct += (predicted == batch['labels'].to(device)).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['categorical'].to(device), batch['continuous'].to(device))
                loss = criterion(outputs, batch['labels'].to(device))
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch['labels'].size(0)
                val_correct += (predicted == batch['labels'].to(device)).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        print_to_log_info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_to_log_info(f"Early stopping at epoch {epoch+1}")
                break
    
    artifacts['training_history'] = training_history
    artifacts['model_params'] = {
        'embedding_sizes': embedding_sizes,
        'n_continuous': n_continuous,
        'n_classes': n_classes,
        'layers': layers
    }
    
    print_to_log_info('train_classifier_pytorch time:', time.time()-t)
    
    return {
        'model': model,
        'artifacts': artifacts,
        'device': device
    }

def train_regression_pytorch(df, y_names, cat_names, cont_names, valid_pct=0.2, 
                           bs=1024*5, layers=None, epochs=3, device=None, 
                           lr=1e-3, patience=3, min_delta=0.001, y_range=(0,1)):
    """Train a regression model using PyTorch."""
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    t = time.time()
    
    # Similar structure to classifier but with regression-specific changes
    artifacts = {
        'categorical_encoders': {col: LabelEncoder() for col in cat_names},
        'continuous_scalers': {col: StandardScaler() for col in cont_names},
        'target_scaler': StandardScaler(),
        'na_fills': {},
        'categorical_feature_names': cat_names,
        'continuous_feature_names': cont_names,
        'target_name': y_names[0],
        'model_params': {},
        'training_history': {},
        'is_classification': False,
        'y_range': y_range
    }

    # Target scaling
    y_scaled = artifacts['target_scaler'].fit_transform(df[y_names].values)
    df[y_names[0]] = y_scaled.flatten()

    # Categorical feature encoding and NA handling
    for col in cat_names:
        df[col] = df[col].astype(str)
        fill_val_cat = "MISSING_CAT"
        artifacts['na_fills'][col] = fill_val_cat

        unique_values_for_fit = pd.unique(df[col].fillna(fill_val_cat)).tolist()
        if fill_val_cat not in unique_values_for_fit:
            unique_values_for_fit.append(fill_val_cat)
        
        artifacts['categorical_encoders'][col].fit(unique_values_for_fit)
        df[col] = artifacts['categorical_encoders'][col].transform(df[col].fillna(fill_val_cat))

    # Continuous feature scaling and NA handling
    for col in cont_names:
        fill_val_cont = df[col].median()
        artifacts['na_fills'][col] = fill_val_cont
        df[col] = df[col].fillna(fill_val_cont)
        df[col] = artifacts['continuous_scalers'][col].fit_transform(df[col].values.reshape(-1, 1)).flatten()

    # Calculate embedding sizes
    embedding_sizes = []
    for col in cat_names:
        num_categories = len(artifacts['categorical_encoders'][col].classes_)
        embed_dim = max(4, min(300, int(num_categories / 2)))
        embedding_sizes.append((num_categories, embed_dim))

    artifacts['embedding_sizes'] = embedding_sizes
    n_continuous = len(cont_names)

    # Calculate input size and layer sizes
    input_size = calculate_input_size(embedding_sizes, n_continuous)
    if layers is None:
        layers = define_layer_sizes(input_size)
    print(f"Input size: {input_size}, Layer sizes: {layers}")

    # Prepare data
    X_cat = torch.tensor(df[cat_names].values, dtype=torch.long) if cat_names else torch.empty((len(df), 0), dtype=torch.long)
    X_cont = torch.tensor(df[cont_names].values, dtype=torch.float32) if cont_names else torch.empty((len(df), 0), dtype=torch.float32)
    y = torch.tensor(df[y_names[0]].values, dtype=torch.float32)

    # Train/validation split
    train_idx, val_idx = train_test_split(range(len(df)), test_size=valid_pct, random_state=42)
    
    train_dataset = TabularDataset(X_cat[train_idx], X_cont[train_idx], y[train_idx])
    val_dataset = TabularDataset(X_cat[val_idx], X_cont[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    # Create model (output size 1 for regression)
    model = TabularNNModel(embedding_sizes, n_continuous, 1, layers)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch['categorical'].to(device), batch['continuous'].to(device))
            loss = criterion(outputs.squeeze(), batch['labels'].to(device))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['categorical'].to(device), batch['continuous'].to(device))
                loss = criterion(outputs.squeeze(), batch['labels'].to(device))
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calculate RMSE in scaled and original units
        train_rmse_scaled = train_loss ** 0.5
        val_rmse_scaled = val_loss ** 0.5
        
        # Convert to original units using target scaler
        original_std = artifacts['target_scaler'].scale_[0]
        train_rmse_original = train_rmse_scaled * original_std
        val_rmse_original = val_rmse_scaled * original_std
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_rmse_scaled': train_rmse_scaled,
            'val_rmse_scaled': val_rmse_scaled,
            'train_rmse_original': train_rmse_original,
            'val_rmse_original': val_rmse_original
        })
        
        print_to_log_info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Train RMSE: {train_rmse_scaled:.4f} (scaled) / {train_rmse_original:.4f} (original), "
                          f"Val RMSE: {val_rmse_scaled:.4f} (scaled) / {val_rmse_original:.4f} (original)")
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_to_log_info(f"Early stopping at epoch {epoch+1}")
                break
    
    artifacts['training_history'] = training_history
    artifacts['model_params'] = {
        'embedding_sizes': embedding_sizes,
        'n_continuous': n_continuous,
        'n_classes': 1,
        'layers': layers
    }
    
    print_to_log_info('train_regression_pytorch time:', time.time()-t)
    
    return {
        'model': model,
        'artifacts': artifacts,
        'device': device
    }

def save_model(learn_dict, f):
    """
    Save a PyTorch model and its artifacts.
    
    Args:
        learn_dict: Dictionary containing model, artifacts, and device info
        f: File path to save to
    """
    t = time.time()
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': learn_dict['model'].state_dict(),
        'artifacts': learn_dict['artifacts'],
        'device': learn_dict['device']
    }
    
    # Save using torch.save
    torch.save(save_dict, f)
    print_to_log_info('save_model time:', time.time()-t)

def load_model(f):
    """
    Load a PyTorch model and its artifacts.
    
    Args:
        f: File path to load from
        
    Returns:
        Dictionary containing model, artifacts, and device info
    """
    t = time.time()
    
    # Load the saved dictionary
    # Note: Using weights_only=False because our models contain sklearn objects (LabelEncoder, StandardScaler)
    # which are safe in our trusted context but not allowed by default in PyTorch 2.6+
    save_dict = torch.load(f, map_location='cpu', weights_only=False)
    
    # Reconstruct the model
    artifacts = save_dict['artifacts']
    model_params = artifacts['model_params']
    
    model = TabularNNModel(
        model_params['embedding_sizes'],
        model_params['n_continuous'],
        model_params['n_classes'],
        model_params['layers']
    )
    
    model.load_state_dict(save_dict['model_state_dict'])
    
    learn_dict = {
        'model': model,
        'artifacts': artifacts,
        'device': save_dict['device']
    }
    
    print_to_log_info('load_model time:', time.time()-t)
    return learn_dict

def preprocess_inference_data(df, artifacts):
    """
    Preprocess inference data using the same transformations as training data.
    
    Args:
        df: Pandas DataFrame with inference data
        artifacts: Artifacts from training containing encoders and scalers
        
    Returns:
        Preprocessed DataFrame
    """
    t = time.time()
    df = df.copy()
    
    # Handle categorical features
    for col in artifacts['categorical_feature_names']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            # Fill missing values
            fill_val = artifacts['na_fills'][col]
            df[col] = df[col].fillna(fill_val).infer_objects(copy=False)
            
            # Handle unseen categories by mapping them to a default value
            encoder = artifacts['categorical_encoders'][col]
            known_categories = set(encoder.classes_)
            
            # Vectorized approach: replace unknown categories with first known category
            unknown_mask = ~df[col].isin(known_categories)
            if unknown_mask.any():
                unknown_values = df[col][unknown_mask].unique()
                print_to_log_info(f'Warning: Column {col} contains {len(unknown_values)} unknown values, mapping to default')
                df.loc[unknown_mask, col] = encoder.classes_[0]
            
            # Now transform all values at once (much faster than apply)
            df[col] = encoder.transform(df[col])
    
    # Handle continuous features - vectorized operations
    for col in artifacts['continuous_feature_names']:
        if col in df.columns:
            # Fill missing values
            fill_val = artifacts['na_fills'][col]
            df[col] = df[col].fillna(fill_val).infer_objects(copy=False)
            
            # Scale using the same scaler from training (vectorized)
            scaler = artifacts['continuous_scalers'][col]
            df[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
    
    print_to_log_info(f'preprocess_inference_data time: {time.time()-t:.4f} seconds')
    return df

def get_predictions(learn_dict, df, y_names=None, device=None):
    """
    Perform inference using a trained PyTorch model.
    
    Args:
        learn_dict: Dictionary containing model, artifacts, and device info (from load_model)
        df: DataFrame containing the inference data
        y_names: List of target column names (optional, will use from artifacts if None)
        device: Device to run inference on (None for auto-detection)
        
    Returns:
        DataFrame with predictions and actual values
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    t = time.time()
    
    model = learn_dict['model']
    artifacts = learn_dict['artifacts']
    
    # Get target column name
    if y_names is None:
        y_names = [artifacts['target_name']]
    assert len(y_names) == 1, 'Only one target variable is supported.'
    y_name = y_names[0]
    
    # Check that required columns are present
    required_cols = artifacts['categorical_feature_names'] + artifacts['continuous_feature_names']
    missing_cols = set(required_cols).difference(df.columns)
    if missing_cols:
        print_to_log_info(f"Warning: Missing columns in inference data: {missing_cols}")
    
    assert not df.empty, 'No data to make inferences on.'
    
    # Preprocess the inference data
    preprocess_start = time.time()
    df_processed = preprocess_inference_data(df, artifacts)
    preprocess_time = time.time() - preprocess_start
    print_to_log_info(f'Data preprocessing completed in {preprocess_time:.4f} seconds')
    
    # Handle target column if present (for evaluation)
    target_start = time.time()
    has_target = y_name in df.columns
    if has_target:
        if artifacts['is_classification']:
            # Handle target encoding for classification
            if artifacts.get('target_encoder'):
                target_encoder = artifacts['target_encoder']
                df_target = df[y_name].astype(str)
                
                # Filter out unknown target values
                known_targets = set(target_encoder.classes_)
                unknown_mask = ~df_target.isin(known_targets)
                if unknown_mask.any():
                    unknown_values = df_target[unknown_mask].unique()
                    print_to_log_info(f'Warning: {y_name} contains values which are missing in training set: {unknown_values}')
                    # Remove rows with unknown target values
                    df_processed = df_processed[~unknown_mask]
                    df_target = df_target[~unknown_mask]
                
                if len(df_processed) == 0:
                    print_to_log_info("No valid data remaining after filtering unknown target values")
                    return pd.DataFrame()
                
                true_labels = df_target.values
                true_codes = target_encoder.transform(df_target)
        else:
            # For regression, use target scaler if available
            if artifacts.get('target_scaler'):
                true_values = artifacts['target_scaler'].transform(df[[y_name]].values).flatten()
            else:
                true_values = df[y_name].values
    
    target_time = time.time() - target_start
    print_to_log_info(f'Target processing completed in {target_time:.4f} seconds')

    # Prepare data for inference
    cat_names = artifacts['categorical_feature_names']
    cont_names = artifacts['continuous_feature_names']
    
    # Move model to device and verify
    model.eval()
    model.to(device)
    print_to_log_info(f"Model moved to device: {device}")
    
    # Verify model is on correct device by checking first parameter
    if hasattr(model, 'parameters'):
        first_param_device = next(model.parameters()).device
        print_to_log_info(f"Model parameters are on device: {first_param_device}")
    
    # Create tensors and move to device immediately
    tensor_start = time.time()
    X_cat = torch.tensor(df_processed[cat_names].values, dtype=torch.long, device=device) if cat_names else torch.empty((len(df_processed), 0), dtype=torch.long, device=device)
    X_cont = torch.tensor(df_processed[cont_names].values, dtype=torch.float32, device=device) if cont_names else torch.empty((len(df_processed), 0), dtype=torch.float32, device=device)
    
    # Create dataset and dataloader
    # For inference, we don't need labels, so create dummy labels on device
    dummy_labels = torch.zeros(len(df_processed), dtype=torch.long if artifacts['is_classification'] else torch.float32, device=device)
    inference_dataset = TabularDataset(X_cat, X_cont, dummy_labels)
    inference_loader = DataLoader(inference_dataset, batch_size=1024, shuffle=False)
    tensor_time = time.time() - tensor_start
    print_to_log_info(f'Tensor creation completed in {tensor_time:.4f} seconds')
    
    # Run inference
    all_predictions = []
    
    print_to_log_info(f"Starting inference on {device} with {len(df_processed)} samples")
    inference_start = time.time()
    
    # Log GPU memory usage before inference
    if device == 'cuda' and torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated(0) / 1024**3
        cached_before = torch.cuda.memory_reserved(0) / 1024**3
        print_to_log_info(f"GPU memory before inference - Allocated: {allocated_before:.2f} GB, Cached: {cached_before:.2f} GB")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(inference_loader):
            # Data should already be on device, but verify for first batch
            if batch_idx == 0:
                print_to_log_info(f"Batch categorical data device: {batch['categorical'].device}")
                print_to_log_info(f"Batch continuous data device: {batch['continuous'].device}")
            
            # Since data is already on device, no need to move it again
            outputs = model(batch['categorical'], batch['continuous'])
            all_predictions.append(outputs.cpu())
    
    inference_time = time.time() - inference_start
    print_to_log_info(f"Inference completed in {inference_time:.4f} seconds on {device}")
    
    # Log GPU memory usage after inference
    if device == 'cuda' and torch.cuda.is_available():
        allocated_after = torch.cuda.memory_allocated(0) / 1024**3
        cached_after = torch.cuda.memory_reserved(0) / 1024**3
        print_to_log_info(f"GPU memory after inference - Allocated: {allocated_after:.2f} GB, Cached: {cached_after:.2f} GB")

    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=0)
    
    # Process predictions based on task type
    result_start = time.time()
    if artifacts['is_classification']:
        # Convert probabilities to class labels
        pred_probs = torch.softmax(predictions, dim=1)
        pred_codes = predictions.argmax(dim=1).numpy()
        
        # Decode predictions back to original labels
        target_encoder = artifacts['target_encoder']
        pred_labels = target_encoder.inverse_transform(pred_codes)
        
        results = {
            f'{y_name}_Pred': pred_labels,
            f'{y_name}_Pred_Code': pred_codes
        }
        
        if has_target:
            results.update({
                f'{y_name}_Actual': true_labels,
                f'{y_name}_Actual_Code': true_codes,
                f'{y_name}_Match': [pred == true for pred, true in zip(pred_labels, true_labels)],
                f'{y_name}_Match_Code': [pred == true for pred, true in zip(pred_codes, true_codes)]
            })
    else:
        # For regression
        pred_values = predictions.squeeze().numpy()
        
        # Inverse transform if target scaler was used
        if artifacts.get('target_scaler'):
            pred_values = artifacts['target_scaler'].inverse_transform(pred_values.reshape(-1, 1)).flatten()
        
        results = {
            f'{y_name}_Pred': pred_values
        }
        
        if has_target:
            # Inverse transform true values if needed
            if artifacts.get('target_scaler'):
                true_values_orig = artifacts['target_scaler'].inverse_transform(true_values.reshape(-1, 1)).flatten()
            else:
                true_values_orig = true_values
                
            results.update({
                f'{y_name}_Actual': true_values_orig,
                f'{y_name}_Error': pred_values - true_values_orig,
                f'{y_name}_AbsoluteError': np.abs(pred_values - true_values_orig)
            })
    
    result_time = time.time() - result_start
    print_to_log_info(f'Result processing completed in {result_time:.4f} seconds')
    
    print_to_log_info('get_predictions time:', time.time()-t)
    return pd.DataFrame(results)

def find_first_linear_layer(module):
    """
    Find the first linear layer in a PyTorch model.
    
    Args:
        module: PyTorch module to search
        
    Returns:
        First nn.Linear layer found, or None if not found
    """
    if isinstance(module, nn.Linear):
        return module
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):
        for layer in module:
            found = find_first_linear_layer(layer)
            if found:
                return found
    elif hasattr(module, 'children'):
        for layer in module.children():
            found = find_first_linear_layer(layer)
            if found:
                return found
    return None

def get_feature_importance(learn_dict):
    """
    Calculate feature importance based on the weights of the first linear layer.
    
    Args:
        learn_dict: Dictionary containing model, artifacts, and device info
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    model = learn_dict['model']
    artifacts = learn_dict['artifacts']
    
    importance = {}

    # Find the first linear layer in the model
    linear_layer = find_first_linear_layer(model)
    if linear_layer is None:
        raise ValueError("No linear layer found in the model.")
    
    # Get the absolute mean of the weights across the input features
    weights = linear_layer.weight.abs().mean(dim=0)

    # Get feature names from artifacts
    cat_names = artifacts['categorical_feature_names']
    cont_names = artifacts['continuous_feature_names']

    # Calculate the total input size to the first linear layer
    # For our TabularNNModel, embeddings are stored in model.embeddings
    emb_szs = {}
    if hasattr(model, 'embeddings') and model.embeddings:
        for i, name in enumerate(cat_names):
            if i < len(model.embeddings):
                emb_szs[name] = model.embeddings[i].embedding_dim
            else:
                # Fallback to artifacts if available
                if 'embedding_sizes' in artifacts.get('model_params', {}):
                    emb_szs[name] = artifacts['model_params']['embedding_sizes'][i][1]
                else:
                    emb_szs[name] = 1  # Default fallback
    
    total_input_size = sum(emb_szs.values()) + len(cont_names)

    print_to_log_info(f"Embedding sizes: {emb_szs}")
    print_to_log_info(f"Total input size to the first linear layer: {total_input_size}")
    print_to_log_info(f"Shape of weights: {weights.shape}")

    # Ensure the number of weights matches the total input size
    if len(weights) != total_input_size:
        raise ValueError(f"Number of weights ({len(weights)}) does not match total input size ({total_input_size}).")

    # Assign importance to each feature
    idx = 0
    for name in cat_names:
        emb_size = emb_szs.get(name, 1)
        if emb_size > 1:
            importance[name] = weights[idx:idx+emb_size].mean().item()  # Average the importance across the embedding dimensions
        else:
            importance[name] = weights[idx].item()
        idx += emb_size
    
    for name in cont_names:
        importance[name] = weights[idx].item()
        idx += 1
    
    return importance

def chart_feature_importance(learn_dict, topn=None):
    """
    Calculate and visualize feature importance.
    
    Args:
        learn_dict: Dictionary containing model, artifacts, and device info
        topn: Number of top features to display (None for all features)
    """
    # Calculate and display feature importance
    importance = get_feature_importance(learn_dict)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top N features if specified
    if topn is not None:
        sorted_importance = sorted_importance[:topn]
        print(f"\nTop {len(sorted_importance)} Feature Importances (out of {len(importance)} total):")
    else:
        print(f"\nFeature Importances {len(importance)}:")
    
    for name, imp in sorted_importance:
        print_to_log_info(f"{name}: {imp:.4f}")

    # Visualize the importance
    try:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(24, 4))
        plt.bar(range(len(sorted_importance)), [imp for name, imp in sorted_importance])
        plt.xticks(range(len(sorted_importance)), [name for name, imp in sorted_importance], rotation=45, ha='right')
        
        if topn is not None:
            plt.title(f'Top {len(sorted_importance)} Feature Importance (out of {len(importance)} total)')
        else:
            plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print_to_log_info("matplotlib not available, skipping visualization")
    except Exception as e:
        print_to_log_info(f"Error creating plot: {e}") 