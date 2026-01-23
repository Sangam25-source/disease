"""
Federated Learning Framework for FedED-SegNAS
==============================================

Implements SimpleFederatedTrainer for distributed training of Fuzzy CNN
without privacy mechanisms (suitable for college project).

Author: FedED-SegNAS Project
Date: January 2025
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
import json
import sys
import os

# Add parent directory to path for NAS imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fuzzy_cnn import build_fuzzy_cnn
from NAS.pso_nas import MultiObjectivePSO, FitnessCalculator


class SimpleFederatedTrainer:
    """
    Federated Learning Trainer WITHOUT Privacy Mechanisms
    
    Implements FedAvg algorithm for distributed training across multiple clients.
    Suitable for college project demonstration of federated learning concepts.
    
    Parameters:
    -----------
    num_snps : int
        Number of SNP features in input data
    num_clients : int, default=50
        Total number of federated clients
    learning_rate : float, default=0.001
        Learning rate for local client training
    
    Attributes:
    -----------
    global_model : FixedFuzzyCNN
        The global model that aggregates knowledge from all clients
    history : dict
        Training history containing metrics per round
    
    Example:
    --------
    >>> trainer = SimpleFederatedTrainer(num_snps=50, num_clients=50)
    >>> history = trainer.train(federated_data, num_rounds=50)
    >>> results = trainer.evaluate(X_test, y_test)
    """
    
    def __init__(self, num_snps, num_clients=50, learning_rate=0.001, use_nas=False):
        """
        Initialize federated learning trainer.
        
        Parameters:
        -----------
        num_snps : int
            Number of SNP features
        num_clients : int
            Total number of federated clients
        learning_rate : float
            Learning rate for Adam optimizer
        use_nas : bool, default=False
            Whether to use Neural Architecture Search
        """
        self.num_snps = num_snps
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        self.use_nas = use_nas
        
        # Initialize NAS if enabled
        if self.use_nas:
            print(f">> NAS enabled - will search every 5 rounds")
            self.pso_optimizer = MultiObjectivePSO(
                num_snps=num_snps,
                num_particles=5,  # Small for speed
                max_iterations=10
            )
            self.current_architecture = None
        
        # Initialize global model
        print(f">> Initializing global Fuzzy CNN model...")
        self.global_model = build_fuzzy_cnn(
            num_snps=num_snps, 
            learning_rate=learning_rate
        )
        print(f">> Global model initialized with {self.global_model.count_params():,} parameters")
        
        # Training history
        self.history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'communication_overhead_mb': [],
            'round_time_seconds': [],
            'learning_rate': [],  # Track learning rate changes
            'nas_searches': [],  # Track NAS searches
            'early_stopped': False,
            'early_stop_round': None,
            'best_val_accuracy': 0.0
        }
    
    def get_cosine_annealing_lr(self, current_round, total_rounds, initial_lr, min_lr):
        """
        Calculate learning rate using cosine annealing schedule.
        
        Parameters:
        -----------
        current_round : int
            Current training round (0-indexed)
        total_rounds : int
            Total number of training rounds
        initial_lr : float
            Initial learning rate
        min_lr : float
            Minimum learning rate
            
        Returns:
        --------
        float : Learning rate for current round
        """
        import math
        if total_rounds <= 1:
            return initial_lr
        
        # Cosine annealing formula
        cosine_factor = 0.5 * (1 + math.cos(math.pi * current_round / (total_rounds - 1)))
        lr = min_lr + (initial_lr - min_lr) * cosine_factor
        return lr
    
    def update_architecture(self, architecture):
        """
        Update global model architecture based on NAS result.
        
        Parameters:
        -----------
        architecture : dict
            New architecture specification from NAS
        """
        print(f"\n>> Updating model architecture...")
        print(f"   New architecture: {architecture['num_blocks']} blocks")
        for i, block in enumerate(architecture['blocks']):
            pool_name = ['MaxPool', 'AvgPool', 'None'][block['pool_type']]
            print(f"   Block {i+1}: {block['filters']} filters, kernel={block['kernel_size']}, pool={pool_name}")
        print(f"   Dense: {architecture['dense_1']}, {architecture['dense_2']}")
        print(f"   Dropout: {architecture['dropout']:.3f}")
        
        # Build new model with discovered architecture
        calculator = FitnessCalculator(self.num_snps)
        new_model = calculator.build_model_from_architecture(architecture)
        
        # Try to transfer weights from old model (if compatible)
        try:
            # Get old weights
            old_weights = self.global_model.get_weights()
            
            # Set new weights (will fail if incompatible, which is OK)
            new_model.set_weights(old_weights)
            print(f">> Transferred weights from old model")
        except Exception as e:
            print(f">> Could not transfer weights (architecture changed significantly)")
        
        # Replace global model
        old_params = self.global_model.count_params()
        del self.global_model
        tf.keras.backend.clear_session()
        self.global_model = new_model
        new_params = self.global_model.count_params()
        
        print(f">> Model updated: {old_params:,} -> {new_params:,} parameters")
        print(f"   Parameter change: {((new_params - old_params) / old_params * 100):+.1f}%\n")
    
    def train_client(self, client_data, global_weights, local_epochs=5, verbose=False):
        """
        Train a single client on its local data.
        
        Parameters:
        -----------
        client_data : dict
            Dictionary with 'X' and 'y' keys containing client's training data
        global_weights : list
            Current global model weights to initialize local model
        local_epochs : int, default=5
            Number of epochs to train on local data
        verbose : bool, default=False
            Whether to print training progress
        
        Returns:
        --------
        local_weights : list
            Updated weights after local training
        local_history : dict
            Training metrics from local training
        """
        # Extract local data
        X_local = client_data['X'].astype(np.float32)
        y_local = client_data['y'].astype(np.int32)
        
        # Create local model with same architecture as global model
        # Clone the global model architecture
        local_model = tf.keras.models.clone_model(self.global_model)
        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Set local model weights to global weights
        local_model.set_weights(global_weights)
        
        # Train on local data
        history = local_model.fit(
            X_local, y_local,
            epochs=local_epochs,
            batch_size=32,
            verbose=1 if verbose else 0,
            shuffle=True
        )
        
        # Get updated local weights
        local_weights = local_model.get_weights()
        
        # Get final metrics
        local_metrics = {
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'num_samples': len(X_local)
        }
        
        # Clean up to free memory
        del local_model
        tf.keras.backend.clear_session()
        
        return local_weights, local_metrics
    
    def aggregate_weights(self, client_weights_list, client_sample_counts=None):
        """
        Aggregate client weights using FedAvg algorithm.
        
        Parameters:
        -----------
        client_weights_list : list of lists
            List of weight arrays from each client
        client_sample_counts : list of ints, optional
            Number of samples per client (for weighted averaging)
            If None, uses simple averaging
        
        Returns:
        --------
        aggregated_weights : list
            Aggregated global weights
        
        Notes:
        ------
        FedAvg formula: w_global = Σ (n_k / N) * w_k
        where n_k is number of samples at client k, N is total samples
        """
        num_clients = len(client_weights_list)
        
        # If no sample counts provided, use simple average
        if client_sample_counts is None:
            client_sample_counts = [1] * num_clients
        
        total_samples = sum(client_sample_counts)
        
        # Initialize aggregated weights
        aggregated_weights = []
        
        # Aggregate each layer's weights
        num_layers = len(client_weights_list[0])
        for layer_idx in range(num_layers):
            # Collect this layer's weights from all clients
            layer_weights = []
            for client_idx in range(num_clients):
                weight = client_weights_list[client_idx][layer_idx]
                sample_count = client_sample_counts[client_idx]
                # Weight by number of samples
                weighted_weight = weight * (sample_count / total_samples)
                layer_weights.append(weighted_weight)
            
            # Sum all weighted contributions
            aggregated_layer_weight = np.sum(layer_weights, axis=0)
            aggregated_weights.append(aggregated_layer_weight)
        
        return aggregated_weights
    
    def calculate_communication_overhead(self, weights):
        """
        Calculate communication overhead in MB.
        
        Parameters:
        -----------
        weights : list
            Model weights
        
        Returns:
        --------
        overhead_mb : float
            Communication overhead in megabytes
        """
        # Each float32 parameter is 4 bytes
        total_params = sum(w.size for w in weights)
        overhead_bytes = total_params * 4
        overhead_mb = overhead_bytes / (1024 * 1024)
        return overhead_mb
    
    def train(self, federated_data, num_rounds=50, clients_per_round=10,
              local_epochs=5, nas_frequency=5, verbose=True, 
              early_stopping=True, patience=20, min_lr=0.0001):
        """
        Main federated training loop with optional NAS, early stopping, and LR scheduling.
        
        Parameters:
        -----------
        federated_data : dict or np.lib.npyio.NpzFile
            Preprocessed federated data with keys:
            - 'client_0_X', 'client_0_y', ..., 'client_N_X', 'client_N_y'
            - 'validation_X', 'validation_y'
            - 'test_X', 'test_y'
        num_rounds : int, default=50
            Number of communication rounds
        clients_per_round : int, default=10
            Number of clients to sample per round
        local_epochs : int, default=5
            Number of epochs each client trains locally
        nas_frequency : int, default=5
            Run NAS every N rounds (if use_nas=True)
        verbose : bool, default=True
            Whether to print progress
        early_stopping : bool, default=True
            Enable early stopping based on validation accuracy
        patience : int, default=20
            Early stopping patience (rounds without improvement)
        min_lr : float, default=0.0001
            Minimum learning rate for cosine annealing
        
        Returns:
        --------
        history : dict
            Training history with metrics per round
        
        Algorithm:
        ----------
        1. Initialize global model
        2. For each round:
           a. (Optional) Run NAS if enabled and it's time
           b. Update learning rate using cosine annealing
           c. Select random subset of clients
           d. Send global model to selected clients
           e. Clients train on local data (local_epochs)
           f. Clients send updated weights to server
           g. Server aggregates weights using FedAvg
           h. Update global model
           i. Evaluate on validation set
           j. Check early stopping condition
        3. Return training history
        """
        # Extract validation data
        X_val = federated_data['validation_X'].astype(np.float32)
        y_val = federated_data['validation_y'].astype(np.int32)
        
        # Count available clients
        available_clients = sum(1 for k in federated_data.keys() if k.endswith('_X') and k.startswith('client_'))
        
        # Early stopping variables
        best_val_accuracy = 0.0
        patience_counter = 0
        initial_lr = self.learning_rate
        
        if verbose:
            print("\n" + "="*80)
            print("FEDERATED LEARNING TRAINING" + (" WITH NAS" if self.use_nas else ""))
            print("="*80)
            print(f"Configuration:")
            print(f"  >> Total clients: {available_clients}")
            print(f"  >> Clients per round: {clients_per_round}")
            print(f"  >> Communication rounds: {num_rounds}")
            print(f"  >> Local epochs: {local_epochs}")
            print(f"  >> Early stopping: {early_stopping} (patience={patience})")
            print(f"  >> Learning rate: {initial_lr} -> {min_lr} (cosine annealing)")
            if self.use_nas:
                print(f"  >> NAS enabled: every {nas_frequency} rounds")
                print(f"  >> NAS particles: {self.pso_optimizer.num_particles}")
                print(f"  >> NAS iterations: {self.pso_optimizer.max_iterations}")
            print(f"  >> Validation samples: {len(X_val)}")
            print("="*80 + "\n")
        
        # Main federated training loop
        for round_num in range(num_rounds):
            round_start_time = time.time()
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"Round {round_num + 1}/{num_rounds}")
                print(f"{'='*80}")
            
            # Update learning rate using cosine annealing
            current_lr = self.get_cosine_annealing_lr(
                round_num, num_rounds, initial_lr, min_lr
            )
            
            # Update optimizer learning rate
            try:
                self.global_model.optimizer.learning_rate.assign(current_lr)
                if verbose and round_num % 10 == 0:  # Print every 10 rounds
                    print(f">> Learning rate updated: {current_lr:.6f}")
            except Exception as e:
                if verbose:
                    print(f">> Warning: Could not update learning rate: {e}")
            
            # Run NAS if enabled and it's time
            if self.use_nas and round_num > 0 and round_num % nas_frequency == 0:
                if verbose:
                    print(f"\n>> Running NAS (Round {round_num})...")
                
                # Prepare small training set for NAS (use first client's data)
                X_train_nas = federated_data['client_0_X'][:200].astype(np.float32)
                y_train_nas = federated_data['client_0_y'][:200].astype(np.int32)
                X_val_nas = X_val[:100]
                y_val_nas = y_val[:100]
                
                # Run PSO search
                nas_start_time = time.time()
                best_arch, best_fitness = self.pso_optimizer.search(
                    X_train_nas, y_train_nas,
                    X_val_nas, y_val_nas,
                    current_round=round_num,
                    max_rounds=num_rounds,
                    verbose=verbose
                )
                nas_time = time.time() - nas_start_time
                
                # Update architecture
                self.update_architecture(best_arch)
                
                # Record NAS search
                self.history['nas_searches'].append({
                    'round': round_num,
                    'architecture': best_arch,
                    'fitness': float(best_fitness),
                    'search_time_seconds': float(nas_time)
                })
                
                if verbose:
                    print(f">> NAS complete in {nas_time/60:.2f} minutes")
            
            # Step 1: Select random clients
            selected_indices = np.random.choice(
                available_clients, 
                size=min(clients_per_round, available_clients), 
                replace=False
            )
            
            if verbose:
                print(f">> Selected clients: {list(selected_indices)}")
            
            # Step 2: Get current global weights
            global_weights = self.global_model.get_weights()
            
            # Step 3: Train selected clients
            client_weights_list = []
            client_sample_counts = []
            client_metrics_list = []
            
            for client_idx in selected_indices:
                # Prepare client data
                client_data = {
                    'X': federated_data[f'client_{client_idx}_X'],
                    'y': federated_data[f'client_{client_idx}_y']
                }
                
                if verbose:
                    print(f"  >> Training client {client_idx} ({len(client_data['X'])} samples)...", end=" ")
                
                # Train client
                local_weights, local_metrics = self.train_client(
                    client_data, 
                    global_weights, 
                    local_epochs=local_epochs,
                    verbose=False
                )
                
                client_weights_list.append(local_weights)
                client_sample_counts.append(local_metrics['num_samples'])
                client_metrics_list.append(local_metrics)
                
                if verbose:
                    print(f"Loss: {local_metrics['final_loss']:.4f}, Acc: {local_metrics['final_accuracy']:.4f}")
            
            # Step 4: Aggregate weights using FedAvg
            if verbose:
                print(f"\n  >> Aggregating weights from {len(client_weights_list)} clients...")
            
            aggregated_weights = self.aggregate_weights(
                client_weights_list, 
                client_sample_counts
            )
            
            # Step 5: Update global model
            self.global_model.set_weights(aggregated_weights)
            
            # Step 6: Evaluate on validation set
            val_results = self.global_model.evaluate(X_val, y_val, verbose=0)
            val_loss = val_results[0]
            val_accuracy = val_results[1]
            
            # Calculate average client metrics
            avg_train_loss = np.mean([m['final_loss'] for m in client_metrics_list])
            avg_train_acc = np.mean([m['final_accuracy'] for m in client_metrics_list])
            
            # Calculate communication overhead
            comm_overhead = self.calculate_communication_overhead(aggregated_weights)
            
            # Calculate round time
            round_time = time.time() - round_start_time
            
            # Record history
            self.history['rounds'].append(round_num + 1)
            self.history['train_loss'].append(float(avg_train_loss))
            self.history['train_accuracy'].append(float(avg_train_acc))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_accuracy'].append(float(val_accuracy))
            self.history['communication_overhead_mb'].append(float(comm_overhead))
            self.history['round_time_seconds'].append(float(round_time))
            self.history['learning_rate'].append(float(current_lr))
            
            # Early stopping logic
            if early_stopping:
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    self.history['best_val_accuracy'] = float(best_val_accuracy)
                    if verbose:
                        print(f"  >> ✅ New best validation accuracy: {best_val_accuracy:.4f}")
                else:
                    patience_counter += 1
                    if verbose:
                        print(f"  >> ⏳ No improvement ({patience_counter}/{patience})")
                
                # Check if we should stop early
                if patience_counter >= patience and round_num >= 10:  # Minimum 10 rounds
                    if verbose:
                        print(f"\n🛑 Early stopping triggered at round {round_num + 1}")
                        print(f"   Best validation accuracy: {best_val_accuracy:.4f}")
                    
                    self.history['early_stopped'] = True
                    self.history['early_stop_round'] = round_num + 1
                    break
            
            if verbose:
                print(f"\n  >> Round {round_num + 1} Results:")
                print(f"     Learning Rate:         {current_lr:.6f}")
                print(f"     Avg Client Train Loss: {avg_train_loss:.4f}")
                print(f"     Avg Client Train Acc:  {avg_train_acc:.4f}")
                print(f"     Global Val Loss:       {val_loss:.4f}")
                print(f"     Global Val Accuracy:   {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
                if early_stopping:
                    print(f"     Best Val Accuracy:     {best_val_accuracy:.4f}")
                print(f"     Communication:         {comm_overhead:.2f} MB")
                print(f"     Round Time:            {round_time:.2f}s")
            
            # Progress indicator
            if verbose and (round_num + 1) % 10 == 0:
                print(f"\n  >> Milestone: {round_num + 1}/{num_rounds} rounds complete!")
        
        # Update final best accuracy
        if not early_stopping:
            self.history['best_val_accuracy'] = max(self.history['val_accuracy'])
        
        if verbose:
            print("\n" + "="*80)
            print("FEDERATED TRAINING COMPLETE")
            print("="*80)
            total_time = sum(self.history['round_time_seconds'])
            final_accuracy = self.history['val_accuracy'][-1]
            best_accuracy = self.history['best_val_accuracy']
            print(f"Total training time: {total_time/60:.2f} minutes")
            print(f"Final validation accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
            print(f"Best validation accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            if self.history['early_stopped']:
                print(f"Early stopped at round: {self.history['early_stop_round']}")
            print("="*80 + "\n")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate global model on test set.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        
        Returns:
        --------
        results : dict
            Test metrics including loss, accuracy
        """
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.int32)
        
        test_results = self.global_model.evaluate(X_test, y_test, verbose=0)
        
        results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1])
        }
        
        return results
    
    def save_model(self, filepath):
        """
        Save trained global model.
        
        Parameters:
        -----------
        filepath : str
            Path to save model (e.g., 'models/model1.h5')
        """
        self.global_model.save(filepath)
        print(f">> Global model saved to: {filepath}")
    
    def save_history(self, filepath):
        """
        Save training history to JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to save history (e.g., 'results/history.json')
        """
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f">> Training history saved to: {filepath}")


# Example usage and testing
if __name__ == '__main__':
    print("="*80)
    print("FEDERATED LEARNING FRAMEWORK TEST")
    print("="*80)
    
    # This is a minimal test - full testing should use real preprocessed data
    print("\n>> SimpleFederatedTrainer class defined")
    print(">> All methods implemented")
    print("\nTo use this framework:")
    print("1. Load preprocessed federated data (.npz file)")
    print("2. Initialize SimpleFederatedTrainer")
    print("3. Call trainer.train()")
    print("4. Evaluate with trainer.evaluate()")
    
    print("\n" + "="*80)
    print("FEDERATED LEARNING MODULE READY")
    print("="*80)
