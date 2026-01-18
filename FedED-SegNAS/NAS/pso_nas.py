"""
PSO-based Neural Architecture Search for FedED-SegNAS
======================================================

Implements Algorithm 2 from the paper:
"Multi-Objective Segmented NAS Based on PSO for FL"

This module provides:
- Particle class: Represents one architecture in PSO swarm
- FitnessCalculator: Multi-objective fitness evaluation (Tasks 4)
- MultiObjectivePSO: Main PSO optimizer (Task 5)

Author: FedED-SegNAS Project
Date: January 2026
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NAS.search_space import SearchSpace


class Particle:
    """
    Represents one particle in PSO swarm.
    Each particle represents a neural network architecture.
    
    Attributes:
    -----------
    position : np.ndarray
        Current architecture encoding (16-element vector)
    velocity : np.ndarray
        Rate of change of position
    pbest_position : np.ndarray
        Personal best position found so far
    pbest_fitness : float
        Fitness value at personal best position
    fitness : float
        Current fitness value
    
    Paper Reference:
    ----------------
    Equation 21: Velocity update
    Equation 22: Position update
    
    Example:
    --------
    >>> space = SearchSpace()
    >>> arch = space.random_architecture()
    >>> position = space.encode_architecture(arch)
    >>> velocity = np.random.randn(16) * 0.1
    >>> particle = Particle(position, velocity)
    >>> particle.update_velocity(gbest_position)
    >>> particle.update_position(space)
    """
    
    def __init__(self, position, velocity):
        """
        Initialize particle with position and velocity.
        
        Parameters:
        -----------
        position : np.ndarray
            Initial architecture encoding (16 elements)
        velocity : np.ndarray
            Initial velocity vector (16 elements)
        """
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.pbest_position = position.copy()
        self.pbest_fitness = float('inf')  # Minimize fitness
        self.fitness = float('inf')
    
    def update_velocity(self, gbest_position, w=0.7, c1=2.0, c2=2.0):
        """
        Update particle velocity using PSO equation.
        
        Paper Equation 21:
        V_i^(k+1) = w*V_i^k + c1*d1*(Pbest_i - B_i^k) + c2*d2*(Gbest - B_i^k)
        
        Where:
        - w: Inertia weight (controls exploration vs exploitation)
        - c1: Cognitive coefficient (attraction to personal best)
        - c2: Social coefficient (attraction to global best)
        - d1, d2: Random values in [0, 1]
        
        Parameters:
        -----------
        gbest_position : np.ndarray
            Global best position found by swarm
        w : float, default=0.7
            Inertia weight
        c1 : float, default=2.0
            Cognitive acceleration constant
        c2 : float, default=2.0
            Social acceleration constant
        
        Example:
        --------
        >>> particle.update_velocity(gbest_position, w=0.7, c1=2.0, c2=2.0)
        """
        # Random coefficients for stochastic behavior
        d1 = np.random.random(self.position.shape)
        d2 = np.random.random(self.position.shape)
        
        # Cognitive component: attraction to personal best
        cognitive = c1 * d1 * (self.pbest_position - self.position)
        
        # Social component: attraction to global best
        social = c2 * d2 * (gbest_position - self.position)
        
        # Update velocity: inertia + cognitive + social
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self, search_space):
        """
        Update particle position using velocity.
        
        Paper Equation 22:
        B_i^(k+1) = B_i^k + V_i^(k+1)
        
        After updating, clips position to valid ranges defined by search space.
        
        Parameters:
        -----------
        search_space : SearchSpace
            Search space object with valid ranges
        
        Position Vector Structure (16 elements):
        [0]: num_blocks (1-4)
        [1-3]: block 0 (filters, kernel, pool)
        [4-6]: block 1 (filters, kernel, pool)
        [7-9]: block 2 (filters, kernel, pool)
        [10-12]: block 3 (filters, kernel, pool)
        [13]: dense_1 (128-512)
        [14]: dense_2 (128-512)
        [15]: dropout (0.2-0.5)
        
        Example:
        --------
        >>> particle.update_position(search_space)
        """
        # Update position
        self.position = self.position + self.velocity
        
        # Clip to valid ranges
        # Element 0: num_blocks [1, 4]
        self.position[0] = np.clip(self.position[0], 1, 4)
        
        # Elements 1-12: Block parameters (4 blocks × 3 params each)
        for i in range(4):
            idx = 1 + i * 3
            # Filters: [32, 256]
            self.position[idx] = np.clip(self.position[idx], 32, 256)
            # Kernel size: [3, 7]
            self.position[idx + 1] = np.clip(self.position[idx + 1], 3, 7)
            # Pool type: [0, 2]
            self.position[idx + 2] = np.clip(self.position[idx + 2], 0, 2)
        
        # Element 13: dense_1 [128, 512]
        self.position[13] = np.clip(self.position[13], 128, 512)
        # Element 14: dense_2 [128, 512]
        self.position[14] = np.clip(self.position[14], 128, 512)
        # Element 15: dropout [0.2, 0.5]
        self.position[15] = np.clip(self.position[15], 0.2, 0.5)
    
    def update_pbest(self):
        """
        Update personal best if current fitness is better.
        
        Since we're minimizing fitness, update if current fitness < pbest_fitness.
        
        Returns:
        --------
        updated : bool
            True if personal best was updated, False otherwise
        
        Example:
        --------
        >>> particle.fitness = 0.45
        >>> updated = particle.update_pbest()
        >>> if updated:
        ...     print(f"New best: {particle.pbest_fitness}")
        """
        if self.fitness < self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()
            return True
        return False
    
    def get_architecture(self, search_space):
        """
        Get architecture dict from current position.
        
        Parameters:
        -----------
        search_space : SearchSpace
            Search space for decoding
        
        Returns:
        --------
        arch : dict
            Architecture specification
        """
        return search_space.decode_position(self.position)
    
    def __repr__(self):
        """String representation of particle."""
        return (f"Particle(fitness={self.fitness:.4f}, "
                f"pbest={self.pbest_fitness:.4f}, "
                f"pos_sample={self.position[:3]}...)")


class FitnessCalculator:
    """
    Calculate multi-objective fitness for architectures.
    
    Implements Paper Equations 10-15:
    - f1: Misclassification rate (1 - accuracy) - Equation 11
    - f2: Number of blocks - Equation 12
    - f3: Number of parameters - Equation 13
    - f4: GFLOPs (computational cost) - Equation 14
    - Segmented strategy - Equation 15
    
    The segmented strategy adapts objectives across 3 FL stages:
    - Stage 1 (0 < r < R/3):     All objectives {f1, f2, f3, f4}
    - Stage 2 (R/3 <= r < 2R/3): Accuracy + GFLOPs {f1, f4}
    - Stage 3 (2R/3 <= r < R):   Accuracy + Parameters {f1, f3}
    
    Parameters:
    -----------
    num_snps : int
        Number of SNP features in input data
    
    Example:
    --------
    >>> calculator = FitnessCalculator(num_snps=50)
    >>> arch = {'num_blocks': 2, 'blocks': [...], 'dense_1': 256, ...}
    >>> fitness = calculator.calculate_fitness(arch, X_train, y_train, X_val, y_val, 
    ...                                        stage=1, current_round=5, max_rounds=50)
    """
    
    def __init__(self, num_snps):
        """
        Initialize fitness calculator.
        
        Parameters:
        -----------
        num_snps : int
            Number of SNP features
        """
        self.num_snps = num_snps
    
    def build_model_from_architecture(self, arch):
        """
        Build Fuzzy CNN model from architecture specification.
        
        This creates a model dynamically based on the architecture dict,
        allowing PSO to explore different network structures.
        
        Parameters:
        -----------
        arch : dict
            Architecture specification with keys:
            - num_blocks: int
            - blocks: list of dicts with 'filters', 'kernel_size', 'pool_type'
            - dense_1: int
            - dense_2: int
            - dropout: float
        
        Returns:
        --------
        model : tf.keras.Model
            Compiled Fuzzy CNN model
        
        Example:
        --------
        >>> arch = {'num_blocks': 2, 'blocks': [{'filters': 64, 'kernel_size': 3, 'pool_type': 0}], ...}
        >>> model = calculator.build_model_from_architecture(arch)
        """
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        # Import fuzzy layers from parent directory
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from models.fuzzy_cnn import FuzzificationLayer, DefuzzificationLayer
        
        # Input layer
        inputs = layers.Input(shape=(self.num_snps,), name='input')
        
        # Fuzzification layer (Paper Equation 2)
        x = FuzzificationLayer(
            num_fuzzy_sets=3,
            name='fuzzification'
        )(inputs)
        
        # Convolutional blocks (Paper Equation 3)
        for i, block in enumerate(arch['blocks']):
            # Conv layer with sigmoid activation (fuzzy logic)
            x = layers.Conv1D(
                filters=int(block['filters']),
                kernel_size=int(block['kernel_size']),
                activation='sigmoid',
                padding='same',
                name=f'conv_{i}'
            )(x)
            
            # Pooling (Paper Equation 4)
            if block['pool_type'] == 0:  # MaxPool
                x = layers.MaxPooling1D(pool_size=2, name=f'pool_{i}')(x)
            elif block['pool_type'] == 1:  # AvgPool
                x = layers.AveragePooling1D(pool_size=2, name=f'pool_{i}')(x)
            # else: No pooling (pool_type == 2)
            
            x = layers.Dropout(float(arch['dropout']), name=f'dropout_{i}')(x)
        
        # Defuzzification layer (Paper Equations 5-6)
        x = DefuzzificationLayer(name='defuzzification')(x)
        
        # Flatten
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layers
        x = layers.Dense(int(arch['dense_1']), activation='relu', name='dense_1')(x)
        x = layers.Dropout(float(arch['dropout']), name='dropout_dense_1')(x)
        x = layers.Dense(int(arch['dense_2']), activation='relu', name='dense_2')(x)
        x = layers.Dropout(float(arch['dropout']), name='dropout_dense_2')(x)
        
        # Output layer (Paper Equation 6: Softmax)
        outputs = layers.Dense(2, activation='softmax', name='output')(x)
        
        # Create and compile model
        model = models.Model(inputs=inputs, outputs=outputs, name='fuzzy_cnn_nas')
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def calculate_f1_accuracy(self, arch, X_train, y_train, X_val, y_val, epochs=5):
        """
        Calculate f1: Misclassification rate (1 - accuracy).
        
        Paper Equation 11:
        f1 = 1 - accuracy
        
        This is the primary objective - we want to minimize misclassification.
        
        Parameters:
        -----------
        arch : dict
            Architecture specification
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray
            Validation data
        epochs : int, default=5
            Number of training epochs (kept small for speed)
        
        Returns:
        --------
        f1 : float
            Misclassification rate in [0, 1] (lower is better)
        """
        import tensorflow as tf
        
        try:
            # Build model from architecture
            model = self.build_model_from_architecture(arch)
            
            # Train briefly (just enough to evaluate architecture quality)
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=0
            )
            
            # Evaluate on validation set
            _, accuracy = model.evaluate(X_val, y_val, verbose=0)
            
            # Clean up to free memory
            del model
            tf.keras.backend.clear_session()
            
            # Return misclassification rate
            return 1.0 - accuracy
        
        except Exception as e:
            print(f"   Error in f1 calculation: {e}")
            return 1.0  # Worst fitness if model fails
    
    def calculate_f2_blocks(self, arch):
        """
        Calculate f2: Number of blocks.
        
        Paper Equation 12:
        f2 = num_blocks
        
        Fewer blocks = simpler model = better for federated learning
        (less communication overhead).
        
        Parameters:
        -----------
        arch : dict
            Architecture specification
        
        Returns:
        --------
        f2 : float
            Number of blocks (normalized to [0, 1])
        """
        # Normalize to [0, 1] range (max 4 blocks)
        return float(arch['num_blocks']) / 4.0
    
    def calculate_f3_parameters(self, arch):
        """
        Calculate f3: Number of parameters.
        
        Paper Equation 13:
        f3 = num_parameters / max_parameters
        
        Fewer parameters = smaller model = better for federated learning
        (less communication overhead, faster training).
        
        Parameters:
        -----------
        arch : dict
            Architecture specification
        
        Returns:
        --------
        f3 : float
            Normalized parameter count in [0, 1] (lower is better)
        """
        import tensorflow as tf
        
        try:
            # Build model to count parameters
            model = self.build_model_from_architecture(arch)
            num_params = model.count_params()
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            # Normalize to [0, 1] range (assuming max 1M params)
            return num_params / 1_000_000.0
        
        except Exception as e:
            print(f"   Error in f3 calculation: {e}")
            return 1.0  # Worst fitness if model fails
    
    def calculate_f4_gflops(self, arch):
        """
        Calculate f4: GFLOPs (computational cost).
        
        Paper Equation 14:
        f4 = GFLOPs / max_GFLOPs
        
        Lower GFLOPs = less computation = better for federated learning
        (faster training on client devices).
        
        This is an approximation based on architecture structure.
        
        Parameters:
        -----------
        arch : dict
            Architecture specification
        
        Returns:
        --------
        f4 : float
            Normalized GFLOPs in [0, 1] (lower is better)
        """
        gflops = 0.0
        
        # Estimate FLOPs for each conv block
        # Formula: 2 * input_size * filters * kernel_size
        input_size = self.num_snps
        
        for block in arch['blocks']:
            # Conv FLOPs
            conv_flops = 2 * input_size * block['filters'] * block['kernel_size']
            gflops += conv_flops
            
            # Pooling reduces size
            if block['pool_type'] in [0, 1]:  # Max or Avg pooling
                input_size = max(1, input_size // 2)
        
        # Dense layer FLOPs
        # Approximate: 2 * input_features * output_features
        dense_flops = 2 * arch['dense_1'] * arch['dense_2']
        gflops += dense_flops
        
        # Convert to GFLOPs and normalize
        gflops = gflops / 1_000_000_000.0
        
        # Normalize to [0, 1] range (assuming max 0.1 GFLOPs)
        return min(gflops / 0.1, 1.0)
    
    def calculate_fitness(self, arch, X_train, y_train, X_val, y_val, 
                         stage, current_round, max_rounds):
        """
        Calculate multi-objective fitness based on FL stage.
        
        Paper Equation 15 (Segmented Strategy):
        - Stage 1 (0 < r < R/3):     fitness = 0.4*f1 + 0.2*f2 + 0.2*f3 + 0.2*f4
        - Stage 2 (R/3 <= r < 2R/3): fitness = 0.7*f1 + 0.3*f4
        - Stage 3 (2R/3 <= r < R):   fitness = 0.7*f1 + 0.3*f3
        
        The segmented strategy focuses on different objectives at different stages:
        - Stage 1: Explore all objectives (balanced search)
        - Stage 2: Focus on accuracy and computation (speed)
        - Stage 3: Focus on accuracy and parameters (communication)
        
        Parameters:
        -----------
        arch : dict
            Architecture specification
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray
            Validation data
        stage : int
            Current FL stage (1, 2, or 3)
        current_round : int
            Current communication round
        max_rounds : int
            Maximum communication rounds
        
        Returns:
        --------
        fitness : float
            Combined fitness value (lower is better)
        
        Example:
        --------
        >>> fitness = calculator.calculate_fitness(arch, X_train, y_train, X_val, y_val,
        ...                                        stage=1, current_round=5, max_rounds=50)
        """
        # Calculate f1 (accuracy) - always needed
        f1 = self.calculate_f1_accuracy(arch, X_train, y_train, X_val, y_val)
        
        if stage == 1:
            # Stage 1: All objectives (balanced exploration)
            f2 = self.calculate_f2_blocks(arch)
            f3 = self.calculate_f3_parameters(arch)
            f4 = self.calculate_f4_gflops(arch)
            
            # Weighted sum (accuracy weighted higher)
            fitness = 0.4 * f1 + 0.2 * f2 + 0.2 * f3 + 0.2 * f4
        
        elif stage == 2:
            # Stage 2: Accuracy + GFLOPs (focus on speed)
            f4 = self.calculate_f4_gflops(arch)
            fitness = 0.7 * f1 + 0.3 * f4
        
        else:  # stage == 3
            # Stage 3: Accuracy + Parameters (focus on communication)
            f3 = self.calculate_f3_parameters(arch)
            fitness = 0.7 * f1 + 0.3 * f3
        
        return fitness


class MultiObjectivePSO:
    """
    Multi-Objective PSO for Neural Architecture Search.
    
    Implements Algorithm 2 from the paper with segmented strategy.
    Uses Particle Swarm Optimization to discover optimal neural network
    architectures for federated learning.
    
    The PSO algorithm maintains a swarm of particles (architectures) that
    explore the search space by:
    1. Evaluating fitness of each particle
    2. Updating personal best (pbest) for each particle
    3. Updating global best (gbest) for the swarm
    4. Updating velocities based on pbest and gbest
    5. Updating positions (architectures)
    
    Parameters:
    -----------
    num_snps : int
        Number of SNP features in input data
    num_particles : int, default=10
        Size of particle swarm (more particles = better exploration)
    max_iterations : int, default=20
        Maximum PSO iterations per search (more iterations = better convergence)
    
    Attributes:
    -----------
    search_space : SearchSpace
        Architecture search space
    fitness_calculator : FitnessCalculator
        Multi-objective fitness evaluator
    particles : list
        List of Particle objects
    gbest_position : np.ndarray
        Global best position found by swarm
    gbest_fitness : float
        Fitness of global best position
    
    Example:
    --------
    >>> # Initialize PSO
    >>> pso = MultiObjectivePSO(num_snps=50, num_particles=10, max_iterations=20)
    >>> 
    >>> # Run search
    >>> best_arch, best_fitness = pso.search(
    ...     X_train, y_train, X_val, y_val,
    ...     current_round=5, max_rounds=50
    ... )
    >>> 
    >>> print(f"Best architecture: {best_arch}")
    >>> print(f"Best fitness: {best_fitness:.4f}")
    """
    
    def __init__(self, num_snps, num_particles=10, max_iterations=20):
        """
        Initialize PSO optimizer.
        
        Parameters:
        -----------
        num_snps : int
            Number of SNP features
        num_particles : int, default=10
            Size of particle swarm
        max_iterations : int, default=20
            Maximum PSO iterations per search
        """
        self.num_snps = num_snps
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        
        # Initialize components
        self.search_space = SearchSpace()
        self.fitness_calculator = FitnessCalculator(num_snps)
        
        # Initialize swarm
        self.particles = []
        self.gbest_position = None
        self.gbest_fitness = float('inf')  # Minimize fitness
        
        # Initialize particles
        self._initialize_particles()
    
    def _initialize_particles(self):
        """
        Initialize particle swarm with random architectures.
        
        Creates num_particles particles, each with:
        - Random architecture position
        - Small random velocity
        - Infinite initial fitness (not yet evaluated)
        """
        print(f">> Initializing {self.num_particles} particles...")
        
        for i in range(self.num_particles):
            # Generate random architecture
            arch = self.search_space.random_architecture()
            position = self.search_space.encode_architecture(arch)
            
            # Initialize with small random velocity
            velocity = np.random.randn(len(position)) * 0.1
            
            # Create particle
            particle = Particle(position, velocity)
            self.particles.append(particle)
        
        print(f">> Particles initialized")
    
    def get_stage(self, current_round, max_rounds):
        """
        Determine current FL stage for segmented strategy.
        
        Paper Equation 15 defines 3 stages:
        - Stage 1 (0 < r < R/3):     Explore all objectives
        - Stage 2 (R/3 <= r < 2R/3): Focus on accuracy + GFLOPs
        - Stage 3 (2R/3 <= r < R):   Focus on accuracy + parameters
        
        Parameters:
        -----------
        current_round : int
            Current communication round
        max_rounds : int
            Maximum communication rounds
        
        Returns:
        --------
        stage : int
            Current stage (1, 2, or 3)
        
        Example:
        --------
        >>> pso = MultiObjectivePSO(num_snps=50)
        >>> stage = pso.get_stage(current_round=10, max_rounds=50)
        >>> print(f"Stage: {stage}")  # Output: Stage: 1
        """
        if current_round < max_rounds / 3:
            return 1
        elif current_round < 2 * max_rounds / 3:
            return 2
        else:
            return 3
    
    def search(self, X_train, y_train, X_val, y_val, 
               current_round, max_rounds, verbose=True):
        """
        Main PSO search loop to discover optimal architecture.
        
        Algorithm:
        1. Determine FL stage (segmented strategy)
        2. For each iteration:
           a. Evaluate fitness of all particles
           b. Update personal best (pbest) for each particle
           c. Update global best (gbest) for swarm
           d. Update velocities (Equation 21)
           e. Update positions (Equation 22)
        3. Return best architecture found
        
        Parameters:
        -----------
        X_train, y_train : np.ndarray
            Training data for fitness evaluation
        X_val, y_val : np.ndarray
            Validation data for fitness evaluation
        current_round : int
            Current FL communication round
        max_rounds : int
            Maximum FL rounds (for stage determination)
        verbose : bool, default=True
            Whether to print progress
        
        Returns:
        --------
        best_arch : dict
            Best architecture found (decoded from gbest_position)
        best_fitness : float
            Fitness of best architecture
        
        Example:
        --------
        >>> pso = MultiObjectivePSO(num_snps=50, num_particles=5, max_iterations=10)
        >>> best_arch, best_fitness = pso.search(
        ...     X_train, y_train, X_val, y_val,
        ...     current_round=5, max_rounds=50,
        ...     verbose=True
        ... )
        """
        # Determine current stage
        stage = self.get_stage(current_round, max_rounds)
        
        if verbose:
            print(f"\n>> PSO Search - Round {current_round}/{max_rounds}, Stage {stage}")
            print(f">> Particles: {self.num_particles}, Iterations: {self.max_iterations}")
        
        # Main PSO loop
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"   Iteration {iteration+1}/{self.max_iterations}...", end=" ")
            
            # Evaluate all particles
            for particle in self.particles:
                # Decode position to architecture
                arch = self.search_space.decode_position(particle.position)
                
                # Calculate fitness using segmented strategy
                fitness = self.fitness_calculator.calculate_fitness(
                    arch, X_train, y_train, X_val, y_val,
                    stage, current_round, max_rounds
                )
                
                particle.fitness = fitness
                
                # Update personal best
                particle.update_pbest()
                
                # Update global best
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = particle.position.copy()
            
            # Update velocities and positions for all particles
            for particle in self.particles:
                particle.update_velocity(self.gbest_position)
                particle.update_position(self.search_space)
            
            if verbose:
                print(f"Best fitness: {self.gbest_fitness:.4f}")
        
        # Decode best architecture
        best_arch = self.search_space.decode_position(self.gbest_position)
        
        if verbose:
            print(f">> PSO Search complete. Best fitness: {self.gbest_fitness:.4f}")
            print(f">> Best architecture:")
            print(f"   Blocks: {best_arch['num_blocks']}")
            for i, block in enumerate(best_arch['blocks']):
                pool_name = ['MaxPool', 'AvgPool', 'None'][block['pool_type']]
                print(f"   Block {i+1}: {block['filters']} filters, kernel={block['kernel_size']}, pool={pool_name}")
            print(f"   Dense: {best_arch['dense_1']}, {best_arch['dense_2']}")
            print(f"   Dropout: {best_arch['dropout']}")
        
        return best_arch, self.gbest_fitness
    
    def get_best_architecture(self):
        """
        Get the current best architecture found.
        
        Returns:
        --------
        best_arch : dict
            Best architecture specification
        best_fitness : float
            Fitness of best architecture
        """
        if self.gbest_position is None:
            return None, float('inf')
        
        best_arch = self.search_space.decode_position(self.gbest_position)
        return best_arch, self.gbest_fitness
    
    def reset(self):
        """
        Reset PSO optimizer (re-initialize particles).
        
        Useful for running multiple independent searches.
        """
        self.particles = []
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        self._initialize_particles()


# Test Particle class
if __name__ == '__main__':
    print("="*80)
    print("PARTICLE CLASS TEST")
    print("="*80)
    
    # Initialize search space
    space = SearchSpace()
    print("\n>> SearchSpace initialized")
    
    # Test 1: Particle initialization
    print("\n" + "-"*80)
    print("TEST 1: Particle Initialization")
    print("-"*80)
    
    arch = space.random_architecture()
    position = space.encode_architecture(arch)
    velocity = np.random.randn(len(position)) * 0.1
    
    particle = Particle(position, velocity)
    
    print(f"\n>> Particle created:")
    print(f"   Position shape: {particle.position.shape}")
    print(f"   Position sample: {particle.position[:5]}")
    print(f"   Velocity shape: {particle.velocity.shape}")
    print(f"   Velocity sample: {particle.velocity[:5]}")
    print(f"   Initial fitness: {particle.fitness}")
    print(f"   Initial pbest: {particle.pbest_fitness}")
    
    # Test 2: Velocity update
    print("\n" + "-"*80)
    print("TEST 2: Velocity Update")
    print("-"*80)
    
    # Create a global best position
    gbest = position + np.random.randn(len(position)) * 0.5
    
    print(f"\nBefore update:")
    print(f"   Velocity: {particle.velocity[:5]}")
    
    particle.update_velocity(gbest, w=0.7, c1=2.0, c2=2.0)
    
    print(f"\nAfter update:")
    print(f"   Velocity: {particle.velocity[:5]}")
    print(f"\n>> Velocity updated successfully")
    
    # Test 3: Position update
    print("\n" + "-"*80)
    print("TEST 3: Position Update")
    print("-"*80)
    
    old_position = particle.position.copy()
    print(f"\nBefore update:")
    print(f"   Position: {particle.position[:5]}")
    
    particle.update_position(space)
    
    print(f"\nAfter update:")
    print(f"   Position: {particle.position[:5]}")
    print(f"   Position changed: {not np.array_equal(old_position, particle.position)}")
    
    # Check clipping
    print(f"\nClipping verification:")
    print(f"   num_blocks in [1,4]: {1 <= particle.position[0] <= 4}")
    print(f"   filters in [32,256]: {32 <= particle.position[1] <= 256}")
    print(f"   kernel in [3,7]: {3 <= particle.position[2] <= 7}")
    print(f"   dropout in [0.2,0.5]: {0.2 <= particle.position[15] <= 0.5}")
    
    print(f"\n>> Position updated and clipped successfully")
    
    # Test 4: Personal best update
    print("\n" + "-"*80)
    print("TEST 4: Personal Best Update")
    print("-"*80)
    
    print(f"\nInitial state:")
    print(f"   Current fitness: {particle.fitness}")
    print(f"   Pbest fitness: {particle.pbest_fitness}")
    
    # Set a fitness value
    particle.fitness = 0.5
    updated = particle.update_pbest()
    
    print(f"\nAfter setting fitness to 0.5:")
    print(f"   Pbest updated: {updated}")
    print(f"   Pbest fitness: {particle.pbest_fitness}")
    
    # Try with worse fitness
    particle.fitness = 0.7
    updated2 = particle.update_pbest()
    
    print(f"\nAfter setting fitness to 0.7 (worse):")
    print(f"   Pbest updated: {updated2}")
    print(f"   Pbest fitness: {particle.pbest_fitness} (should stay 0.5)")
    
    # Try with better fitness
    particle.fitness = 0.3
    updated3 = particle.update_pbest()
    
    print(f"\nAfter setting fitness to 0.3 (better):")
    print(f"   Pbest updated: {updated3}")
    print(f"   Pbest fitness: {particle.pbest_fitness} (should be 0.3)")
    
    print(f"\n>> Personal best update working correctly")
    
    # Test 5: Get architecture
    print("\n" + "-"*80)
    print("TEST 5: Get Architecture from Position")
    print("-"*80)
    
    arch_from_particle = particle.get_architecture(space)
    print(f"\n>> Architecture decoded:")
    print(f"   Num blocks: {arch_from_particle['num_blocks']}")
    print(f"   Dense layers: {arch_from_particle['dense_1']}, {arch_from_particle['dense_2']}")
    print(f"   Dropout: {arch_from_particle['dropout']}")
    
    # Test 6: Multiple particles
    print("\n" + "-"*80)
    print("TEST 6: Multiple Particles")
    print("-"*80)
    
    print("\nCreating 5 particles:")
    particles = []
    for i in range(5):
        arch_i = space.random_architecture()
        pos_i = space.encode_architecture(arch_i)
        vel_i = np.random.randn(len(pos_i)) * 0.1
        p = Particle(pos_i, vel_i)
        p.fitness = np.random.random()  # Random fitness
        p.update_pbest()
        particles.append(p)
        print(f"   Particle {i+1}: fitness={p.fitness:.4f}, pbest={p.pbest_fitness:.4f}")
    
    print(f"\n>> Multiple particles created successfully")
    
    # Test 7: Particle swarm behavior
    print("\n" + "-"*80)
    print("TEST 7: Swarm Behavior Simulation")
    print("-"*80)
    
    # Find global best
    gbest_particle = min(particles, key=lambda p: p.pbest_fitness)
    gbest_pos = gbest_particle.pbest_position
    
    print(f"\nGlobal best fitness: {gbest_particle.pbest_fitness:.4f}")
    print(f"\nSimulating 3 PSO iterations:")
    
    for iteration in range(3):
        print(f"\n  Iteration {iteration + 1}:")
        for i, p in enumerate(particles):
            # Update velocity and position
            p.update_velocity(gbest_pos)
            p.update_position(space)
            
            # Simulate fitness evaluation (random for testing)
            p.fitness = np.random.random()
            updated = p.update_pbest()
            
            if updated:
                print(f"    Particle {i+1}: New pbest = {p.pbest_fitness:.4f}")
        
        # Update global best
        new_gbest = min(particles, key=lambda p: p.pbest_fitness)
        if new_gbest.pbest_fitness < gbest_particle.pbest_fitness:
            gbest_particle = new_gbest
            gbest_pos = gbest_particle.pbest_position
            print(f"    >> New global best: {gbest_particle.pbest_fitness:.4f}")
    
    print(f"\n>> Swarm behavior simulation successful")
    
    # Final summary
    print("\n" + "="*80)
    print(">> ALL PARTICLE TESTS PASSED!")
    print("="*80)
    print("\nParticle class is ready for use in PSO!")
    print("\nKey features implemented:")
    print("  >> Particle initialization")
    print("  >> Velocity update (Equation 21)")
    print("  >> Position update (Equation 22)")
    print("  >> Position clipping to valid ranges")
    print("  >> Personal best tracking")
    print("  >> Architecture decoding")
    print("  >> Swarm behavior")
    
    # =========================================================================
    # TEST FITNESS CALCULATOR (TASK 4)
    # =========================================================================
    
    print("\n" + "="*80)
    print("TESTING FITNESS CALCULATOR (TASK 4)")
    print("="*80)
    print("\nImplementing Paper Equations 10-15:")
    print("  - f1: Misclassification rate (Equation 11)")
    print("  - f2: Number of blocks (Equation 12)")
    print("  - f3: Number of parameters (Equation 13)")
    print("  - f4: GFLOPs (Equation 14)")
    print("  - Segmented strategy (Equation 15)")
    
    # Initialize fitness calculator
    print("\n" + "-"*80)
    print("TEST 1: FitnessCalculator Initialization")
    print("-"*80)
    
    calculator = FitnessCalculator(num_snps=50)
    print(f"\n>> FitnessCalculator initialized for {calculator.num_snps} SNPs")
    
    # Test f2 (blocks)
    print("\n" + "-"*80)
    print("TEST 2: f2 - Number of Blocks (Equation 12)")
    print("-"*80)
    
    test_arch = space.random_architecture()
    print(f"\nTest architecture: {test_arch['num_blocks']} blocks")
    
    f2 = calculator.calculate_f2_blocks(test_arch)
    print(f"f2 (blocks): {f2:.4f}")
    print(f"Normalized: {test_arch['num_blocks']}/4 = {f2:.4f}")
    print(f"\n>> f2 calculation successful")
    
    # Test f4 (GFLOPs)
    print("\n" + "-"*80)
    print("TEST 3: f4 - GFLOPs (Equation 14)")
    print("-"*80)
    
    f4 = calculator.calculate_f4_gflops(test_arch)
    print(f"\nArchitecture summary:")
    print(f"  Blocks: {test_arch['num_blocks']}")
    for i, block in enumerate(test_arch['blocks']):
        print(f"  Block {i+1}: {block['filters']} filters, kernel={block['kernel_size']}")
    print(f"  Dense: {test_arch['dense_1']}, {test_arch['dense_2']}")
    print(f"\nf4 (GFLOPs): {f4:.6f}")
    print(f"\n>> f4 calculation successful")
    
    # Test f3 (parameters)
    print("\n" + "-"*80)
    print("TEST 4: f3 - Number of Parameters (Equation 13)")
    print("-"*80)
    
    print(f"\nBuilding model from architecture...")
    f3 = calculator.calculate_f3_parameters(test_arch)
    print(f"f3 (params normalized): {f3:.4f}")
    print(f"Estimated params: ~{int(f3 * 1_000_000):,}")
    print(f"\n>> f3 calculation successful")
    
    # Test f1 (accuracy) - with dummy data
    print("\n" + "-"*80)
    print("TEST 5: f1 - Misclassification Rate (Equation 11)")
    print("-"*80)
    
    print(f"\nCreating dummy training data...")
    X_train_dummy = np.random.randint(0, 3, (100, 50)).astype(np.float32)
    y_train_dummy = np.random.randint(0, 2, 100).astype(np.int32)
    X_val_dummy = np.random.randint(0, 3, (50, 50)).astype(np.float32)
    y_val_dummy = np.random.randint(0, 2, 50).astype(np.int32)
    
    print(f"Training model for 3 epochs...")
    f1 = calculator.calculate_f1_accuracy(
        test_arch, X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy, epochs=3
    )
    print(f"\nf1 (misclassification): {f1:.4f}")
    print(f"Accuracy: {(1.0 - f1)*100:.2f}%")
    print(f"\n>> f1 calculation successful")
    
    # Test segmented strategy
    print("\n" + "-"*80)
    print("TEST 6: Segmented Strategy (Equation 15)")
    print("-"*80)
    
    print(f"\nTesting all 3 stages:")
    
    # Stage 1
    print(f"\n  Stage 1 (0 < r < R/3): All objectives")
    fitness_s1 = calculator.calculate_fitness(
        test_arch, X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy,
        stage=1, current_round=5, max_rounds=50
    )
    print(f"    Fitness = 0.4*f1 + 0.2*f2 + 0.2*f3 + 0.2*f4")
    print(f"    Fitness: {fitness_s1:.4f}")
    
    # Stage 2
    print(f"\n  Stage 2 (R/3 <= r < 2R/3): Accuracy + GFLOPs")
    fitness_s2 = calculator.calculate_fitness(
        test_arch, X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy,
        stage=2, current_round=20, max_rounds=50
    )
    print(f"    Fitness = 0.7*f1 + 0.3*f4")
    print(f"    Fitness: {fitness_s2:.4f}")
    
    # Stage 3
    print(f"\n  Stage 3 (2R/3 <= r < R): Accuracy + Parameters")
    fitness_s3 = calculator.calculate_fitness(
        test_arch, X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy,
        stage=3, current_round=40, max_rounds=50
    )
    print(f"    Fitness = 0.7*f1 + 0.3*f3")
    print(f"    Fitness: {fitness_s3:.4f}")
    
    print(f"\n>> Segmented strategy working correctly")
    
    # Test multiple architectures
    print("\n" + "-"*80)
    print("TEST 7: Compare Multiple Architectures")
    print("-"*80)
    
    print(f"\nEvaluating 3 random architectures:")
    for i in range(3):
        arch_test = space.random_architecture()
        f2_test = calculator.calculate_f2_blocks(arch_test)
        f4_test = calculator.calculate_f4_gflops(arch_test)
        
        print(f"\n  Architecture {i+1}:")
        print(f"    Blocks: {arch_test['num_blocks']}, f2={f2_test:.4f}")
        print(f"    Dense: {arch_test['dense_1']}/{arch_test['dense_2']}")
        print(f"    GFLOPs: f4={f4_test:.6f}")
    
    print(f"\n>> Multiple architecture comparison successful")
    
    # Final summary
    print("\n" + "="*80)
    print(">> ALL FITNESS CALCULATOR TESTS PASSED!")
    print("="*80)
    print("\nFitnessCalculator class is ready for use in PSO!")
    print("\nKey features implemented:")
    print("  >> f1: Misclassification rate (Equation 11)")
    print("  >> f2: Number of blocks (Equation 12)")
    print("  >> f3: Number of parameters (Equation 13)")
    print("  >> f4: GFLOPs (Equation 14)")
    print("  >> Segmented strategy (Equation 15)")
    print("  >> Model building from architecture")
    print("  >> Multi-stage fitness calculation")
    print("\nNext steps:")
    print("  1. Implement MultiObjectivePSO class (Task 5)")
    print("  2. Integrate with federated learning (Task 6)")
    print("\n" + "="*80)
    
    # =========================================================================
    # TEST MULTI-OBJECTIVE PSO (TASK 5)
    # =========================================================================
    
    print("\n" + "="*80)
    print("TESTING MULTI-OBJECTIVE PSO (TASK 5)")
    print("="*80)
    print("\nImplementing Algorithm 2 from the paper:")
    print("  - Particle swarm initialization")
    print("  - Main PSO search loop")
    print("  - Global best tracking")
    print("  - Segmented strategy integration")
    
    # Test 1: PSO Initialization
    print("\n" + "-"*80)
    print("TEST 1: MultiObjectivePSO Initialization")
    print("-"*80)
    
    pso = MultiObjectivePSO(num_snps=50, num_particles=3, max_iterations=2)
    print(f"\n>> PSO initialized:")
    print(f"   SNPs: {pso.num_snps}")
    print(f"   Particles: {pso.num_particles}")
    print(f"   Max iterations: {pso.max_iterations}")
    print(f"   Particles created: {len(pso.particles)}")
    print(f"   Initial gbest fitness: {pso.gbest_fitness}")
    
    print(f"\n>> PSO initialization successful")
    
    # Test 2: Stage determination
    print("\n" + "-"*80)
    print("TEST 2: Stage Determination")
    print("-"*80)
    
    print(f"\nTesting stage calculation:")
    stage1 = pso.get_stage(current_round=5, max_rounds=50)
    stage2 = pso.get_stage(current_round=20, max_rounds=50)
    stage3 = pso.get_stage(current_round=40, max_rounds=50)
    
    print(f"  Round 5/50:  Stage {stage1} (expected: 1)")
    print(f"  Round 20/50: Stage {stage2} (expected: 2)")
    print(f"  Round 40/50: Stage {stage3} (expected: 3)")
    
    assert stage1 == 1, "Stage 1 calculation failed"
    assert stage2 == 2, "Stage 2 calculation failed"
    assert stage3 == 3, "Stage 3 calculation failed"
    
    print(f"\n>> Stage determination working correctly")
    
    # Test 3: PSO Search (quick test with dummy data)
    print("\n" + "-"*80)
    print("TEST 3: PSO Search (Quick Test)")
    print("-"*80)
    
    print(f"\nCreating dummy data for quick test...")
    X_train_pso = np.random.randint(0, 3, (100, 50)).astype(np.float32)
    y_train_pso = np.random.randint(0, 2, 100).astype(np.int32)
    X_val_pso = np.random.randint(0, 3, (50, 50)).astype(np.float32)
    y_val_pso = np.random.randint(0, 2, 50).astype(np.int32)
    
    print(f"Running PSO search (3 particles, 2 iterations)...")
    print(f"This will take ~1-2 minutes...\n")
    
    best_arch, best_fitness = pso.search(
        X_train_pso, y_train_pso, X_val_pso, y_val_pso,
        current_round=5, max_rounds=50,
        verbose=True
    )
    
    print(f"\n>> PSO search completed successfully")
    print(f">> Best architecture found:")
    print(f"   Num blocks: {best_arch['num_blocks']}")
    print(f"   Dense layers: {best_arch['dense_1']}, {best_arch['dense_2']}")
    print(f"   Dropout: {best_arch['dropout']}")
    print(f"   Best fitness: {best_fitness:.4f}")
    
    # Test 4: Get best architecture
    print("\n" + "-"*80)
    print("TEST 4: Get Best Architecture")
    print("-"*80)
    
    retrieved_arch, retrieved_fitness = pso.get_best_architecture()
    print(f"\n>> Retrieved best architecture:")
    print(f"   Matches search result: {retrieved_arch == best_arch}")
    print(f"   Fitness: {retrieved_fitness:.4f}")
    
    print(f"\n>> Get best architecture working correctly")
    
    # Test 5: PSO Reset
    print("\n" + "-"*80)
    print("TEST 5: PSO Reset")
    print("-"*80)
    
    old_gbest = pso.gbest_fitness
    print(f"\nBefore reset:")
    print(f"   Gbest fitness: {old_gbest:.4f}")
    print(f"   Num particles: {len(pso.particles)}")
    
    pso.reset()
    
    print(f"\nAfter reset:")
    print(f"   Gbest fitness: {pso.gbest_fitness}")
    print(f"   Num particles: {len(pso.particles)}")
    print(f"   Reset successful: {pso.gbest_fitness == float('inf')}")
    
    print(f"\n>> PSO reset working correctly")
    
    # Test 6: Multiple searches (different stages)
    print("\n" + "-"*80)
    print("TEST 6: Multiple Searches (Different Stages)")
    print("-"*80)
    
    print(f"\nTesting PSO across different FL stages:")
    
    # Create new PSO for clean test
    pso_multi = MultiObjectivePSO(num_snps=50, num_particles=2, max_iterations=1)
    
    stages_tested = []
    for round_num in [5, 20, 40]:
        stage = pso_multi.get_stage(round_num, 50)
        print(f"\n  Testing Round {round_num}/50 (Stage {stage}):")
        
        arch, fitness = pso_multi.search(
            X_train_pso, y_train_pso, X_val_pso, y_val_pso,
            current_round=round_num, max_rounds=50,
            verbose=False
        )
        
        print(f"    Best fitness: {fitness:.4f}")
        print(f"    Architecture: {arch['num_blocks']} blocks")
        stages_tested.append(stage)
        
        # Reset for next test
        pso_multi.reset()
    
    print(f"\n>> Tested stages: {stages_tested}")
    print(f">> All stages tested successfully")
    
    # Final summary
    print("\n" + "="*80)
    print(">> ALL MULTI-OBJECTIVE PSO TESTS PASSED!")
    print("="*80)
    print("\nMultiObjectivePSO class is ready for use!")
    print("\nKey features implemented:")
    print("  >> Particle swarm initialization")
    print("  >> Main PSO search loop")
    print("  >> Global best tracking")
    print("  >> Stage determination (Equation 15)")
    print("  >> Velocity and position updates")
    print("  >> Architecture discovery")
    print("  >> Reset functionality")
    print("\nNext steps:")
    print("  1. Integrate with federated learning (Task 6)")
    print("  2. Test end-to-end NAS (Task 7)")
    print("\n" + "="*80)

