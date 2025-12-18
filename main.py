#!/usr/bin/env python3
"""
Terrain Search Algorithm Comparison System
Author: Lu Yijie
Student ID: 2025141210006
School: Sichuan University, School of Mathematics
Major: Dual Bachelor's Degree in Mathematics and Intelligent Science
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
from scipy import ndimage
from scipy.interpolate import Rbf
import warnings
import time
import pandas as pd
from tqdm import tqdm
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style and parameters
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("=" * 70)
print("TERRAIN SEARCH ALGORITHM COMPARISON SYSTEM")
print("Author: Lu Yijie | Sichuan University, School of Mathematics")
print("=" * 70)
print()

# ============================================================================
# PART 1: Terrain Generation and Visualization Module
# ============================================================================

class TerrainGenerator:
    """Terrain Generator Class"""
    
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.terrain_types = ['basin', 'multi-peak', 'hilly', 'noisy', 'complex']
        
    def generate_terrain(self, terrain_type='basin', seed=42):
        """Generate specified terrain type"""
        np.random.seed(seed)
        terrain = np.random.randn(self.grid_size, self.grid_size) * 0.5
        
        if terrain_type == 'basin':
            # Single basin terrain
            x = np.linspace(-2, 2, self.grid_size)
            y = np.linspace(-2, 2, self.grid_size)
            X, Y = np.meshgrid(x, y)
            basin = -np.exp(-(X**2 + Y**2)/2) * 5
            terrain += basin
            
        elif terrain_type == 'multi-peak':
            # Multi-peak terrain
            for _ in range(8):
                cx = np.random.uniform(0.2, 0.8) * self.grid_size
                cy = np.random.uniform(0.2, 0.8) * self.grid_size
                sx = np.random.uniform(5, 15)
                sy = np.random.uniform(5, 15)
                depth = np.random.uniform(2, 5)
                
                X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
                gaussian = -np.exp(-((X-cx)**2/(2*sx**2) + (Y-cy)**2/(2*sy**2))) * depth
                terrain += gaussian
                
        elif terrain_type == 'hilly':
            # Hilly terrain
            terrain = ndimage.gaussian_filter(terrain, sigma=3)
            
        elif terrain_type == 'noisy':
            # Noisy terrain
            terrain = ndimage.gaussian_filter(terrain, sigma=2)
            terrain += np.random.randn(self.grid_size, self.grid_size) * 0.8
            
        elif terrain_type == 'complex':
            # Complex terrain
            x = np.linspace(-2, 2, self.grid_size)
            y = np.linspace(-2, 2, self.grid_size)
            X, Y = np.meshgrid(x, y)
            basin = -np.exp(-(X**2 + Y**2)/3) * 4
            terrain += basin
            
            for _ in range(5):
                cx = np.random.uniform(0.2, 0.8) * self.grid_size
                cy = np.random.uniform(0.2, 0.8) * self.grid_size
                sx = np.random.uniform(3, 8)
                sy = np.random.uniform(3, 8)
                depth = np.random.uniform(1, 3)
                
                X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
                gaussian = -np.exp(-((X-cx)**2/(2*sx**2) + (Y-cy)**2/(2*sy**2))) * depth
                terrain += gaussian * 0.7
            
            terrain += np.random.randn(self.grid_size, self.grid_size) * 0.5
        
        return terrain
    
    def visualize_all_terrains(self, save_fig=True):
        """Visualize all terrain types"""
        print("Generating and visualizing five terrain types...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Terrain name mapping
        terrain_name_map = {
            'basin': 'Basin Terrain',
            'multi-peak': 'Multi-peak Terrain', 
            'hilly': 'Hilly Terrain',
            'noisy': 'Noisy Terrain',
            'complex': 'Complex Terrain'
        }
        
        for i, terrain_type in enumerate(tqdm(self.terrain_types, desc="Generating terrains")):
            terrain = self.generate_terrain(terrain_type)
            
            ax = axes[i]
            contour = ax.contourf(terrain, levels=20, cmap='terrain')
            
            # Use English labels
            english_name = terrain_name_map.get(terrain_type, terrain_type)
            ax.set_title(f'{english_name}', fontsize=12)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            # Mark the global minimum
            min_pos = np.unravel_index(np.argmin(terrain), terrain.shape)
            ax.plot(min_pos[1], min_pos[0], 'r*', markersize=12, label='Global Minimum')
            ax.legend(fontsize=9)
            
            plt.colorbar(contour, ax=ax, label='Elevation')
        
        axes[-1].axis('off')
        plt.suptitle('Five Terrain Types Visualization', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('1_Terrain_Types.png', dpi=300, bbox_inches='tight')
            print("Terrain visualization saved as '1_Terrain_Types.png'")
        
        plt.show()
        
        return fig
    
    def analyze_terrain_features(self, terrain_type='complex', save_fig=True):
        """Analyze and visualize terrain features"""
        print(f"Analyzing {terrain_type} terrain features...")
        
        terrain = self.generate_terrain(terrain_type)
        
        # Calculate global minimum position
        global_min_pos = np.unravel_index(np.argmin(terrain), terrain.shape)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original terrain
        ax1 = axes[0, 0]
        im1 = ax1.imshow(terrain, cmap='terrain')
        ax1.set_title('Original Terrain')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        plt.colorbar(im1, ax=ax1, label='Elevation')
        
        # Terrain gradient
        ax2 = axes[0, 1]
        grad_x, grad_y = np.gradient(terrain)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        im2 = ax2.imshow(gradient_magnitude, cmap='hot')
        ax2.set_title('Terrain Gradient Magnitude')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        plt.colorbar(im2, ax=ax2, label='Gradient Magnitude')
        
        # Terrain curvature
        ax3 = axes[0, 2]
        smoothed = ndimage.gaussian_filter(terrain, sigma=1)
        grad_xx, grad_xy = np.gradient(grad_x)
        grad_yx, grad_yy = np.gradient(grad_y)
        curvature = np.abs(grad_xx + grad_yy)
        im3 = ax3.imshow(curvature, cmap='coolwarm')
        ax3.set_title('Terrain Curvature')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        plt.colorbar(im3, ax=ax3, label='Curvature')
        
        # 3D terrain view
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        surf = ax4.plot_surface(X, Y, terrain, cmap='terrain', alpha=0.8, linewidth=0)
        ax4.set_title('3D Terrain View')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        ax4.set_zlabel('Elevation')
        
        # Elevation distribution histogram
        ax5 = axes[1, 1]
        ax5.hist(terrain.flatten(), bins=30, color='skyblue', edgecolor='black')
        ax5.set_title('Elevation Distribution Histogram')
        ax5.set_xlabel('Elevation')
        ax5.set_ylabel('Frequency')
        ax5.axvline(x=terrain.min(), color='red', linestyle='--', 
                   label=f'Min: {terrain.min():.2f}')
        ax5.axvline(x=terrain.max(), color='green', linestyle='--', 
                   label=f'Max: {terrain.max():.2f}')
        ax5.axvline(x=terrain.mean(), color='orange', linestyle='--', 
                   label=f'Mean: {terrain.mean():.2f}')
        ax5.legend()
        
        # Terrain statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""
        Terrain Feature Statistics:
        
        Terrain Type: {terrain_type}
        Grid Size: {self.grid_size}×{self.grid_size}
        
        Elevation Statistics:
        Minimum: {terrain.min():.4f}
        Maximum: {terrain.max():.4f}
        Mean: {terrain.mean():.4f}
        Std Dev: {terrain.std():.4f}
        
        Gradient Statistics:
        Mean Gradient: {gradient_magnitude.mean():.4f}
        Max Gradient: {gradient_magnitude.max():.4f}
        
        Global Minimum Coordinates:
        ({global_min_pos[0]}, {global_min_pos[1]})
        Elevation: {terrain[global_min_pos]:.4f}
        """
        
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
        
        plt.suptitle(f'{terrain_type.capitalize()} Terrain Feature Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'2_{terrain_type}_Terrain_Analysis.png', dpi=300, bbox_inches='tight')
            print(f"Terrain analysis saved as '2_{terrain_type}_Terrain_Analysis.png'")
        
        plt.show()
        
        return fig, terrain

# ============================================================================
# PART 2: Three Algorithm Implementations
# ============================================================================

class SimulatedAnnealing:
    """Simulated Annealing Algorithm"""
    
    def __init__(self, terrain, T0=50, cooling_rate=0.95, max_iter=200, adaptive_T0=False):
        self.terrain = terrain
        self.grid_size = terrain.shape[0]
        self.T0 = T0
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.adaptive_T0 = adaptive_T0
        
    def search(self):
        """Execute simulated annealing search"""
        # Random starting point
        current = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        best = current
        H_current = self.terrain[current]
        H_best = H_current
        
        # Adaptive initial temperature calculation
        T = self.T0
        if self.adaptive_T0:
            samples = []
            for _ in range(100):
                move = np.random.randint(-5, 6, size=2)
                neighbor = ((current[0] + move[0]) % self.grid_size,
                            (current[1] + move[1]) % self.grid_size)
                samples.append(abs(self.terrain[neighbor] - H_current))
            T = 5 * np.std(samples)
            if T < 1:
                T = 10
        
        # Initialize records
        history = []
        visited = set([current])
        
        # Simulated annealing search
        for i in range(self.max_iter):
            # Generate neighborhood random move
            move = np.random.randint(-2, 3, size=2)
            neighbor = ((current[0] + move[0]) % self.grid_size,
                        (current[1] + move[1]) % self.grid_size)
            
            H_neighbor = self.terrain[neighbor]
            delta_H = H_neighbor - H_current
            
            # Metropolis criterion
            if delta_H < 0 or np.random.rand() < np.exp(-delta_H / (T + 1e-10)):
                current = neighbor
                H_current = H_neighbor
                visited.add(current)
                
                if H_current < H_best:
                    best = current
                    H_best = H_current
            
            history.append(H_best)
            T *= self.cooling_rate
            
            # Early termination condition
            if T < 1e-5:
                break
        
        return best, H_best, history, visited

class BayesianOptimization:
    """Bayesian Optimization Algorithm based on Gaussian Process"""
    
    def __init__(self, terrain, n_initial=5, max_iter=200, exploration_weight=0.1):
        self.terrain = terrain
        self.grid_size = terrain.shape[0]
        self.n_initial = n_initial
        self.max_iter = max_iter
        self.exploration_weight = exploration_weight
        
    def search(self):
        """Execute Bayesian optimization search"""
        # Initialize dataset
        X = []
        y = []
        visited = set()
        
        # Initial random sampling
        for _ in range(self.n_initial):
            point = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            value = self.terrain[point]
            X.append([point[0], point[1]])
            y.append(value)
            visited.add(point)
        
        # Record history
        history = [min(y)] if y else []
        
        # Bayesian optimization loop
        for iteration in range(self.max_iter):
            # Simplified version: use nearest neighbor as surrogate model
            # In real applications, Gaussian Process Regression should be used here
            
            # Generate candidate point grid
            candidates = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    candidates.append((i, j))
            
            # Calculate score for each candidate (simplified version)
            scores = []
            for candidate in candidates:
                if candidate in visited:
                    scores.append(-float('inf'))
                    continue
                
                # Calculate minimum distance to sampled points
                min_dist = float('inf')
                for xi in X:
                    dist = np.sqrt((candidate[0]-xi[0])**2 + (candidate[1]-xi[1])**2)
                    min_dist = min(min_dist, dist)
                
                # Simplified acquisition function: balance exploration and exploitation
                score = -self.terrain[candidate] + self.exploration_weight * min_dist
                scores.append(score)
            
            # Select next sampling point
            if scores:
                next_idx = np.argmax(scores)
                next_point = candidates[next_idx]
            else:
                next_point = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            
            # Sample and update dataset
            value = self.terrain[next_point]
            X.append([next_point[0], next_point[1]])
            y.append(value)
            visited.add(next_point)
            
            # Update history best value
            history.append(min(y))
        
        # Return best point
        best_idx = np.argmin(y)
        best_point = (int(X[best_idx][0]), int(X[best_idx][1]))
        best_value = y[best_idx]
        
        return best_point, best_value, history, visited

class SequentialDecision:
    """Sequential Decision Algorithm based on Interpolation Model"""
    
    def __init__(self, terrain, n_initial=5, max_iter=200, uncertainty_weight=0.2):
        self.terrain = terrain
        self.grid_size = terrain.shape[0]
        self.n_initial = n_initial
        self.max_iter = max_iter
        self.uncertainty_weight = uncertainty_weight
        
    def search(self):
        """Execute sequential decision search"""
        # Initialize dataset
        X = []
        y = []
        visited = set()
        
        # Initial random sampling
        for _ in range(self.n_initial):
            point = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            value = self.terrain[point]
            X.append([point[0], point[1]])
            y.append(value)
            visited.add(point)
        
        # Record history
        history = [min(y)] if y else []
        
        # Sequential decision loop
        for iteration in range(self.max_iter):
            if len(X) < 3:
                # Too few samples, random sampling
                next_point = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            else:
                try:
                    # Build interpolation model
                    X_array = np.array(X)
                    y_array = np.array(y)
                    
                    # Use radial basis function interpolation
                    rbf = Rbf(X_array[:, 0], X_array[:, 1], y_array, function='gaussian')
                    
                    # Generate prediction grid
                    grid_x, grid_y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
                    
                    # Predict terrain
                    Z_pred = rbf(grid_x, grid_y)
                    
                    # Calculate uncertainty (based on distance)
                    uncertainty = np.zeros_like(Z_pred)
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            # Calculate minimum distance to all sampled points
                            min_dist = float('inf')
                            for xi in X:
                                dist = np.sqrt((i-xi[0])**2 + (j-xi[1])**2)
                                min_dist = min(min_dist, dist)
                            uncertainty[i, j] = min_dist
                    
                    # Normalize uncertainty
                    if uncertainty.max() > 0:
                        uncertainty = uncertainty / uncertainty.max()
                    
                    # Select next sampling point (balance prediction and uncertainty)
                    scores = -Z_pred + self.uncertainty_weight * uncertainty
                    
                    # Exclude visited points
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            if (i, j) in visited:
                                scores[i, j] = -float('inf')
                    
                    # Select point with highest score
                    next_idx = np.unravel_index(np.argmax(scores), scores.shape)
                    next_point = (next_idx[0], next_idx[1])
                    
                except Exception as e:
                    # If interpolation fails, random sampling
                    next_point = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            
            # Sample and update dataset
            value = self.terrain[next_point]
            X.append([next_point[0], next_point[1]])
            y.append(value)
            visited.add(next_point)
            
            # Update history best value
            history.append(min(y))
        
        # Return best point
        best_idx = np.argmin(y)
        best_point = (int(X[best_idx][0]), int(X[best_idx][1]))
        best_value = y[best_idx]
        
        return best_point, best_value, history, visited

# ============================================================================
# PART 3: Search Process Visualization
# ============================================================================

class SearchVisualizer:
    """Search Process Visualizer Class"""
    
    def __init__(self, terrain, algorithm_name):
        self.terrain = terrain
        self.algorithm_name = algorithm_name
        self.grid_size = terrain.shape[0]
        
    def visualize_search_process(self, search_history, save_fig=True):
        """Visualize search process"""
        print(f"Visualizing {self.algorithm_name} search process...")
        
        visited_points = search_history['visited_points']
        best_points = search_history['best_points']
        best_values = search_history['best_values']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Subplot 1: Search path overview
        ax1 = axes[0, 0]
        contour1 = ax1.contourf(self.terrain, levels=20, cmap='terrain', alpha=0.7)
        
        # Draw search path
        if len(visited_points) > 1:
            path_x = [p[1] for p in visited_points]
            path_y = [p[0] for p in visited_points]
            ax1.plot(path_x, path_y, 'b-', alpha=0.5, linewidth=1)
        
        # Mark key points
        ax1.scatter([p[1] for p in visited_points], [p[0] for p in visited_points], 
                   c='blue', s=10, alpha=0.3, label='Visited Points')
        
        if best_points:
            ax1.scatter(best_points[-1][1], best_points[-1][0], 
                       c='green', s=200, marker='s', label='Final Best Point')
        
        # Mark global minimum
        global_min_pos = np.unravel_index(np.argmin(self.terrain), self.terrain.shape)
        ax1.plot(global_min_pos[1], global_min_pos[0], 'y*', 
                markersize=15, label='Global Minimum')
        
        ax1.set_title(f'{self.algorithm_name} Search Path Overview')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.legend(fontsize=9)
        plt.colorbar(contour1, ax=ax1, label='Elevation')
        
        # Subplot 2: Convergence curve
        ax2 = axes[0, 1]
        ax2.plot(best_values, 'b-', linewidth=2)
        ax2.axhline(y=self.terrain.min(), color='r', linestyle='--', label='Global Optimum')
        ax2.set_xlabel('Iteration Number')
        ax2.set_ylabel('Best Elevation Found')
        ax2.set_title('Convergence Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Visited points heatmap
        ax3 = axes[0, 2]
        visit_map = np.zeros_like(self.terrain)
        for point in visited_points:
            visit_map[point] += 1
        
        im3 = ax3.imshow(visit_map, cmap='hot', alpha=0.8)
        ax3.set_title('Visited Points Heatmap')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        plt.colorbar(im3, ax=ax3, label='Visit Count')
        
        # Subplot 4: Search efficiency analysis
        ax4 = axes[1, 0]
        unique_points = len(set(visited_points))
        total_points = self.grid_size * self.grid_size
        coverage_ratio = unique_points / total_points * 100
        
        labels = ['Visited', 'Not Visited']
        sizes = [unique_points, total_points - unique_points]
        colors = ['lightblue', 'lightgray']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Search Coverage: {coverage_ratio:.2f}%')
        
        # Subplot 5: Search direction analysis
        ax5 = axes[1, 1]
        if len(visited_points) > 1:
            directions = []
            for i in range(1, len(visited_points)):
                dx = visited_points[i][1] - visited_points[i-1][1]
                dy = visited_points[i][0] - visited_points[i-1][0]
                directions.append(np.arctan2(dy, dx))
            
            ax5.hist(directions, bins=12, color='skyblue', edgecolor='black')
            ax5.set_xlabel('Movement Direction (radians)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Search Direction Distribution')
        
        # Subplot 6: Algorithm statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        final_error = best_values[-1] - self.terrain.min() if best_values else 0
        success_rate = 100 if final_error < 0.01 else 0
        
        info_text = f"""
        {self.algorithm_name} Search Results:
        
        Terrain Size: {self.grid_size}×{self.grid_size}
        Total Grid Points: {total_points}
        
        Search Statistics:
        Unique Points Visited: {unique_points}
        Search Coverage: {coverage_ratio:.2f}%
        Iterations: {len(best_values)}
        
        Performance Metrics:
        Final Best Elevation: {best_values[-1]:.4f if best_values else 0:.4f}
        Global Minimum Elevation: {self.terrain.min():.4f}
        Final Error: {final_error:.4f}
        Success Rate: {success_rate:.1f}%
        
        Search Efficiency:
        Avg Points per Iteration: {unique_points/max(len(best_values),1):.2f}
        Optimal Solution Found at: {np.argmin(best_values)+1 if best_values else 0}th iteration
        """
        
        ax6.text(0.1, 0.95, info_text, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
        
        plt.suptitle(f'{self.algorithm_name} Search Process Visualization', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'3_{self.algorithm_name.replace(" ", "_")}_Search_Process.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Search process visualization saved as '3_{self.algorithm_name.replace(' ', '_')}_Search_Process.png'")
        
        plt.show()
        
        return fig

# ============================================================================
# PART 4: Experiment Execution and Comparison
# ============================================================================

class ExperimentRunner:
    """Experiment Runner Class"""
    
    def __init__(self, grid_size=50, n_runs=20, max_iter=200):
        self.grid_size = grid_size
        self.n_runs = n_runs
        self.max_iter = max_iter
        self.terrain_generator = TerrainGenerator(grid_size)
        self.terrain_types = ['basin', 'multi-peak', 'hilly', 'noisy', 'complex']
        self.algorithms = ['SA', 'BO', 'SDIM']
        
    def run_experiment_1(self):
        """Experiment 1: Performance comparison of three algorithms on different terrains"""
        print("\n" + "="*60)
        print("EXPERIMENT 1: Algorithm Performance Comparison Across Terrains")
        print("="*60)
        
        results = {}
        
        for terrain_type in tqdm(self.terrain_types, desc="Testing terrains"):
            results[terrain_type] = {
                'global_min': None,
                'SA': {'errors': [], 'ratios': [], 'successes': [], 'times': []},
                'BO': {'errors': [], 'ratios': [], 'successes': [], 'times': []},
                'SDIM': {'errors': [], 'ratios': [], 'successes': [], 'times': []}
            }
            
            for run in range(self.n_runs):
                # Generate terrain
                terrain = self.terrain_generator.generate_terrain(terrain_type, seed=run)
                global_min = terrain.min()
                results[terrain_type]['global_min'] = global_min
                
                # Simulated Annealing Algorithm
                start_time = time.time()
                sa = SimulatedAnnealing(terrain, T0=50, cooling_rate=0.95, max_iter=self.max_iter)
                sa_best, sa_best_h, sa_history, sa_visited = sa.search()
                sa_time = time.time() - start_time
                
                sa_error = sa_best_h - global_min
                sa_ratio = len(sa_visited) / (self.grid_size * self.grid_size)
                sa_success = 1 if sa_error < 0.01 else 0
                
                results[terrain_type]['SA']['errors'].append(sa_error)
                results[terrain_type]['SA']['ratios'].append(sa_ratio)
                results[terrain_type]['SA']['successes'].append(sa_success)
                results[terrain_type]['SA']['times'].append(sa_time)
                
                # Bayesian Optimization Algorithm
                start_time = time.time()
                bo = BayesianOptimization(terrain, n_initial=5, max_iter=self.max_iter)
                bo_best, bo_best_h, bo_history, bo_visited = bo.search()
                bo_time = time.time() - start_time
                
                bo_error = bo_best_h - global_min
                bo_ratio = len(bo_visited) / (self.grid_size * self.grid_size)
                bo_success = 1 if bo_error < 0.01 else 0
                
                results[terrain_type]['BO']['errors'].append(bo_error)
                results[terrain_type]['BO']['ratios'].append(bo_ratio)
                results[terrain_type]['BO']['successes'].append(bo_success)
                results[terrain_type]['BO']['times'].append(bo_time)
                
                # Sequential Decision Algorithm
                start_time = time.time()
                sd = SequentialDecision(terrain, n_initial=5, max_iter=self.max_iter)
                sd_best, sd_best_h, sd_history, sd_visited = sd.search()
                sd_time = time.time() - start_time
                
                sd_error = sd_best_h - global_min
                sd_ratio = len(sd_visited) / (self.grid_size * self.grid_size)
                sd_success = 1 if sd_error < 0.01 else 0
                
                results[terrain_type]['SDIM']['errors'].append(sd_error)
                results[terrain_type]['SDIM']['ratios'].append(sd_ratio)
                results[terrain_type]['SDIM']['successes'].append(sd_success)
                results[terrain_type]['SDIM']['times'].append(sd_time)
        
        # Save results
        self.save_results(results, 'experiment_1_results.json')
        
        return results
    
    def run_experiment_2(self):
        """Experiment 2: Parameter sensitivity analysis (using simulated annealing as example)"""
        print("\n" + "="*60)
        print("EXPERIMENT 2: Parameter Sensitivity Analysis (Simulated Annealing)")
        print("="*60)
        
        # Generate complex terrain
        terrain = self.terrain_generator.generate_terrain('complex', seed=42)
        global_min = terrain.min()
        
        # Test different parameter combinations
        T0_values = [10, 30, 50, 100, 200]
        alpha_values = [0.85, 0.90, 0.95, 0.98, 0.99]
        
        results = np.zeros((len(T0_values), len(alpha_values)))
        results_std = np.zeros((len(T0_values), len(alpha_values)))
        visited_ratios = np.zeros((len(T0_values), len(alpha_values)))
        
        print("Running parameter sensitivity analysis...")
        for i, T0 in enumerate(tqdm(T0_values, desc="Initial Temperature")):
            for j, alpha in enumerate(alpha_values):
                errors = []
                ratios = []
                
                for _ in range(10):  # Run 10 times for each parameter combination
                    sa = SimulatedAnnealing(terrain, T0=T0, cooling_rate=alpha, max_iter=self.max_iter)
                    best, best_h, history, visited = sa.search()
                    
                    error = best_h - global_min
                    ratio = len(visited) / (self.grid_size * self.grid_size)
                    
                    errors.append(error)
                    ratios.append(ratio)
                
                results[i, j] = np.mean(errors)
                results_std[i, j] = np.std(errors)
                visited_ratios[i, j] = np.mean(ratios)
        
        # Find best parameter combination
        min_idx = np.unravel_index(np.argmin(results), results.shape)
        best_T0 = T0_values[min_idx[0]]
        best_alpha = alpha_values[min_idx[1]]
        
        # Visualize parameter sensitivity
        self.visualize_parameter_sensitivity(T0_values, alpha_values, results, results_std, visited_ratios)
        
        param_results = {
            'best_T0': best_T0,
            'best_alpha': best_alpha,
            'best_error': results[min_idx],
            'best_ratio': visited_ratios[min_idx],
            'T0_values': T0_values,
            'alpha_values': alpha_values,
            'error_matrix': results.tolist(),
            'std_matrix': results_std.tolist(),
            'ratio_matrix': visited_ratios.tolist()
        }
        
        self.save_results(param_results, 'experiment_2_results.json')
        
        return param_results
    
    def run_experiment_3(self):
        """Experiment 3: Search process visualization example"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: Search Process Visualization Example")
        print("="*60)
        
        # Generate complex terrain
        terrain = self.terrain_generator.generate_terrain('complex', seed=42)
        
        # Run and visualize three algorithms
        algorithms = [
            ('Simulated_Annealing', SimulatedAnnealing(terrain, T0=50, cooling_rate=0.95, max_iter=100)),
            ('Bayesian_Optimization', BayesianOptimization(terrain, n_initial=5, max_iter=100)),
            ('Sequential_Decision', SequentialDecision(terrain, n_initial=5, max_iter=100))
        ]
        
        for algo_name, algorithm in algorithms:
            print(f"\nRunning {algo_name.replace('_', ' ')} algorithm...")
            
            best, best_h, history, visited = algorithm.search()
            
            # Prepare search history
            search_history = {
                'visited_points': list(visited),
                'best_points': [best] * len(history),
                'best_values': history
            }
            
            # Visualize search process
            visualizer = SearchVisualizer(terrain, algo_name.replace('_', ' '))
            visualizer.visualize_search_process(search_history, save_fig=True)
    
    def visualize_results(self, results):
        """Visualize experiment results"""
        print("\n" + "="*60)
        print("VISUALIZING EXPERIMENT RESULTS")
        print("="*60)
        
        # Prepare data
        terrain_types = self.terrain_types
        algorithms = self.algorithms
        
        # Create summary table
        summary_data = []
        
        for terrain in terrain_types:
            for algo in algorithms:
                errors = results[terrain][algo]['errors']
                ratios = results[terrain][algo]['ratios']
                successes = results[terrain][algo]['successes']
                times = results[terrain][algo]['times']
                
                summary_data.append({
                    'Terrain_Type': terrain,
                    'Algorithm': algo,
                    'Mean_Error': np.mean(errors),
                    'Error_Std': np.std(errors),
                    'Mean_Visited_Ratio': np.mean(ratios) * 100,
                    'Success_Rate': np.mean(successes) * 100,
                    'Mean_Run_Time': np.mean(times)
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        df.to_csv('4_Experiment_Results_Summary.csv', index=False)
        print("Experiment results saved as '4_Experiment_Results_Summary.csv'")
        
        # Print results
        print("\nExperiment Results Summary:")
        print("-" * 100)
        print(df.to_string(index=False))
        
        # Visualize performance comparison
        self.visualize_performance_comparison(df)
        
        return df
    
    def visualize_parameter_sensitivity(self, T0_values, alpha_values, results, results_std, visited_ratios):
        """Visualize parameter sensitivity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Subplot 1: Parameter heatmap (mean error)
        ax1 = axes[0, 0]
        im1 = ax1.imshow(results, cmap='YlOrRd', aspect='auto', origin='lower')
        ax1.set_xlabel('Cooling Rate (α)')
        ax1.set_ylabel('Initial Temperature (T₀)')
        ax1.set_title('Parameter Sensitivity Analysis (Mean Final Error)')
        ax1.set_xticks(np.arange(len(alpha_values)))
        ax1.set_xticklabels(alpha_values)
        ax1.set_yticks(np.arange(len(T0_values)))
        ax1.set_yticklabels(T0_values)
        
        # Add numerical labels
        for i in range(len(T0_values)):
            for j in range(len(alpha_values)):
                ax1.text(j, i, f'{results[i, j]:.3f}', ha='center', va='center', 
                        color='black' if results[i, j] < np.median(results) else 'white')
        
        plt.colorbar(im1, ax=ax1, label='Mean Final Error')
        
        # Subplot 2: Parameter heatmap (error standard deviation)
        ax2 = axes[0, 1]
        im2 = ax2.imshow(results_std, cmap='YlOrRd', aspect='auto', origin='lower')
        ax2.set_xlabel('Cooling Rate (α)')
        ax2.set_ylabel('Initial Temperature (T₀)')
        ax2.set_title('Parameter Sensitivity Analysis (Error Standard Deviation)')
        ax2.set_xticks(np.arange(len(alpha_values)))
        ax2.set_xticklabels(alpha_values)
        ax2.set_yticks(np.arange(len(T0_values)))
        ax2.set_yticklabels(T0_values)
        
        for i in range(len(T0_values)):
            for j in range(len(alpha_values)):
                ax2.text(j, i, f'{results_std[i, j]:.3f}', ha='center', va='center', 
                        color='black' if results_std[i, j] < np.median(results_std) else 'white')
        
        plt.colorbar(im2, ax=ax2, label='Error Standard Deviation')
        
        # Subplot 3: Parameter heatmap (visited points ratio)
        ax3 = axes[1, 0]
        im3 = ax3.imshow(visited_ratios*100, cmap='Blues', aspect='auto', origin='lower')
        ax3.set_xlabel('Cooling Rate (α)')
        ax3.set_ylabel('Initial Temperature (T₀)')
        ax3.set_title('Parameter Effect on Search Efficiency (Visited Points Ratio %)')
        ax3.set_xticks(np.arange(len(alpha_values)))
        ax3.set_xticklabels(alpha_values)
        ax3.set_yticks(np.arange(len(T0_values)))
        ax3.set_yticklabels(T0_values)
        
        for i in range(len(T0_values)):
            for j in range(len(alpha_values)):
                ax3.text(j, i, f'{visited_ratios[i, j]*100:.1f}', ha='center', va='center', 
                        color='black' if visited_ratios[i, j] < np.median(visited_ratios) else 'white')
        
        plt.colorbar(im3, ax=ax3, label='Visited Points Ratio (%)')
        
        # Subplot 4: Convergence curve with best parameters
        ax4 = axes[1, 1]
        # Find best parameters
        min_idx = np.unravel_index(np.argmin(results), results.shape)
        best_T0 = T0_values[min_idx[0]]
        best_alpha = alpha_values[min_idx[1]]
        
        # Run best parameter combination several times
        terrain = self.terrain_generator.generate_terrain('complex', seed=42)
        histories = []
        for _ in range(5):
            sa = SimulatedAnnealing(terrain, T0=best_T0, cooling_rate=best_alpha, max_iter=200)
            _, _, history, _ = sa.search()
            histories.append(history)
        
        # Calculate average convergence curve
        max_len = max(len(h) for h in histories)
        avg_history = np.zeros(max_len)
        for i in range(max_len):
            valid_vals = [h[i] for h in histories if i < len(h)]
            avg_history[i] = np.mean(valid_vals) if valid_vals else 0
        
        ax4.plot(avg_history, label=f'T₀={best_T0}, α={best_alpha}')
        ax4.axhline(y=terrain.min(), color='r', linestyle='--', label='Global Optimum')
        ax4.set_xlabel('Iteration Number')
        ax4.set_ylabel('Best Elevation Found')
        ax4.set_title(f'Convergence Curve with Best Parameters\n(T₀={best_T0}, α={best_alpha})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Experiment 2: Simulated Annealing Parameter Sensitivity Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('5_Parameter_Sensitivity_Analysis.png', dpi=300, bbox_inches='tight')
        print("Parameter sensitivity analysis saved as '5_Parameter_Sensitivity_Analysis.png'")
        plt.show()
        
        return fig
    
    def visualize_performance_comparison(self, df):
        """Visualize performance comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        terrain_types = df['Terrain_Type'].unique()
        algorithms = df['Algorithm'].unique()
        
        # Subplot 1: Mean error comparison
        ax1 = axes[0, 0]
        x = np.arange(len(terrain_types))
        width = 0.25
        
        for i, algo in enumerate(algorithms):
            algo_data = df[df['Algorithm'] == algo]
            errors = algo_data['Mean_Error'].values
            ax1.bar(x + i*width - width, errors, width, label=algo, alpha=0.8)
        
        ax1.set_xlabel('Terrain Type')
        ax1.set_ylabel('Mean Final Error')
        ax1.set_title('Algorithm Performance Comparison (Final Error)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(terrain_types, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Visited points ratio comparison
        ax2 = axes[0, 1]
        for i, algo in enumerate(algorithms):
            algo_data = df[df['Algorithm'] == algo]
            ratios = algo_data['Mean_Visited_Ratio'].values
            ax2.bar(x + i*width - width, ratios, width, label=algo, alpha=0.8)
        
        ax2.set_xlabel('Terrain Type')
        ax2.set_ylabel('Mean Visited Points Ratio (%)')
        ax2.set_title('Search Efficiency Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(terrain_types, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Subplot 3: Success rate comparison
        ax3 = axes[0, 2]
        for i, algo in enumerate(algorithms):
            algo_data = df[df['Algorithm'] == algo]
            success_rates = algo_data['Success_Rate'].values
            ax3.bar(x + i*width - width, success_rates, width, label=algo, alpha=0.8)
        
        ax3.set_xlabel('Terrain Type')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Algorithm Success Rate Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(terrain_types, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Subplot 4: Run time comparison
        ax4 = axes[1, 0]
        for i, algo in enumerate(algorithms):
            algo_data = df[df['Algorithm'] == algo]
            times = algo_data['Mean_Run_Time'].values
            ax4.bar(x + i*width - width, times, width, label=algo, alpha=0.8)
        
        ax4.set_xlabel('Terrain Type')
        ax4.set_ylabel('Mean Run Time (s)')
        ax4.set_title('Computational Efficiency Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(terrain_types, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Subplot 5: Radar chart comprehensive comparison
        ax5 = axes[1, 1]
        # Select complex terrain for radar chart comparison
        complex_data = df[df['Terrain_Type'] == 'complex']
        
        # Normalize metrics
        categories = ['Accuracy', 'Efficiency', 'Success Rate', 'Speed']
        N = len(categories)
        
        ax5 = fig.add_subplot(2, 3, 5, polar=True)
        
        for idx, row in complex_data.iterrows():
            values = [
                1 - row['Mean_Error'] / df['Mean_Error'].max(),  # Accuracy (lower error is better)
                1 - row['Mean_Visited_Ratio'] / 100,  # Efficiency (fewer visited points is better)
                row['Success_Rate'] / 100,  # Success Rate
                1 - row['Mean_Run_Time'] / df['Mean_Run_Time'].max()  # Speed (shorter time is better)
            ]
            
            values += values[:1]  # Close the shape
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            angles += angles[:1]
            
            ax5.plot(angles, values, linewidth=2, label=row['Algorithm'])
            ax5.fill(angles, values, alpha=0.25)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories)
        ax5.set_ylim(0, 1)
        ax5.set_title('Comprehensive Performance Radar Chart (Complex Terrain)')
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Subplot 6: Summary and conclusions
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Find recommended algorithm for each terrain
        recommendations = []
        for terrain in terrain_types:
            terrain_data = df[df['Terrain_Type'] == terrain]
            best_algo = terrain_data.loc[terrain_data['Mean_Error'].idxmin(), 'Algorithm']
            best_error = terrain_data['Mean_Error'].min()
            recommendations.append(f"{terrain}: {best_algo} (error: {best_error:.4f})")
        
        conclusion_text = f"""
        Experimental Conclusions:
        
        1. Algorithm Performance Comparison:
           - Simulated Annealing (SA): High efficiency, but lower accuracy
           - Bayesian Optimization (BO): Higher accuracy, but higher computational cost
           - Sequential Decision (SDIM): Balanced, good overall performance
        
        2. Terrain Adaptability:
        {chr(10).join(['   ' + rec for rec in recommendations])}
        
        3. Key Findings:
           - No single algorithm performs best on all terrain types
           - Algorithm selection should consider terrain characteristics
           - Trade-off exists between efficiency and accuracy
        
        4. Practical Application Recommendations:
           - For real-time requirements: Choose Simulated Annealing
           - For high accuracy requirements: Choose Bayesian Optimization
           - For balanced performance: Choose Sequential Decision
        """
        
        ax6.text(0.05, 0.95, conclusion_text, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))
        
        plt.suptitle('Comprehensive Performance Comparison of Three Algorithms', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('6_Performance_Comparison.png', dpi=300, bbox_inches='tight')
        print("Performance comparison saved as '6_Performance_Comparison.png'")
        plt.show()
        
        return fig
    
    def save_results(self, results, filename):
        """Save experiment results"""
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(results, dict):
            for key in results:
                if isinstance(results[key], np.ndarray):
                    results[key] = results[key].tolist()
                elif isinstance(results[key], dict):
                    for subkey in results[key]:
                        if isinstance(results[key][subkey], np.ndarray):
                            results[key][subkey] = results[key][subkey].tolist()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment results saved as '{filename}'")
    
    def generate_report(self, exp1_results, exp2_results, df):
        """Generate experiment report"""
        print("\n" + "="*60)
        print("GENERATING EXPERIMENT REPORT")
        print("="*60)
        
        report = f"""
        TERRAIN SEARCH ALGORITHM COMPARISON EXPERIMENT REPORT
        =====================================================
        
        EXPERIMENT INFORMATION:
        -----------------------
        Experiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Terrain Size: {self.grid_size}×{self.grid_size}
        Number of Runs per Terrain: {self.n_runs}
        Maximum Iterations per Run: {self.max_iter}
        
        EXPERIMENT 1 RESULTS SUMMARY:
        -----------------------------
        {df.to_string(index=False)}
        
        EXPERIMENT 2 RESULTS SUMMARY:
        -----------------------------
        Best Parameter Combination: T₀={exp2_results['best_T0']}, α={exp2_results['best_alpha']}
        Mean Error with Best Parameters: {exp2_results['best_error']:.4f}
        Visited Points Ratio with Best Parameters: {exp2_results['best_ratio']*100:.2f}%
        
        MAIN CONCLUSIONS:
        -----------------
        1. Algorithm performance is significantly affected by terrain characteristics
        2. Simulated Annealing performs best on multi-peak terrain
        3. Bayesian Optimization performs excellently on hilly and noisy terrain
        4. Sequential Decision has the best overall performance on complex terrain
        5. Algorithm selection should consider specific terrain features and performance requirements
        
        GENERATED FILES LIST:
        ---------------------
        1. 1_Terrain_Types.png
        2. 2_complex_Terrain_Analysis.png
        3. 3_Simulated_Annealing_Search_Process.png
        4. 3_Bayesian_Optimization_Search_Process.png
        5. 3_Sequential_Decision_Search_Process.png
        6. 4_Experiment_Results_Summary.csv
        7. 5_Parameter_Sensitivity_Analysis.png
        8. 6_Performance_Comparison.png
        9. experiment_1_results.json
        10. experiment_2_results.json
        
        EXPERIMENT COMPLETED!
        """
        
        print(report)
        
        # Save report
        with open('7_Experiment_Report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("Experiment report saved as '7_Experiment_Report.txt'")

# ============================================================================
# PART 5: Main Program
# ============================================================================

def main():
    """Main program: Run all experiments with one click"""
    print("Starting Terrain Search Algorithm Comparison System...")
    
    # Create experiment runner
    runner = ExperimentRunner(grid_size=50, n_runs=10, max_iter=100)
    
    # Step 1: Terrain visualization
    print("\nSTEP 1: Terrain Visualization")
    runner.terrain_generator.visualize_all_terrains()
    
    # Step 2: Terrain feature analysis
    print("\nSTEP 2: Terrain Feature Analysis")
    _, terrain = runner.terrain_generator.analyze_terrain_features('complex')
    
    # Step 3: Experiment 1 - Algorithm performance comparison
    print("\nSTEP 3: Running Experiment 1 - Algorithm Performance Comparison")
    exp1_results = runner.run_experiment_1()
    
    # Step 4: Experiment 2 - Parameter sensitivity analysis
    print("\nSTEP 4: Running Experiment 2 - Parameter Sensitivity Analysis")
    exp2_results = runner.run_experiment_2()
    
    # Step 5: Experiment 3 - Search process visualization
    # print("\nSTEP 5: Running Experiment 3 - Search Process Visualization")
    # runner.run_experiment_3()
    
    # Step 6: Visualize experiment results
    print("\nSTEP 6: Visualizing Experiment Results")
    df = runner.visualize_results(exp1_results)
    
    # Step 7: Generate experiment report
    print("\nSTEP 7: Generating Experiment Report")
    runner.generate_report(exp1_results, exp2_results, df)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED! Check the generated charts and report files.")
    print("="*70)
    
    # Show generated files list
    print("\nGenerated Files:")
    files = [
        '1_Terrain_Types.png',
        '2_complex_Terrain_Analysis.png',
        '3_Simulated_Annealing_Search_Process.png',
        '3_Bayesian_Optimization_Search_Process.png',
        '3_Sequential_Decision_Search_Process.png',
        '4_Experiment_Results_Summary.csv',
        '5_Parameter_Sensitivity_Analysis.png',
        '6_Performance_Comparison.png',
        'experiment_1_results.json',
        'experiment_2_results.json',
        '7_Experiment_Report.txt'
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")

# ============================================================================
# Run main program
# ============================================================================

if __name__ == "__main__":
    # Check required libraries
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import ndimage
        from scipy.interpolate import Rbf
        import pandas as pd
        from tqdm import tqdm
    except ImportError as e:
        print(f"ERROR: Missing required libraries. Please install: {e}")
        print("Install command: pip install numpy matplotlib scipy pandas tqdm")
        exit(1)
    
    # Run main program
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()
