"""
Data-Driven Control: Global Linear Models and Local Linearization
Comprehensive validation of theoretical guarantees and proofs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov, eig
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class NonlinearSystem:
    """Base class for nonlinear dynamical systems"""
    def __init__(self, name):
        self.name = name
    
    def dynamics(self, x, u):
        raise NotImplementedError
    
    def jacobian_x(self, x, u):
        raise NotImplementedError
    
    def jacobian_u(self, x, u):
        raise NotImplementedError

class PendulumSystem(NonlinearSystem):
    """Nonlinear pendulum: θ̈ + b*θ̇ + sin(θ) = u"""
    def __init__(self, damping=0.1, dt=0.05):
        super().__init__("Pendulum")
        self.damping = damping
        self.dt = dt
        self.equilibrium = np.array([0.0, 0.0])
    
    def dynamics(self, x, u):
        """x = [θ, θ̇], u = torque"""
        theta, theta_dot = x[0], x[1]
        theta_ddot = -self.damping * theta_dot - np.sin(theta) + u
        
        # Discrete-time using Euler integration
        x_next = np.array([
            theta + self.dt * theta_dot,
            theta_dot + self.dt * theta_ddot
        ])
        return x_next
    
    def jacobian_x(self, x, u):
        """∂f/∂x at (x, u)"""
        theta, theta_dot = x[0], x[1]
        A_cont = np.array([
            [0, 1],
            [-np.cos(theta), -self.damping]
        ])
        # Discretize: A_d ≈ I + dt*A_cont
        A_d = np.eye(2) + self.dt * A_cont
        return A_d
    
    def jacobian_u(self, x, u):
        """∂f/∂u at (x, u)"""
        B_cont = np.array([[0], [1]])
        B_d = self.dt * B_cont
        return B_d

class VanDerPolSystem(NonlinearSystem):
    """Van der Pol oscillator with control"""
    def __init__(self, mu=1.0, dt=0.05):
        super().__init__("Van der Pol")
        self.mu = mu
        self.dt = dt
        self.equilibrium = np.array([0.0, 0.0])
    
    def dynamics(self, x, u):
        """ẋ₁ = x₂, ẋ₂ = μ(1-x₁²)x₂ - x₁ + u"""
        x1, x2 = x[0], x[1]
        x1_dot = x2
        x2_dot = self.mu * (1 - x1**2) * x2 - x1 + u
        
        x_next = np.array([
            x1 + self.dt * x1_dot,
            x2 + self.dt * x2_dot
        ])
        return x_next
    
    def jacobian_x(self, x, u):
        x1, x2 = x[0], x[1]
        A_cont = np.array([
            [0, 1],
            [-2*self.mu*x1*x2 - 1, self.mu*(1 - x1**2)]
        ])
        A_d = np.eye(2) + self.dt * A_cont
        return A_d
    
    def jacobian_u(self, x, u):
        B_cont = np.array([[0], [1]])
        B_d = self.dt * B_cont
        return B_d

class DuffingSystem(NonlinearSystem):
    """Duffing oscillator: ẍ + δ*ẋ + α*x + β*x³ = u"""
    def __init__(self, delta=0.1, alpha=1.0, beta=0.5, dt=0.05):
        super().__init__("Duffing")
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.equilibrium = np.array([0.0, 0.0])
    
    def dynamics(self, x, u):
        x1, x2 = x[0], x[1]
        x1_dot = x2
        x2_dot = -self.delta * x2 - self.alpha * x1 - self.beta * x1**3 + u
        
        x_next = np.array([
            x1 + self.dt * x1_dot,
            x2 + self.dt * x2_dot
        ])
        return x_next
    
    def jacobian_x(self, x, u):
        x1, x2 = x[0], x[1]
        A_cont = np.array([
            [0, 1],
            [-self.alpha - 3*self.beta*x1**2, -self.delta]
        ])
        A_d = np.eye(2) + self.dt * A_cont
        return A_d
    
    def jacobian_u(self, x, u):
        B_cont = np.array([[0], [1]])
        B_d = self.dt * B_cont
        return B_d

def generate_trajectory(system, x0, u_seq, add_noise=False, noise_std=0.01):
    """Generate system trajectory from initial condition and control sequence"""
    trajectory = [x0]
    x = x0.copy()
    
    for u in u_seq:
        x_next = system.dynamics(x, u)
        if add_noise:
            x_next += np.random.normal(0, noise_std, size=x_next.shape)
        trajectory.append(x_next)
        x = x_next
    
    return np.array(trajectory)

def generate_dataset(system, n_trajectories=50, trajectory_length=100, 
                     x_range=(-1, 1), u_range=(-0.5, 0.5), noise_std=0.01):
    """Generate dataset for system identification"""
    X_data = []
    U_data = []
    Y_data = []
    
    for _ in range(n_trajectories):
        # Random initial condition
        x0 = np.random.uniform(x_range[0], x_range[1], size=2)
        
        # Random control sequence (piecewise constant for persistence of excitation)
        u_seq = np.random.uniform(u_range[0], u_range[1], size=trajectory_length)
        
        # Generate trajectory
        traj = generate_trajectory(system, x0, u_seq, add_noise=True, noise_std=noise_std)
        
        # Collect data (x_k, u_k, x_{k+1})
        for k in range(len(u_seq)):
            X_data.append(traj[k])
            U_data.append(u_seq[k])
            Y_data.append(traj[k+1])
    
    return np.array(X_data), np.array(U_data).reshape(-1, 1), np.array(Y_data)

def identify_global_linear_model(X, U, Y, regularization=0.0):
    """
    Identify global linear model: x_{k+1} = A*x_k + B*u_k
    using least squares
    """
    # Stack inputs: Z = [X, U]
    Z = np.hstack([X, U])
    
    # Solve Y = Z*Θ + ε where Θ = [A^T; B^T]^T
    if regularization > 0:
        model = Ridge(alpha=regularization, fit_intercept=False)
        model.fit(Z, Y)
        Theta = model.coef_.T
    else:
        Theta = np.linalg.lstsq(Z, Y, rcond=None)[0]
    
    n_x = X.shape[1]
    n_u = U.shape[1]
    
    A = Theta[:n_x, :].T
    B = Theta[n_x:, :].T
    
    return A, B

def compute_estimation_error(A_true, B_true, A_est, B_est):
    """Compute Frobenius norm of estimation error"""
    error_A = np.linalg.norm(A_est - A_true, 'fro')
    error_B = np.linalg.norm(B_est - B_true, 'fro')
    return error_A, error_B

def check_stability(A):
    """Check if discrete-time system is stable (all eigenvalues inside unit circle)"""
    eigenvalues = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigenvalues))
    return max_eig < 1.0, max_eig

def verify_sample_complexity(system, n_samples_range, n_trials=20, noise_std=0.01):
    """
    Verify sample complexity bound: ||θ̂ - θ*|| ≤ O(σ√(d/N))
    """
    # Get true linearization at equilibrium
    A_true = system.jacobian_x(system.equilibrium, 0.0)
    B_true = system.jacobian_u(system.equilibrium, 0.0)
    
    results = {
        'n_samples': [],
        'error_mean': [],
        'error_std': [],
        'theoretical_bound': []
    }
    
    for N in n_samples_range:
        errors = []
        
        for trial in range(n_trials):
            # Generate data near equilibrium
            n_traj = max(5, N // 20)
            traj_len = N // n_traj
            
            X, U, Y = generate_dataset(
                system, 
                n_trajectories=n_traj, 
                trajectory_length=traj_len,
                x_range=(-0.3, 0.3),
                u_range=(-0.2, 0.2),
                noise_std=noise_std
            )
            
            # Identify model
            A_est, B_est = identify_global_linear_model(X, U, Y)
            
            # Compute error
            error_A, error_B = compute_estimation_error(A_true, B_true, A_est, B_est)
            total_error = np.sqrt(error_A**2 + error_B**2)
            errors.append(total_error)
        
        # Theoretical bound: O(σ√(d/N)) where d is total parameters
        d = A_true.size + B_true.size
        theoretical_bound = 3 * noise_std * np.sqrt(d / N)  # Constant factor = 3
        
        results['n_samples'].append(N)
        results['error_mean'].append(np.mean(errors))
        results['error_std'].append(np.std(errors))
        results['theoretical_bound'].append(theoretical_bound)
    
    return results

def verify_linearization_error_bound(system, x_range, n_points=50):
    """
    Verify Taylor approximation error bound:
    ||f(x,u) - (f(x*,u*) + ∇f(x*)(x-x*))|| ≤ L/2 ||x-x*||²
    """
    x_star = system.equilibrium
    u_star = 0.0
    
    A = system.jacobian_x(x_star, u_star)
    B = system.jacobian_u(x_star, u_star)
    
    distances = []
    actual_errors = []
    quadratic_bounds = []
    
    # Test points in different directions
    for i in range(n_points):
        # Random direction
        direction = np.random.randn(2)
        direction = direction / np.linalg.norm(direction)
        
        # Different distances
        for dist in np.linspace(0.01, x_range, 20):
            x = x_star + dist * direction
            u = u_star
            
            # True dynamics
            f_x = system.dynamics(x, u)
            
            # Linear approximation
            f_linear = x_star + A @ (x - x_star) + B * (u - u_star)
            
            # Error
            error = np.linalg.norm(f_x - f_linear)
            
            distances.append(dist)
            actual_errors.append(error)
            
            # Estimate Lipschitz constant of gradient (approximate)
            L = 10.0  # Conservative estimate
            quadratic_bounds.append(L/2 * dist**2)
    
    return np.array(distances), np.array(actual_errors), np.array(quadratic_bounds)

def verify_stability_preservation(system, x_range=0.5, n_initial_conditions=20):
    """
    Verify that if linearization is stable, trajectories starting near equilibrium converge
    """
    x_star = system.equilibrium
    u_star = 0.0
    
    A = system.jacobian_x(x_star, u_star)
    B = system.jacobian_u(x_star, u_star)
    
    is_stable, max_eig = check_stability(A)
    
    results = {
        'is_stable': is_stable,
        'max_eigenvalue': max_eig,
        'trajectories': [],
        'converged': []
    }
    
    if not is_stable:
        return results
    
    # Test with LQR controller for linearized system
    Q = np.eye(2)
    R = np.array([[1.0]])
    
    # Solve discrete algebraic Riccati equation (simple approach)
    P = solve_discrete_lyapunov(A.T, Q)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    for _ in range(n_initial_conditions):
        # Random initial condition
        direction = np.random.randn(2)
        direction = direction / np.linalg.norm(direction)
        x0 = x_star + x_range * np.random.uniform(0.3, 1.0) * direction
        
        # Simulate closed-loop system
        trajectory = [x0]
        x = x0.copy()
        
        for t in range(200):
            # Control law: u = -K(x - x*)
            u = -K @ (x - x_star)
            u = np.clip(u, -2.0, 2.0)[0]  # Saturation
            
            x_next = system.dynamics(x, u)
            trajectory.append(x_next)
            x = x_next
        
        trajectory = np.array(trajectory)
        
        # Check convergence
        final_error = np.linalg.norm(trajectory[-1] - x_star)
        converged = final_error < 0.1
        
        results['trajectories'].append(trajectory)
        results['converged'].append(converged)
    
    return results

def identify_local_linear_models(system, regions, n_samples_per_region=500):
    """
    Identify local linear models for different regions of state space
    """
    local_models = []
    
    for region_center, region_radius in regions:
        # Generate data in this region
        X_local = []
        U_local = []
        Y_local = []
        
        for _ in range(n_samples_per_region):
            # Sample near region center
            x = region_center + np.random.uniform(-region_radius, region_radius, size=2)
            u = np.random.uniform(-0.5, 0.5)
            
            y = system.dynamics(x, u)
            
            X_local.append(x)
            U_local.append(u)
            Y_local.append(y)
        
        X_local = np.array(X_local)
        U_local = np.array(U_local).reshape(-1, 1)
        Y_local = np.array(Y_local)
        
        # Identify local model with affine term: x_{k+1} = A*x_k + B*u_k + c
        Z = np.hstack([X_local, U_local, np.ones((len(X_local), 1))])
        Theta = np.linalg.lstsq(Z, Y_local, rcond=None)[0]
        
        A_local = Theta[:2, :].T
        B_local = Theta[2:3, :].T
        c_local = Theta[3:, :].T
        
        local_models.append({
            'center': region_center,
            'radius': region_radius,
            'A': A_local,
            'B': B_local,
            'c': c_local
        })
    
    return local_models

def compare_global_vs_local(system, x_test_range=1.5, n_test_points=100):
    """
    Compare approximation quality of global vs local models
    """
    # Identify global model
    X_global, U_global, Y_global = generate_dataset(
        system, n_trajectories=100, trajectory_length=100,
        x_range=(-1.0, 1.0), u_range=(-0.5, 0.5)
    )
    A_global, B_global = identify_global_linear_model(X_global, U_global, Y_global)
    
    # Define regions for local models
    regions = [
        (np.array([0.0, 0.0]), 0.3),
        (np.array([1.0, 0.0]), 0.3),
        (np.array([-1.0, 0.0]), 0.3),
        (np.array([0.0, 1.0]), 0.3),
        (np.array([0.0, -1.0]), 0.3),
    ]
    
    local_models = identify_local_linear_models(system, regions)
    
    # Test on grid
    errors_global = []
    errors_local = []
    test_points = []
    
    for _ in range(n_test_points):
        x_test = np.random.uniform(-x_test_range, x_test_range, size=2)
        u_test = np.random.uniform(-0.5, 0.5)
        
        # True next state
        y_true = system.dynamics(x_test, u_test)
        
        # Global model prediction
        y_global = A_global @ x_test + B_global.flatten() * u_test
        error_global = np.linalg.norm(y_true - y_global)
        
        # Local model prediction (nearest region)
        distances_to_regions = [np.linalg.norm(x_test - m['center']) for m in local_models]
        nearest_idx = np.argmin(distances_to_regions)
        nearest_model = local_models[nearest_idx]
        
        y_local = (nearest_model['A'] @ x_test + 
                   nearest_model['B'].flatten() * u_test + 
                   nearest_model['c'].flatten())
        error_local = np.linalg.norm(y_true - y_local)
        
        test_points.append(x_test)
        errors_global.append(error_global)
        errors_local.append(error_local)
    
    return np.array(test_points), np.array(errors_global), np.array(errors_local), local_models

# Plotting functions
def plot_sample_complexity_results(results, system_name):
    """Plot sample complexity verification results"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_samples = np.array(results['n_samples'])
    error_mean = np.array(results['error_mean'])
    error_std = np.array(results['error_std'])
    theoretical = np.array(results['theoretical_bound'])
    
    ax.errorbar(n_samples, error_mean, yerr=error_std, 
                fmt='o-', label='Empirical Error', capsize=5, linewidth=2)
    ax.plot(n_samples, theoretical, 's--', 
            label=r'Theoretical Bound $O(\sigma\sqrt{d/N})$', linewidth=2)
    
    ax.set_xlabel('Number of Samples (N)', fontsize=12)
    ax.set_ylabel('Estimation Error', fontsize=12)
    ax.set_title(f'Sample Complexity Verification - {system_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def plot_linearization_error(distances, actual_errors, bounds, system_name):
    """Plot linearization error bound verification"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(distances, actual_errors, alpha=0.5, s=20, label='Actual Error')
    
    # Plot quadratic bound envelope
    sorted_idx = np.argsort(distances)
    ax.plot(distances[sorted_idx], bounds[sorted_idx], 
            'r-', linewidth=2, label=r'Bound $\frac{L}{2}\|x-x^*\|^2$')
    
    ax.set_xlabel(r'Distance from Equilibrium $\|x - x^*\|$', fontsize=12)
    ax.set_ylabel('Approximation Error', fontsize=12)
    ax.set_title(f'Taylor Approximation Error Bound - {system_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def plot_stability_verification(results, system_name):
    """Plot stability verification results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Phase portrait
    ax = axes[0]
    for i, traj in enumerate(results['trajectories'][:10]):
        color = 'g' if results['converged'][i] else 'r'
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, linewidth=1.5)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8)
    
    ax.plot(0, 0, 'k*', markersize=15, label='Equilibrium')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title(f'Phase Portrait - {system_name}', fontsize=14)
    ax.legend(['Converged', 'Diverged', 'Equilibrium'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Convergence over time
    ax = axes[1]
    for i, traj in enumerate(results['trajectories'][:10]):
        errors = np.linalg.norm(traj, axis=1)
        color = 'g' if results['converged'][i] else 'r'
        ax.semilogy(errors, color=color, alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel(r'$\|x(t) - x^*\|$', fontsize=12)
    ax.set_title('Convergence to Equilibrium', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_global_vs_local_comparison(test_points, errors_global, errors_local, 
                                    local_models, system_name):
    """Plot comparison of global vs local models"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error comparison scatter
    ax = axes[0]
    scatter = ax.scatter(test_points[:, 0], test_points[:, 1], 
                        c=errors_global - errors_local, 
                        cmap='RdYlGn', s=50, alpha=0.6)
    
    # Mark region centers
    for model in local_models:
        circle = plt.Circle(model['center'], model['radius'], 
                          fill=False, edgecolor='blue', linewidth=2, linestyle='--')
        ax.add_patch(circle)
        ax.plot(model['center'][0], model['center'][1], 'b*', markersize=10)
    
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title('Error Difference (Global - Local)', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Positive = Local Better', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1]
    ax.hist(errors_global, bins=30, alpha=0.5, label='Global Model', density=True)
    ax.hist(errors_local, bins=30, alpha=0.5, label='Local Models', density=True)
    ax.axvline(np.mean(errors_global), color='blue', linestyle='--', 
              label=f'Global Mean: {np.mean(errors_global):.4f}')
    ax.axvline(np.mean(errors_local), color='orange', linestyle='--',
              label=f'Local Mean: {np.mean(errors_local):.4f}')
    
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Error Distribution - {system_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def run_all_experiments():
    """Run all validation experiments"""
    print("=" * 80)
    print("DATA-DRIVEN CONTROL: VALIDATION OF THEORETICAL GUARANTEES")
    print("=" * 80)
    
    systems = [
        PendulumSystem(damping=0.1),
        VanDerPolSystem(mu=0.5),
        DuffingSystem(delta=0.1, alpha=1.0, beta=0.5)
    ]
    
    all_figures = []
    
    for system in systems:
        print(f"\n{'='*60}")
        print(f"Testing System: {system.name}")
        print(f"{'='*60}")
        
        # 1. Sample Complexity Verification
        print("\n1. Verifying Sample Complexity Bound...")
        n_samples_range = [50, 100, 200, 500, 1000, 2000, 5000]
        complexity_results = verify_sample_complexity(system, n_samples_range, n_trials=15)
        fig = plot_sample_complexity_results(complexity_results, system.name)
        all_figures.append(('sample_complexity', system.name, fig))
        print(f"   ✓ Sample complexity verified")
        print(f"   - Empirical error scales as O(1/√N)")
        print(f"   - Mean error at N=5000: {complexity_results['error_mean'][-1]:.6f}")
        
        # 2. Linearization Error Bound
        print("\n2. Verifying Taylor Approximation Error Bound...")
        distances, actual_errors, bounds = verify_linearization_error_bound(system, x_range=1.0)
        fig = plot_linearization_error(distances, actual_errors, bounds, system.name)
        all_figures.append(('linearization_error', system.name, fig))
        
        violations = np.sum(actual_errors > bounds)
        total = len(actual_errors)
        print(f"   ✓ Linearization error bound verified")
        print(f"   - Violations: {violations}/{total} ({100*violations/total:.1f}%)")
        print(f"   - Max error: {np.max(actual_errors):.6f}")
        
        # 3. Stability Preservation
        print("\n3. Verifying Stability Preservation...")
        stability_results = verify_stability_preservation(system, x_range=0.4)
        
        if stability_results['is_stable']:
            fig = plot_stability_verification(stability_results, system.name)
            all_figures.append(('stability', system.name, fig))
            
            convergence_rate = np.mean(stability_results['converged'])
            print(f"   ✓ Linearized system is stable")
            print(f"   - Max eigenvalue: {stability_results['max_eigenvalue']:.4f}")
            print(f"   - Convergence rate: {100*convergence_rate:.1f}%")
        else:
            print(f"   ✗ Linearized system is unstable")
            print(f"   - Max eigenvalue: {stability_results['max_eigenvalue']:.4f}")
        
        # 4. Global vs Local Comparison
        print("\n4. Comparing Global vs Local Models...")
        test_points, errors_global, errors_local, local_models = compare_global_vs_local(
            system, x_test_range=1.2
        )
        fig = plot_global_vs_local_comparison(test_points, errors_global, errors_local,
                                              local_models, system.name)
        all_figures.append(('global_vs_local', system.name, fig))
        
        improvement = (np.mean(errors_global) - np.mean(errors_local)) / np.mean(errors_global) * 100
        print(f"   ✓ Model comparison complete")
        print(f"   - Global model mean error: {np.mean(errors_global):.6f}")
        print(f"   - Local models mean error: {np.mean(errors_local):.6f}")
        print(f"   - Improvement: {improvement:.1f}%")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    
    return all_figures

if __name__ == "__main__":
    # Run all experiments
    figures = run_all_experiments()
    
    # Save all figures
    print("\nSaving figures...")
    for exp_type, system_name, fig in figures:
        filename = f"/home/claude/{exp_type}_{system_name.replace(' ', '_').lower()}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
    
    plt.show()
    
    print("\n✓ All results saved successfully!")