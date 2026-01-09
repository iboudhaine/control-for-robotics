"""
Interactive Visualization of Abstraction and Post() for Unicycle Dynamics.

This script provides an interactive workflow:
1. Select eta (discretization parameter) using a slider
2. Perform abstraction (discretization) on button click
3. Select the starting angle (theta) using a slider
4. Select from all available controls using prev/next buttons
5. Click on any cell in the grid to show its successors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# Add symbolic_control to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symbolic_control.dynamics import UnicycleDynamics
from symbolic_control.abstraction import Abstraction


class InteractivePostVisualizer:
    """Interactive visualization for discretization and successor cells."""
    
    def __init__(self):
        # Default parameters
        self.tau = 0.5
        self.w_bound = 0.05
        self.eta = 0.5  # Default discretization parameter
        
        # State bounds for 2D projection (x, y)
        # Note: For unicycle, we have 3D state (x, y, theta) but visualize x-y
        self.state_bounds_2d = np.array([
            [0.0, 4.0],   # x bounds
            [0.0, 4.0],   # y bounds
        ])
        
        # Full 3D state bounds for unicycle
        self.state_bounds_3d = np.array([
            [0.0, 4.0],           # x bounds
            [0.0, 4.0],           # y bounds
            [-np.pi, np.pi],      # theta bounds
        ])
        
        # Dynamics and abstraction (will be created on discretize)
        self.dynamics = UnicycleDynamics(tau=self.tau, w_bound=self.w_bound, v_values=np.linspace(0.25, 1, num=4), omega_values=np.linspace(-np.pi, np.pi, num=10))
        self.abstraction = None
        
        # Selected cell
        self.selected_cell = None
        
        # Control selection - use ALL controls from dynamics
        self.all_controls = self.dynamics.control_set
        self.selected_control_idx = 0
        
        # Theta (angle) selection
        self.theta_idx = 0  # Will be updated after discretization
        self.num_theta_cells = 1  # Will be updated after discretization
        
        # Setup figure
        self.setup_figure()
        
    def setup_figure(self):
        """Create the interactive figure with widgets."""
        self.fig = plt.figure(figsize=(16, 11))
        self.fig.suptitle('Interactive Abstraction & Post() Visualization', fontsize=14, fontweight='bold')
        
        # Main grid plot
        self.ax_grid = self.fig.add_axes([0.08, 0.38, 0.45, 0.52])
        self.ax_grid.set_title('Discretized State Space (X-Y Projection)\nClick on a cell to show successors')
        self.ax_grid.set_xlabel('X position')
        self.ax_grid.set_ylabel('Y position')
        
        # Successor details plot
        self.ax_succ = self.fig.add_axes([0.58, 0.38, 0.38, 0.52])
        self.ax_succ.set_title('Successor Details')
        self.ax_succ.set_xlabel('X position')
        self.ax_succ.set_ylabel('Y position')
        
        # Info text area
        self.ax_info = self.fig.add_axes([0.08, 0.02, 0.88, 0.08])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.5, 0.5, 'Select eta parameter and click "Discretize" to begin.',
                                            ha='center', va='center', fontsize=11,
                                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # === Row 1: Eta slider and Discretize button ===
        self.ax_eta = self.fig.add_axes([0.12, 0.28, 0.30, 0.03])
        self.slider_eta = Slider(
            ax=self.ax_eta,
            label='η (eta)',
            valmin=0.1,
            valmax=1.0,
            valinit=self.eta,
            valstep=0.1,
            color='steelblue'
        )
        self.slider_eta.on_changed(self.on_eta_change)
        
        # Discretize button
        self.ax_btn_discretize = self.fig.add_axes([0.45, 0.27, 0.10, 0.04])
        self.btn_discretize = Button(self.ax_btn_discretize, 'Discretize', color='lightgreen', hovercolor='lime')
        self.btn_discretize.on_clicked(self.on_discretize)
        
        # === Row 2: Theta slider ===
        self.ax_theta = self.fig.add_axes([0.12, 0.21, 0.43, 0.03])
        self.slider_theta = Slider(
            ax=self.ax_theta,
            label='θ index',
            valmin=0,
            valmax=10,  # Will be updated after discretization
            valinit=0,
            valstep=1,
            color='coral'
        )
        self.slider_theta.on_changed(self.on_theta_change)
        
        # Theta value display
        self.ax_theta_val = self.fig.add_axes([0.56, 0.21, 0.10, 0.03])
        self.ax_theta_val.axis('off')
        self.theta_val_text = self.ax_theta_val.text(0.5, 0.5, 'θ = 0.00°', 
                                                      ha='center', va='center', fontsize=10,
                                                      bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        # === Row 3: Control selection ===
        # Previous control button
        self.ax_btn_prev = self.fig.add_axes([0.12, 0.13, 0.06, 0.04])
        self.btn_prev = Button(self.ax_btn_prev, '◀ Prev', color='lightblue', hovercolor='deepskyblue')
        self.btn_prev.on_clicked(self.on_prev_control)
        
        # Control display
        self.ax_control_display = self.fig.add_axes([0.19, 0.13, 0.30, 0.04])
        self.ax_control_display.axis('off')
        self.control_text = self.ax_control_display.text(
            0.5, 0.5, f'Control [1/{len(self.all_controls)}]: v=0.25, ω=-1.00',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='steelblue', linewidth=2)
        )
        
        # Next control button
        self.ax_btn_next = self.fig.add_axes([0.50, 0.13, 0.06, 0.04])
        self.btn_next = Button(self.ax_btn_next, 'Next ▶', color='lightblue', hovercolor='deepskyblue')
        self.btn_next.on_clicked(self.on_next_control)
        
        # Control info (show total)
        self.ax_control_info = self.fig.add_axes([0.58, 0.13, 0.35, 0.04])
        self.ax_control_info.axis('off')
        self.control_info_text = self.ax_control_info.text(
            0.0, 0.5, f'Total controls: {len(self.all_controls)} (v ∈ {{0.25, 0.5, 0.75, 1.0}}, ω ∈ {{-1, -0.5, 0, 0.5, 1}})',
            ha='left', va='center', fontsize=9, style='italic'
        )
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Draw initial empty grid
        self.draw_empty_grid()
        self.update_control_display()
        
    def draw_empty_grid(self):
        """Draw an empty grid outline."""
        self.ax_grid.clear()
        self.ax_grid.set_xlim(self.state_bounds_2d[0])
        self.ax_grid.set_ylim(self.state_bounds_2d[1])
        self.ax_grid.set_aspect('equal')
        self.ax_grid.grid(True, alpha=0.3)
        self.ax_grid.set_title('Discretized State Space (X-Y Projection)\nClick on a cell to show successors')
        self.ax_grid.set_xlabel('X position')
        self.ax_grid.set_ylabel('Y position')
        
        # Draw boundary
        rect = Rectangle((self.state_bounds_2d[0, 0], self.state_bounds_2d[1, 0]),
                          self.state_bounds_2d[0, 1] - self.state_bounds_2d[0, 0],
                          self.state_bounds_2d[1, 1] - self.state_bounds_2d[1, 0],
                          fill=False, edgecolor='black', linewidth=2)
        self.ax_grid.add_patch(rect)
        
    def on_eta_change(self, val):
        """Handle eta slider change."""
        self.eta = val
        self.update_info(f'η = {self.eta:.2f} selected. Click "Discretize" to apply.')
        
    def on_theta_change(self, val):
        """Handle theta slider change."""
        self.theta_idx = int(val)
        self.update_theta_display()
        
        # Redraw successors if a cell is selected
        if self.selected_cell is not None and self.abstraction is not None:
            # Update selected cell with new theta
            self.selected_cell['theta_idx'] = self.theta_idx
            multi_idx = (self.selected_cell['x_idx'], self.selected_cell['y_idx'], self.theta_idx)
            if all(0 <= multi_idx[d] < self.abstraction.grid_shape[d] for d in range(3)):
                self.selected_cell['cell_idx'] = np.ravel_multi_index(multi_idx, self.abstraction.grid_shape)
                self.show_successors()
    
    def update_theta_display(self):
        """Update the theta value display."""
        if self.abstraction is not None:
            theta_lo = self.state_bounds_3d[2, 0] + self.theta_idx * self.eta
            theta_hi = theta_lo + self.eta
            theta_center = (theta_lo + theta_hi) / 2
            self.theta_val_text.set_text(f'θ ∈ [{np.degrees(theta_lo):.0f}°, {np.degrees(theta_hi):.0f}°]')
        else:
            self.theta_val_text.set_text('θ = --')
        self.fig.canvas.draw_idle()
        
    def update_control_display(self):
        """Update the control display text."""
        u = self.all_controls[self.selected_control_idx]
        self.control_text.set_text(
            f'Control [{self.selected_control_idx + 1}/{len(self.all_controls)}]: v={u[0]:.2f}, ω={u[1]:.2f}'
        )
        self.fig.canvas.draw_idle()
        
    def on_prev_control(self, event):
        """Go to previous control."""
        self.selected_control_idx = (self.selected_control_idx - 1) % len(self.all_controls)
        self.update_control_display()
        if self.selected_cell is not None:
            self.show_successors()
            
    def on_next_control(self, event):
        """Go to next control."""
        self.selected_control_idx = (self.selected_control_idx + 1) % len(self.all_controls)
        self.update_control_display()
        if self.selected_cell is not None:
            self.show_successors()
        
    def on_discretize(self, event):
        """Perform discretization with current eta."""
        self.update_info(f'Discretizing with η = {self.eta:.2f}...')
        self.fig.canvas.draw_idle()
        
        # Create abstraction (3D for unicycle)
        self.abstraction = Abstraction(
            dynamics=self.dynamics,
            state_bounds=self.state_bounds_3d,
            eta=self.eta
        )
        
        # Update theta slider range
        self.num_theta_cells = self.abstraction.grid_shape[2]
        self.slider_theta.valmax = self.num_theta_cells - 1
        self.slider_theta.ax.set_xlim(0, self.num_theta_cells - 1)
        
        # Reset theta index to middle
        self.theta_idx = self.num_theta_cells // 2
        self.slider_theta.set_val(self.theta_idx)
        self.update_theta_display()
        
        # Draw the grid
        self.draw_grid()
        
        # Update info
        grid_info = f'Grid: {self.abstraction.grid_shape[0]}×{self.abstraction.grid_shape[1]}×{self.abstraction.grid_shape[2]} = {self.abstraction.num_cells} cells'
        self.update_info(f'Discretization complete! η = {self.eta:.2f}\n{grid_info}\nClick on any cell to see its successors.')
        
        # Clear successor view
        self.selected_cell = None
        self.ax_succ.clear()
        self.ax_succ.set_title('Successor Details\n(Select a cell)')
        
        self.fig.canvas.draw_idle()
        
    def draw_grid(self):
        """Draw the discretized grid."""
        self.ax_grid.clear()
        
        x_bounds = self.state_bounds_2d[0]
        y_bounds = self.state_bounds_2d[1]
        
        # Draw grid lines
        x_ticks = np.arange(x_bounds[0], x_bounds[1] + self.eta, self.eta)
        y_ticks = np.arange(y_bounds[0], y_bounds[1] + self.eta, self.eta)
        
        for x in x_ticks:
            self.ax_grid.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
        for y in y_ticks:
            self.ax_grid.axhline(y, color='gray', linewidth=0.5, alpha=0.5)
        
        # Draw cells as patches with light fill
        patches = []
        for i, x in enumerate(x_ticks[:-1]):
            for j, y in enumerate(y_ticks[:-1]):
                rect = Rectangle((x, y), self.eta, self.eta)
                patches.append(rect)
        
        pc = PatchCollection(patches, facecolor='lightblue', edgecolor='steelblue', 
                             alpha=0.3, linewidth=0.5)
        self.ax_grid.add_collection(pc)
        
        self.ax_grid.set_xlim(x_bounds[0] - 0.1, x_bounds[1] + 0.1)
        self.ax_grid.set_ylim(y_bounds[0] - 0.1, y_bounds[1] + 0.1)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_title(f'Discretized Grid (η = {self.eta:.2f})\nClick on a cell to show successors')
        self.ax_grid.set_xlabel('X position')
        self.ax_grid.set_ylabel('Y position')
            
    def on_click(self, event):
        """Handle mouse click on the grid."""
        if event.inaxes != self.ax_grid:
            return
        if self.abstraction is None:
            self.update_info('Please discretize first!')
            return
            
        x_click, y_click = event.xdata, event.ydata
        
        # Check if within bounds
        if (x_click < self.state_bounds_2d[0, 0] or x_click >= self.state_bounds_2d[0, 1] or
            y_click < self.state_bounds_2d[1, 0] or y_click >= self.state_bounds_2d[1, 1]):
            return
            
        # Convert to cell indices (2D)
        x_idx = int((x_click - self.state_bounds_2d[0, 0]) / self.eta)
        y_idx = int((y_click - self.state_bounds_2d[1, 0]) / self.eta)
        
        # Store selected cell info
        self.selected_cell = {
            'x_idx': x_idx,
            'y_idx': y_idx,
            'theta_idx': self.theta_idx,
            'x_lo': self.state_bounds_2d[0, 0] + x_idx * self.eta,
            'y_lo': self.state_bounds_2d[1, 0] + y_idx * self.eta,
        }
        
        # Compute 3D cell index
        multi_idx = (x_idx, y_idx, self.theta_idx)
        if all(0 <= multi_idx[d] < self.abstraction.grid_shape[d] for d in range(3)):
            self.selected_cell['cell_idx'] = np.ravel_multi_index(multi_idx, self.abstraction.grid_shape)
        else:
            self.selected_cell['cell_idx'] = None
            
        # Show successors
        self.show_successors()
        
    def show_successors(self):
        """Show successors for the selected cell and control."""
        if self.selected_cell is None or self.selected_cell['cell_idx'] is None:
            return
            
        cell = self.selected_cell
        u = self.all_controls[self.selected_control_idx]
        
        # Get cell bounds from abstraction
        cell_idx = cell['cell_idx']
        x_lo, x_hi = self.abstraction.cell_to_bounds(cell_idx)
        
        # Compute over-approximated successor
        succ_lo, succ_hi = self.dynamics.post(x_lo, x_hi, u)
        
        # Find successor cells
        succ_cells = self.abstraction.bounds_to_cells(succ_lo, succ_hi)
        
        # Redraw grid with highlighting
        self.draw_grid()
        
        # Highlight selected cell (blue)
        rect_selected = Rectangle((cell['x_lo'], cell['y_lo']), self.eta, self.eta,
                                    facecolor='blue', edgecolor='darkblue', alpha=0.6, linewidth=2)
        self.ax_grid.add_patch(rect_selected)
        
        # Highlight successor cells (orange/red)
        for sc_idx in succ_cells:
            sc_lo, sc_hi = self.abstraction.cell_to_bounds(sc_idx)
            # Only draw x-y projection
            rect_succ = Rectangle((sc_lo[0], sc_lo[1]), self.eta, self.eta,
                                    facecolor='orange', edgecolor='red', alpha=0.4, linewidth=1)
            self.ax_grid.add_patch(rect_succ)
        
        # Draw arrow from selected cell center to successor region center
        cell_center = (cell['x_lo'] + self.eta/2, cell['y_lo'] + self.eta/2)
        succ_center = ((succ_lo[0] + succ_hi[0])/2, (succ_lo[1] + succ_hi[1])/2)
        self.ax_grid.annotate('', xy=succ_center, xytext=cell_center,
                              arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Update successor details plot
        self.ax_succ.clear()
        
        # Draw selected cell
        margin = 0.5
        all_x = [cell['x_lo'], cell['x_lo'] + self.eta, succ_lo[0], succ_hi[0]]
        all_y = [cell['y_lo'], cell['y_lo'] + self.eta, succ_lo[1], succ_hi[1]]
        
        rect_orig = Rectangle((x_lo[0], x_lo[1]), x_hi[0] - x_lo[0], x_hi[1] - x_lo[1],
                                facecolor='blue', edgecolor='darkblue', alpha=0.5, linewidth=2,
                                label='Original Cell')
        self.ax_succ.add_patch(rect_orig)
        
        # Draw over-approximation bounding box
        rect_succ_box = Rectangle((succ_lo[0], succ_lo[1]), succ_hi[0] - succ_lo[0], succ_hi[1] - succ_lo[1],
                                    facecolor='orange', edgecolor='red', alpha=0.3, linewidth=2,
                                    label='Over-approx Bound')
        self.ax_succ.add_patch(rect_succ_box)
        
        # Sample trajectories
        n_samples = 100
        samples = []
        for _ in range(n_samples):
            x_sample = np.random.uniform(x_lo, x_hi)
            w_sample = np.random.uniform(-self.w_bound, self.w_bound, size=3)
            x_next = self.dynamics.step(x_sample, u, w_sample)
            samples.append(x_next)
        samples = np.array(samples)
        
        self.ax_succ.scatter(samples[:, 0], samples[:, 1], c='green', s=10, alpha=0.5,
                             label='Sample Successors')
        
        self.ax_succ.set_xlim(min(all_x) - margin, max(all_x) + margin)
        self.ax_succ.set_ylim(min(all_y) - margin, max(all_y) + margin)
        self.ax_succ.set_aspect('equal')
        self.ax_succ.legend(loc='upper left', fontsize=8)
        self.ax_succ.grid(True, alpha=0.3)
        self.ax_succ.set_title(f'Successor Details\nControl: v={u[0]:.2f}, ω={u[1]:.2f}')
        self.ax_succ.set_xlabel('X position')
        self.ax_succ.set_ylabel('Y position')
        
        # Update info
        theta_lo = x_lo[2]
        theta_hi = x_hi[2]
        theta_center = (theta_lo + theta_hi) / 2
        info = (f'Selected Cell: ({cell["x_idx"]}, {cell["y_idx"]}, {cell["theta_idx"]}) '
                f'[θ ∈ ({np.degrees(theta_lo):.0f}°, {np.degrees(theta_hi):.0f}°)]\n'
                f'Control: v={u[0]:.2f}, ω={u[1]:.2f}  |  '
                f'Successor cells: {len(succ_cells)}')
        self.update_info(info)
        
        self.fig.canvas.draw_idle()
        
    def update_info(self, text):
        """Update the info text box."""
        self.info_text.set_text(text)
        self.fig.canvas.draw_idle()
        
    def run(self):
        """Show the interactive visualization."""
        plt.show()


def visualize_unicycle_post():
    """
    Original non-interactive visualization (kept for backwards compatibility).
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    def create_box_vertices(lo, hi):
        """Create vertices of a 3D box given lower and upper corners."""
        x_lo, y_lo, theta_lo = lo
        x_hi, y_hi, theta_hi = hi
        
        vertices = np.array([
            [x_lo, y_lo, theta_lo],
            [x_hi, y_lo, theta_lo],
            [x_hi, y_hi, theta_lo],
            [x_lo, y_hi, theta_lo],
            [x_lo, y_lo, theta_hi],
            [x_hi, y_lo, theta_hi],
            [x_hi, y_hi, theta_hi],
            [x_lo, y_hi, theta_hi],
        ])
        return vertices

    def create_box_faces(vertices):
        """Create faces of a 3D box from its vertices."""
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
        ]
        return faces
    
    tau = 0.5
    w_bound = 0.05
    dynamics = UnicycleDynamics(tau=tau, w_bound=w_bound)
    
    cell_width = 0.3
    angle_width = 0.2
    
    x_lo = np.array([1.0, 1.0, np.pi/4 - angle_width/2])
    x_hi = np.array([1.0 + cell_width, 1.0 + cell_width, np.pi/4 + angle_width/2])
    
    u = np.array([0.5, 0.25])
    
    succ_lo, succ_hi = dynamics.post(x_lo, x_hi, u)
    
    print("=" * 60)
    print("UNICYCLE DYNAMICS POST() VISUALIZATION")
    print("=" * 60)
    print(f"\nDynamics Parameters:")
    print(f"  τ (sampling period): {tau}")
    print(f"  w_bound (disturbance): {w_bound}")
    print(f"\nOriginal Cell:")
    print(f"  Lower corner: x={x_lo[0]:.3f}, y={x_lo[1]:.3f}, θ={x_lo[2]:.3f} rad ({np.degrees(x_lo[2]):.1f}°)")
    print(f"  Upper corner: x={x_hi[0]:.3f}, y={x_hi[1]:.3f}, θ={x_hi[2]:.3f} rad ({np.degrees(x_hi[2]):.1f}°)")
    print(f"\nControl Input:")
    print(f"  v (linear velocity): {u[0]}")
    print(f"  ω (angular velocity): {u[1]}")
    print(f"\nOver-approximated Successor Cell:")
    print(f"  Lower corner: x={succ_lo[0]:.3f}, y={succ_lo[1]:.3f}, θ={succ_lo[2]:.3f} rad ({np.degrees(succ_lo[2]):.1f}°)")
    print(f"  Upper corner: x={succ_hi[0]:.3f}, y={succ_hi[1]:.3f}, θ={succ_hi[2]:.3f} rad ({np.degrees(succ_hi[2]):.1f}°)")
    
    n_samples = 200
    sample_successors = []
    
    for _ in range(n_samples):
        x_sample = np.random.uniform(x_lo, x_hi)
        w_sample = np.random.uniform(-w_bound, w_bound, size=3)
        x_next = dynamics.step(x_sample, u, w_sample)
        sample_successors.append(x_next)
    
    sample_successors = np.array(sample_successors)
    
    within_bounds = np.all((sample_successors >= succ_lo) & (sample_successors <= succ_hi))
    print(f"\nVerification:")
    print(f"  All {n_samples} samples within over-approximation: {within_bounds}")
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    vertices_orig = create_box_vertices(x_lo, x_hi)
    faces_orig = create_box_faces(vertices_orig)
    poly_orig = Poly3DCollection(faces_orig, alpha=0.3, facecolor='blue', edgecolor='blue', linewidth=1)
    ax1.add_collection3d(poly_orig)
    
    vertices_succ = create_box_vertices(succ_lo, succ_hi)
    faces_succ = create_box_faces(vertices_succ)
    poly_succ = Poly3DCollection(faces_succ, alpha=0.2, facecolor='orange', edgecolor='red', linewidth=1)
    ax1.add_collection3d(poly_succ)
    
    ax1.scatter(sample_successors[:, 0], sample_successors[:, 1], sample_successors[:, 2],
                c='green', s=5, alpha=0.5, label='Sample successors')
    
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_zlabel('θ (heading)')
    ax1.set_title('3D View: Cell and Successor')
    
    all_points = np.vstack([vertices_orig, vertices_succ])
    margin = 0.2
    ax1.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax1.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    ax1.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
    
    ax2 = fig.add_subplot(122)
    
    rect_orig = plt.Rectangle((x_lo[0], x_lo[1]), x_hi[0] - x_lo[0], x_hi[1] - x_lo[1],
                                fill=True, alpha=0.4, facecolor='blue', edgecolor='blue', linewidth=2,
                                label='Original Cell')
    ax2.add_patch(rect_orig)
    
    rect_succ = plt.Rectangle((succ_lo[0], succ_lo[1]), succ_hi[0] - succ_lo[0], succ_hi[1] - succ_lo[1],
                                fill=True, alpha=0.3, facecolor='orange', edgecolor='red', linewidth=2,
                                label='Successor Cell (Over-approx)')
    ax2.add_patch(rect_succ)
    
    ax2.scatter(sample_successors[:, 0], sample_successors[:, 1], c='green', s=10, alpha=0.5,
                label='Sample Successors')
    
    x_center = (x_lo + x_hi) / 2
    ax2.annotate('', xy=(succ_lo[0] + (succ_hi[0]-succ_lo[0])/2, succ_lo[1] + (succ_hi[1]-succ_lo[1])/2),
                 xytext=(x_center[0], x_center[1]),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    ax2.set_title(f'X-Y Projection\nControl: v={u[0]}, ω={u[1]}')
    ax2.legend(loc='upper left')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    all_x = [x_lo[0], x_hi[0], succ_lo[0], succ_hi[0]]
    all_y = [x_lo[1], x_hi[1], succ_lo[1], succ_hi[1]]
    ax2.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax2.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    plt.tight_layout()
    plt.savefig('unicycle_post_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: unicycle_post_visualization.png")
    plt.show()
    
    return dynamics, x_lo, x_hi, u, succ_lo, succ_hi


if __name__ == "__main__":
    print("=" * 60)
    print("INTERACTIVE ABSTRACTION & POST() VISUALIZATION")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Use the 'η (eta)' slider to select discretization parameter")
    print("2. Click 'Discretize' to create the grid")
    print("3. Use the 'θ index' slider to select the starting angle")
    print("4. Use '◀ Prev' and 'Next ▶' buttons to cycle through ALL controls")
    print("5. Click on any cell to see its successors")
    print("=" * 60)
    
    visualizer = InteractivePostVisualizer()
    visualizer.run()
