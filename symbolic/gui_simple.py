"""
Simple Synthesis GUI - Interactive interface for safety + reachability synthesis.

This GUI provides a simple synthesis pipeline (no automata):
1. Select dynamics model (Integrator or Unicycle)
2. Configure grid parameters (bounds, per-dimension eta)
3. Draw obstacles and select target region interactively
4. Run synthesis and visualize results
5. Simulate controller with interactive start position
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
import threading
import time
from typing import Dict, List, Set, Optional


class SimpleSynthesisGUI:
    """GUI for simple synthesis (safety + reachability)."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Synthesis - Safety + Reachability")
        self.root.geometry("1400x900")
        
        # State variables
        self.abstraction = None
        self.synth = None
        self.dynamics = None
        self.trajectory = None
        self.obstacles: List[List[float]] = []  # List of [x_min, x_max, y_min, y_max]
        self.target: Optional[List[float]] = None  # [x_min, x_max, y_min, y_max]
        self.synthesis_start_time = None
        
        # Drawing state
        self.drawing = False
        self.draw_start_point = None
        self.temp_rect = None
        self.draw_mode = "obstacle"  # "obstacle" or "target"
        
        # Create main layout
        self._create_layout()
        self._create_parameter_panel()
        self._create_output_panel()
        self._create_visualization_panel()
        
    def _create_layout(self):
        """Create the main layout."""
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for parameters
        self.left_frame = ttk.Frame(self.main_paned, width=400)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Right panel for visualization
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=3)
        
    def _create_parameter_panel(self):
        """Create the parameter configuration panel."""
        # Create scrollable frame
        canvas = tk.Canvas(self.left_frame)
        scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=canvas.yview)
        self.param_frame = ttk.Frame(canvas)
        
        self.param_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.param_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        row = 0
        
        # === Model Selection ===
        model_frame = ttk.LabelFrame(self.param_frame, text="Dynamics Model", padding=10)
        model_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        self.model_var = tk.StringVar(value="integrator")
        ttk.Radiobutton(model_frame, text="2D Integrator", variable=self.model_var, 
                       value="integrator", command=self._on_model_change).pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="3D Unicycle", variable=self.model_var, 
                       value="unicycle", command=self._on_model_change).pack(anchor=tk.W)
        
        # === Dynamics Parameters ===
        dyn_frame = ttk.LabelFrame(self.param_frame, text="Dynamics Parameters", padding=10)
        dyn_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(dyn_frame, text="Sampling Period (τ):").grid(row=0, column=0, sticky=tk.W)
        self.tau_var = tk.StringVar(value="0.5")
        ttk.Entry(dyn_frame, textvariable=self.tau_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(dyn_frame, text="Disturbance Bound:").grid(row=1, column=0, sticky=tk.W)
        self.w_bound_var = tk.StringVar(value="0.1")
        ttk.Entry(dyn_frame, textvariable=self.w_bound_var, width=10).grid(row=1, column=1, padx=5)
        
        # === State Bounds ===
        bounds_frame = ttk.LabelFrame(self.param_frame, text="State Bounds", padding=10)
        bounds_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(bounds_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x_min_var = tk.StringVar(value="-5")
        self.x_max_var = tk.StringVar(value="5")
        ttk.Entry(bounds_frame, textvariable=self.x_min_var, width=8).grid(row=0, column=1)
        ttk.Label(bounds_frame, text="to").grid(row=0, column=2)
        ttk.Entry(bounds_frame, textvariable=self.x_max_var, width=8).grid(row=0, column=3)
        
        ttk.Label(bounds_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        self.y_min_var = tk.StringVar(value="-5")
        self.y_max_var = tk.StringVar(value="5")
        ttk.Entry(bounds_frame, textvariable=self.y_min_var, width=8).grid(row=1, column=1)
        ttk.Label(bounds_frame, text="to").grid(row=1, column=2)
        ttk.Entry(bounds_frame, textvariable=self.y_max_var, width=8).grid(row=1, column=3)
        
        # Theta bounds (for unicycle)
        self.theta_bounds_row = ttk.Frame(bounds_frame)
        self.theta_bounds_row.grid(row=2, column=0, columnspan=4, sticky="ew")
        ttk.Label(self.theta_bounds_row, text="θ:").grid(row=0, column=0, sticky=tk.W)
        self.theta_min_var = tk.StringVar(value="-3.14159")
        self.theta_max_var = tk.StringVar(value="3.14159")
        ttk.Entry(self.theta_bounds_row, textvariable=self.theta_min_var, width=8).grid(row=0, column=1)
        ttk.Label(self.theta_bounds_row, text="to").grid(row=0, column=2)
        ttk.Entry(self.theta_bounds_row, textvariable=self.theta_max_var, width=8).grid(row=0, column=3)
        self.theta_bounds_row.grid_remove()
        
        # === Discretization (Eta) ===
        eta_frame = ttk.LabelFrame(self.param_frame, text="Discretization (η per dimension)", padding=10)
        eta_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(eta_frame, text="η_x:").grid(row=0, column=0, sticky=tk.W)
        self.eta_x_var = tk.StringVar(value="0.2")
        ttk.Entry(eta_frame, textvariable=self.eta_x_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(eta_frame, text="η_y:").grid(row=1, column=0, sticky=tk.W)
        self.eta_y_var = tk.StringVar(value="0.2")
        ttk.Entry(eta_frame, textvariable=self.eta_y_var, width=10).grid(row=1, column=1, padx=5)
        
        self.eta_theta_row = ttk.Frame(eta_frame)
        self.eta_theta_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(self.eta_theta_row, text="η_θ:").grid(row=0, column=0, sticky=tk.W)
        self.eta_theta_var = tk.StringVar(value="0.5")
        ttk.Entry(self.eta_theta_row, textvariable=self.eta_theta_var, width=10).grid(row=0, column=1, padx=5)
        self.eta_theta_row.grid_remove()
        
        # === Control Discretization ===
        ctrl_frame = ttk.LabelFrame(self.param_frame, text="Control Discretization", padding=10)
        ctrl_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        # Integrator controls
        self.integrator_ctrl_frame = ttk.Frame(ctrl_frame)
        self.integrator_ctrl_frame.pack(fill=tk.X)
        
        u_row = ttk.Frame(self.integrator_ctrl_frame)
        u_row.pack(fill=tk.X, pady=2)
        ttk.Label(u_row, text="u bounds:").pack(side=tk.LEFT)
        self.u_min_var = tk.StringVar(value="-1")
        self.u_max_var = tk.StringVar(value="1")
        self.u_num_var = tk.StringVar(value="5")
        ttk.Entry(u_row, textvariable=self.u_min_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(u_row, text="to").pack(side=tk.LEFT)
        ttk.Entry(u_row, textvariable=self.u_max_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(u_row, text="# pts:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(u_row, textvariable=self.u_num_var, width=4).pack(side=tk.LEFT, padx=2)
        
        # Unicycle controls
        self.unicycle_ctrl_frame = ttk.Frame(ctrl_frame)
        
        v_row = ttk.Frame(self.unicycle_ctrl_frame)
        v_row.pack(fill=tk.X, pady=2)
        ttk.Label(v_row, text="v (linear):").pack(side=tk.LEFT)
        self.v_min_var = tk.StringVar(value="-1")
        self.v_max_var = tk.StringVar(value="1")
        self.v_num_var = tk.StringVar(value="5")
        ttk.Entry(v_row, textvariable=self.v_min_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(v_row, text="to").pack(side=tk.LEFT)
        ttk.Entry(v_row, textvariable=self.v_max_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(v_row, text="# pts:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(v_row, textvariable=self.v_num_var, width=4).pack(side=tk.LEFT, padx=2)
        
        omega_row = ttk.Frame(self.unicycle_ctrl_frame)
        omega_row.pack(fill=tk.X, pady=2)
        ttk.Label(omega_row, text="ω (angular):").pack(side=tk.LEFT)
        self.omega_min_var = tk.StringVar(value="-1")
        self.omega_max_var = tk.StringVar(value="1")
        self.omega_num_var = tk.StringVar(value="5")
        ttk.Entry(omega_row, textvariable=self.omega_min_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(omega_row, text="to").pack(side=tk.LEFT)
        ttk.Entry(omega_row, textvariable=self.omega_max_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(omega_row, text="# pts:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(omega_row, textvariable=self.omega_num_var, width=4).pack(side=tk.LEFT, padx=2)
        
        # === Obstacles & Target ===
        regions_frame = ttk.LabelFrame(self.param_frame, text="Obstacles & Target", padding=10)
        regions_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        # Drawing mode selection
        mode_row = ttk.Frame(regions_frame)
        mode_row.pack(fill=tk.X, pady=5)
        
        self.draw_mode_var = tk.StringVar(value="obstacle")
        ttk.Radiobutton(mode_row, text="Draw Obstacle", variable=self.draw_mode_var,
                       value="obstacle").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Draw Target", variable=self.draw_mode_var,
                       value="target").pack(side=tk.LEFT, padx=10)
        
        # Draw enable checkbox
        self.draw_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(regions_frame, text="Enable Drawing Mode",
                       variable=self.draw_enabled_var).pack(anchor=tk.W)
        
        # Buttons
        btn_row = ttk.Frame(regions_frame)
        btn_row.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_row, text="Clear Obstacles", 
                  command=self._clear_obstacles).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Clear Target",
                  command=self._clear_target).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Undo Last",
                  command=self._undo_last).pack(side=tk.LEFT, padx=2)
        
        # Obstacles list display
        ttk.Label(regions_frame, text="Obstacles:").pack(anchor=tk.W)
        self.obstacles_text = scrolledtext.ScrolledText(regions_frame, height=3, width=40, state='disabled')
        self.obstacles_text.pack(fill=tk.X)
        
        # Target display
        ttk.Label(regions_frame, text="Target:").pack(anchor=tk.W)
        self.target_label = ttk.Label(regions_frame, text="Not set (draw on grid)")
        self.target_label.pack(anchor=tk.W)
        
        # === Start Position ===
        start_frame = ttk.LabelFrame(self.param_frame, text="Start Position", padding=10)
        start_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        pos_row = ttk.Frame(start_frame)
        pos_row.pack(fill=tk.X)
        
        ttk.Label(pos_row, text="X:").pack(side=tk.LEFT)
        self.start_x_var = tk.StringVar(value="0")
        ttk.Entry(pos_row, textvariable=self.start_x_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(pos_row, text="Y:").pack(side=tk.LEFT)
        self.start_y_var = tk.StringVar(value="-1.5")
        ttk.Entry(pos_row, textvariable=self.start_y_var, width=8).pack(side=tk.LEFT, padx=2)
        
        self.start_theta_frame = ttk.Frame(start_frame)
        self.start_theta_frame.pack(fill=tk.X)
        ttk.Label(self.start_theta_frame, text="θ:").pack(side=tk.LEFT)
        self.start_theta_var = tk.StringVar(value="0.785")
        ttk.Entry(self.start_theta_frame, textvariable=self.start_theta_var, width=8).pack(side=tk.LEFT, padx=2)
        self.start_theta_frame.pack_forget()
        
        self.click_start_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(start_frame, text="Click to set start (disables drawing)",
                       variable=self.click_start_var,
                       command=self._toggle_click_start).pack(anchor=tk.W)
        
        # === Simulation Parameters ===
        sim_frame = ttk.LabelFrame(self.param_frame, text="Simulation", padding=10)
        sim_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(sim_frame, text="Max Steps:").pack(anchor=tk.W)
        self.max_steps_var = tk.StringVar(value="100")
        ttk.Entry(sim_frame, textvariable=self.max_steps_var, width=10).pack(anchor=tk.W)
        
        # === Action Buttons ===
        btn_frame = ttk.Frame(self.param_frame)
        btn_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        
        self.run_btn = ttk.Button(btn_frame, text="Run Synthesis", command=self._run_synthesis)
        self.run_btn.pack(fill=tk.X, pady=2)
        
        self.sim_btn = ttk.Button(btn_frame, text="Run Simulation", command=self._run_simulation)
        self.sim_btn.pack(fill=tk.X, pady=2)
        self.sim_btn.state(['disabled'])
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear Output", command=self._clear_output)
        self.clear_btn.pack(fill=tk.X, pady=2)
        
        # Progress section
        progress_frame = ttk.LabelFrame(btn_frame, text="Progress", padding=5)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            mode='determinate', maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(anchor=tk.W)
        
        self.time_label = ttk.Label(progress_frame, text="")
        self.time_label.pack(anchor=tk.W)
        
    def _create_output_panel(self):
        """Create the output/log panel."""
        output_frame = ttk.LabelFrame(self.param_frame, text="Output Log", padding=5)
        output_frame.grid(row=100, column=0, sticky="nsew", padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8, width=40,
                                                     state='normal', wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
    def _create_visualization_panel(self):
        """Create the matplotlib visualization panel."""
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.ax_grid = self.axes[0]
        self.ax_sim = self.axes[1]
        
        self.ax_grid.set_title("Grid & Regions")
        self.ax_sim.set_title("Simulation")
        
        # Set default axis limits
        for ax in [self.ax_grid, self.ax_sim]:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.right_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.mpl_connect('button_release_event', self._on_plot_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_plot_motion)
        
        # Draw initial preview
        self._draw_preview()
        
    def _on_model_change(self):
        """Handle model selection change."""
        model = self.model_var.get()
        
        self.theta_bounds_row.grid_remove()
        self.eta_theta_row.grid_remove()
        self.start_theta_frame.pack_forget()
        self.integrator_ctrl_frame.pack_forget()
        self.unicycle_ctrl_frame.pack_forget()
        
        if model == "integrator":
            self.integrator_ctrl_frame.pack(fill=tk.X)
        elif model == "unicycle":
            self.theta_bounds_row.grid()
            self.eta_theta_row.grid()
            self.start_theta_frame.pack(fill=tk.X)
            self.unicycle_ctrl_frame.pack(fill=tk.X)
    
    def _toggle_click_start(self):
        """Toggle between drawing mode and click-to-set-start mode."""
        if self.click_start_var.get():
            self.draw_enabled_var.set(False)
        
    def _clear_obstacles(self):
        """Clear all obstacles."""
        self.obstacles = []
        self._update_obstacles_display()
        self._draw_preview()
        
    def _clear_target(self):
        """Clear the target region."""
        self.target = None
        self.target_label.config(text="Not set (draw on grid)")
        self._draw_preview()
        
    def _undo_last(self):
        """Undo the last drawn region."""
        if self.obstacles:
            self.obstacles.pop()
            self._update_obstacles_display()
            self._draw_preview()
    
    def _update_obstacles_display(self):
        """Update the obstacles text display."""
        self.obstacles_text.config(state='normal')
        self.obstacles_text.delete("1.0", tk.END)
        for i, obs in enumerate(self.obstacles):
            self.obstacles_text.insert(tk.END, 
                f"O{i+1}: [{obs[0]:.1f}, {obs[1]:.1f}] x [{obs[2]:.1f}, {obs[3]:.1f}]\n")
        self.obstacles_text.config(state='disabled')
        
    def _on_plot_click(self, event):
        """Handle mouse click on plot."""
        if event.inaxes not in [self.ax_grid, self.ax_sim]:
            return
        
        # Click to set start position
        if self.click_start_var.get():
            self.start_x_var.set(f"{event.xdata:.2f}")
            self.start_y_var.set(f"{event.ydata:.2f}")
            self._draw_preview()
            return
        
        # Drawing mode
        if self.draw_enabled_var.get() and event.button == 1:
            self.drawing = True
            self.draw_start_point = (event.xdata, event.ydata)
            self.draw_mode = self.draw_mode_var.get()
    
    def _on_plot_release(self, event):
        """Handle mouse release on plot."""
        if not self.drawing or self.draw_start_point is None:
            return
        
        if event.inaxes not in [self.ax_grid, self.ax_sim]:
            self.drawing = False
            self.draw_start_point = None
            self._remove_temp_rect()
            return
        
        x1, y1 = self.draw_start_point
        x2, y2 = event.xdata, event.ydata
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Check minimum size
        if abs(x_max - x_min) < 0.1 or abs(y_max - y_min) < 0.1:
            self._log("Region too small, ignoring.")
            self.drawing = False
            self.draw_start_point = None
            self._remove_temp_rect()
            return
        
        region = [x_min, x_max, y_min, y_max]
        
        if self.draw_mode == "obstacle":
            self.obstacles.append(region)
            self._update_obstacles_display()
            self._log(f"Added obstacle: [{x_min:.2f}, {x_max:.2f}] x [{y_min:.2f}, {y_max:.2f}]")
        else:
            self.target = region
            self.target_label.config(text=f"[{x_min:.2f}, {x_max:.2f}] x [{y_min:.2f}, {y_max:.2f}]")
            self._log(f"Set target: [{x_min:.2f}, {x_max:.2f}] x [{y_min:.2f}, {y_max:.2f}]")
        
        self.drawing = False
        self.draw_start_point = None
        self._remove_temp_rect()
        self._draw_preview()
    
    def _on_plot_motion(self, event):
        """Handle mouse motion for drawing preview."""
        if not self.drawing or self.draw_start_point is None:
            return
        
        if event.inaxes not in [self.ax_grid, self.ax_sim]:
            return
        
        x1, y1 = self.draw_start_point
        x2, y2 = event.xdata, event.ydata
        
        self._remove_temp_rect()
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        width = x_max - x_min
        height = y_max - y_min
        
        color = 'red' if self.draw_mode == "obstacle" else 'green'
        self.temp_rect = Rectangle(
            (x_min, y_min), width, height,
            facecolor=to_rgba(color, 0.3),
            edgecolor=color, linewidth=2, linestyle='--'
        )
        self.temp_rect.temp_rect = True
        self.ax_grid.add_patch(self.temp_rect)
        self.canvas.draw_idle()
    
    def _remove_temp_rect(self):
        """Remove temporary drawing rectangle."""
        if self.temp_rect is not None:
            try:
                self.temp_rect.remove()
            except Exception:
                pass
            self.temp_rect = None
        
        for ax in [self.ax_grid, self.ax_sim]:
            for artist in list(ax.patches):
                if hasattr(artist, 'temp_rect'):
                    try:
                        artist.remove()
                    except Exception:
                        pass
        self.canvas.draw_idle()
    
    def _draw_preview(self):
        """Draw preview of obstacles and target."""
        self.ax_grid.clear()
        self.ax_sim.clear()
        
        try:
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
        except ValueError:
            x_min, x_max = -5, 5
            y_min, y_max = -5, 5
        
        for ax in [self.ax_grid, self.ax_sim]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        self.ax_grid.set_title("Grid & Regions (Preview)")
        self.ax_sim.set_title("Simulation")
        
        # Draw obstacles
        for obs in self.obstacles:
            for ax in [self.ax_grid, self.ax_sim]:
                rect = Rectangle(
                    (obs[0], obs[2]), obs[1] - obs[0], obs[3] - obs[2],
                    facecolor=to_rgba('black', 0.7),
                    edgecolor='black', linewidth=2
                )
                ax.add_patch(rect)
        
        # Draw target
        if self.target is not None:
            for ax in [self.ax_grid, self.ax_sim]:
                rect = Rectangle(
                    (self.target[0], self.target[2]),
                    self.target[1] - self.target[0],
                    self.target[3] - self.target[2],
                    facecolor=to_rgba('blue', 0.4),
                    edgecolor='blue', linewidth=2
                )
                ax.add_patch(rect)
                ax.text(
                    (self.target[0] + self.target[1]) / 2,
                    (self.target[2] + self.target[3]) / 2,
                    "Target", ha='center', va='center', fontsize=10, fontweight='bold'
                )
        
        # Draw start position marker
        try:
            start_x = float(self.start_x_var.get())
            start_y = float(self.start_y_var.get())
            for ax in [self.ax_grid, self.ax_sim]:
                ax.plot(start_x, start_y, 'g*', markersize=15)
                if self.model_var.get() == "unicycle":
                    try:
                        theta = float(self.start_theta_var.get())
                        dx = 0.5 * np.cos(theta)
                        dy = 0.5 * np.sin(theta)
                        ax.arrow(start_x, start_y, dx, dy, head_width=0.2, head_length=0.1,
                                fc='green', ec='green')
                    except ValueError:
                        pass
        except ValueError:
            pass
        
        self.canvas.draw()
    
    def _log(self, message: str):
        """Log a message."""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.update_idletasks()
        
    def _clear_output(self):
        """Clear the output log."""
        self.output_text.delete("1.0", tk.END)
    
    def _update_progress(self, current, total, message):
        """Update progress bar and label."""
        progress = (current / total) * 100 if total > 0 else 0
        
        if self.synthesis_start_time is not None:
            elapsed = time.time() - self.synthesis_start_time
            if current > 0:
                rate = elapsed / current
                remaining = rate * (total - current)
                time_str = f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s"
            else:
                time_str = f"Elapsed: {elapsed:.1f}s"
        else:
            time_str = ""
        
        self.root.after(0, lambda: self.progress_var.set(progress))
        self.root.after(0, lambda: self.progress_label.config(text=message))
        self.root.after(0, lambda: self.time_label.config(text=time_str))
    
    def _run_synthesis(self):
        """Run the synthesis pipeline."""
        if self.target is None:
            messagebox.showwarning("Warning", "Please draw a target region first")
            return
        
        self.run_btn.state(['disabled'])
        self.sim_btn.state(['disabled'])
        self.progress_var.set(0)
        self.progress_label.config(text="Starting...")
        self.time_label.config(text="")
        self.synthesis_start_time = None
        
        thread = threading.Thread(target=self._synthesis_thread)
        thread.start()
    
    def _synthesis_thread(self):
        """Worker thread for synthesis."""
        try:
            self._do_synthesis()
        except Exception as e:
            import traceback
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self._log(f"ERROR: {e}"))
            self.root.after(0, lambda: self._log(traceback.format_exc()))
        finally:
            self.root.after(0, self._synthesis_complete)
    
    def _synthesis_complete(self):
        """Called when synthesis is complete."""
        self.progress_var.set(100)
        self.progress_label.config(text="Complete!")
        self.run_btn.state(['!disabled'])
        if self.synth is not None and len(self.synth.winning_set) > 0:
            self.sim_btn.state(['!disabled'])
    
    def _do_synthesis(self):
        """Execute the synthesis pipeline."""
        from symbolic_control import IntegratorDynamics, UnicycleDynamics, Abstraction
        from symbolic_control.synthesis import Synthesis
        
        self.synthesis_start_time = time.time()
        
        self.root.after(0, lambda: self._log("=" * 50))
        self.root.after(0, lambda: self._log("Starting simple synthesis pipeline..."))
        self.root.after(0, lambda: self._log("=" * 50))
        
        # Parse parameters
        tau = float(self.tau_var.get())
        w_bound = float(self.w_bound_var.get())
        
        x_min = float(self.x_min_var.get())
        x_max = float(self.x_max_var.get())
        y_min = float(self.y_min_var.get())
        y_max = float(self.y_max_var.get())
        
        eta_x = float(self.eta_x_var.get())
        eta_y = float(self.eta_y_var.get())
        
        model = self.model_var.get()
        
        if model == "integrator":
            u_min = float(self.u_min_var.get())
            u_max = float(self.u_max_var.get())
            u_num = int(self.u_num_var.get())
            u_values = list(np.linspace(u_min, u_max, num=u_num))
            
            self.dynamics = IntegratorDynamics(tau=tau, w_bound=w_bound, u_values=u_values)
            state_bounds = np.array([[x_min, x_max], [y_min, y_max]])
            eta = [eta_x, eta_y]
            self.root.after(0, lambda: self._log("Model: 2D Integrator"))
        else:
            theta_min = float(self.theta_min_var.get())
            theta_max = float(self.theta_max_var.get())
            eta_theta = float(self.eta_theta_var.get())
            
            v_min = float(self.v_min_var.get())
            v_max = float(self.v_max_var.get())
            v_num = int(self.v_num_var.get())
            v_values = list(np.linspace(v_min, v_max, num=v_num))
            
            omega_min = float(self.omega_min_var.get())
            omega_max = float(self.omega_max_var.get())
            omega_num = int(self.omega_num_var.get())
            omega_values = list(np.linspace(omega_min, omega_max, num=omega_num))
            
            self.dynamics = UnicycleDynamics(tau=tau, w_bound=w_bound,
                                            v_values=v_values, omega_values=omega_values)
            state_bounds = np.array([[x_min, x_max], [y_min, y_max], [theta_min, theta_max]])
            eta = [eta_x, eta_y, eta_theta]
            self.root.after(0, lambda: self._log("Model: 3D Unicycle"))
        
        self.root.after(0, lambda: self._log(f"State bounds: {state_bounds.tolist()}"))
        self.root.after(0, lambda: self._log(f"Eta: {eta}"))
        self.root.after(0, lambda: self._log(f"Control set size: {len(self.dynamics.control_set)}"))
        
        # Create abstraction
        self._update_progress(0, 100, "Creating abstraction...")
        
        self.abstraction = Abstraction(
            dynamics=self.dynamics,
            state_bounds=state_bounds,
            eta=eta
        )
        
        self.root.after(0, lambda: self._log(f"Grid shape: {self.abstraction.grid_shape}"))
        self.root.after(0, lambda: self._log(f"Total cells: {self.abstraction.num_cells}"))
        
        # Build transitions
        self._update_progress(5, 100, "Building transitions...")
        self.root.after(0, lambda: self._log("\nBuilding transitions..."))
        
        transition_start = time.time()
        self.abstraction.build_transitions(progress_callback=self._transition_progress)
        transition_time = time.time() - transition_start
        self.root.after(0, lambda: self._log(f"Transitions built in {transition_time:.1f}s"))
        
        # Convert obstacles and target to cell sets
        self._update_progress(60, 100, "Processing regions...")
        
        obstacle_cells = self._regions_to_cells(self.obstacles)
        target_cells = self._region_to_cells(self.target)
        
        self.root.after(0, lambda: self._log(f"\nObstacle cells: {len(obstacle_cells)}"))
        self.root.after(0, lambda: self._log(f"Target cells: {len(target_cells)}"))
        
        # Run synthesis
        self._update_progress(70, 100, "Running synthesis...")
        self.root.after(0, lambda: self._log("\nRunning synthesis..."))
        
        self.synth = Synthesis(self.abstraction, obstacle_cells, target_cells)
        winning_set = self.synth.run(verbose=False)
        
        self.root.after(0, lambda: self._log(f"\nSafe cells: {len(self.synth.safe_set)}"))
        self.root.after(0, lambda: self._log(f"Winning cells: {len(winning_set)}"))
        
        total_time = time.time() - self.synthesis_start_time
        self.root.after(0, lambda: self._log(f"\nTotal synthesis time: {total_time:.1f}s"))
        
        self._update_progress(100, 100, "Synthesis complete!")
        
        # Update visualization
        self.root.after(0, self._draw_synthesis_result)
    
    def _transition_progress(self, current, total, message):
        """Progress callback for transition building."""
        progress = 5 + (current / total) * 55 if total > 0 else 5
        self._update_progress(progress, 100, message)
    
    def _region_to_cells(self, region: List[float]) -> Set[int]:
        """Convert a region to cell indices."""
        if region is None:
            return set()
        
        rx_min, rx_max, ry_min, ry_max = region
        cells = set()
        
        state_dim = self.dynamics.state_dim
        grid_shape = self.abstraction.grid_shape
        
        if state_dim == 2:
            nx, ny = grid_shape
            for i in range(nx):
                for j in range(ny):
                    cx = self.abstraction.state_bounds[0, 0] + (i + 0.5) * self.abstraction.eta[0]
                    cy = self.abstraction.state_bounds[1, 0] + (j + 0.5) * self.abstraction.eta[1]
                    if rx_min <= cx < rx_max and ry_min <= cy < ry_max:
                        cell_idx = np.ravel_multi_index((i, j), grid_shape)
                        cells.add(cell_idx)
        else:  # 3D (unicycle)
            nx, ny, ntheta = grid_shape
            for i in range(nx):
                for j in range(ny):
                    cx = self.abstraction.state_bounds[0, 0] + (i + 0.5) * self.abstraction.eta[0]
                    cy = self.abstraction.state_bounds[1, 0] + (j + 0.5) * self.abstraction.eta[1]
                    if rx_min <= cx < rx_max and ry_min <= cy < ry_max:
                        # Add all theta slices
                        for k in range(ntheta):
                            cell_idx = np.ravel_multi_index((i, j, k), grid_shape)
                            cells.add(cell_idx)
        
        return cells
    
    def _regions_to_cells(self, regions: List[List[float]]) -> Set[int]:
        """Convert multiple regions to cell indices."""
        cells = set()
        for region in regions:
            cells |= self._region_to_cells(region)
        return cells
    
    def _draw_synthesis_result(self):
        """Draw the synthesis results."""
        self.ax_grid.clear()
        self.ax_sim.clear()
        
        x_min, x_max = self.abstraction.state_bounds[0]
        y_min, y_max = self.abstraction.state_bounds[1]
        
        eta_x = self.abstraction.eta[0]
        eta_y = self.abstraction.eta[1]
        
        state_dim = self.dynamics.state_dim
        grid_shape = self.abstraction.grid_shape
        
        # Project to 2D if needed
        if state_dim == 2:
            nx, ny = grid_shape
            winning_xy = set()
            safe_xy = set()
            obstacle_xy = set()
            target_xy = set()
            
            for cell_idx in self.synth.winning_set:
                i, j = np.unravel_index(cell_idx, grid_shape)
                winning_xy.add((i, j))
            for cell_idx in self.synth.safe_set:
                i, j = np.unravel_index(cell_idx, grid_shape)
                safe_xy.add((i, j))
            for cell_idx in self.synth.obstacles:
                i, j = np.unravel_index(cell_idx, grid_shape)
                obstacle_xy.add((i, j))
            for cell_idx in self.synth.target:
                i, j = np.unravel_index(cell_idx, grid_shape)
                target_xy.add((i, j))
        else:
            nx, ny, ntheta = grid_shape
            winning_xy = set()
            safe_xy = set()
            obstacle_xy = set()
            target_xy = set()
            
            for cell_idx in self.synth.winning_set:
                i, j, k = np.unravel_index(cell_idx, grid_shape)
                winning_xy.add((i, j))
            for cell_idx in self.synth.safe_set:
                i, j, k = np.unravel_index(cell_idx, grid_shape)
                safe_xy.add((i, j))
            for cell_idx in self.synth.obstacles:
                i, j, k = np.unravel_index(cell_idx, grid_shape)
                obstacle_xy.add((i, j))
            for cell_idx in self.synth.target:
                i, j, k = np.unravel_index(cell_idx, grid_shape)
                target_xy.add((i, j))
        
        # Draw cells
        for i in range(nx):
            for j in range(ny):
                x = self.abstraction.state_bounds[0, 0] + i * eta_x
                y = self.abstraction.state_bounds[1, 0] + j * eta_y
                
                if (i, j) in obstacle_xy:
                    color = 'black'
                    alpha = 0.9
                elif (i, j) in target_xy:
                    color = 'blue'
                    alpha = 0.7
                elif (i, j) in winning_xy:
                    color = 'lightgreen'
                    alpha = 0.6
                elif (i, j) in safe_xy:
                    color = 'khaki'
                    alpha = 0.5
                else:
                    color = 'salmon'
                    alpha = 0.5
                
                rect = Rectangle(
                    (x, y), eta_x, eta_y,
                    linewidth=0.3, edgecolor='gray',
                    facecolor=color, alpha=alpha
                )
                self.ax_grid.add_patch(rect)
        
        # Set limits and labels
        self.ax_grid.set_xlim(x_min, x_max)
        self.ax_grid.set_ylim(y_min, y_max)
        self.ax_grid.set_xlabel('X')
        self.ax_grid.set_ylabel('Y')
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_title(f"Synthesis Results (η=[{eta_x:.2f}, {eta_y:.2f}])")
        self.ax_grid.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(color='black', alpha=0.9, label='Obstacle'),
            Patch(color='blue', alpha=0.7, label='Target'),
            Patch(color='lightgreen', alpha=0.6, label='Winning'),
            Patch(color='khaki', alpha=0.5, label='Safe (not winning)'),
            Patch(color='salmon', alpha=0.5, label='Unsafe'),
        ]
        self.ax_grid.legend(handles=legend_patches, loc='upper left', fontsize=8)
        
        # Copy to sim axis
        self.ax_sim.set_xlim(x_min, x_max)
        self.ax_sim.set_ylim(y_min, y_max)
        self.ax_sim.set_xlabel('X')
        self.ax_sim.set_ylabel('Y')
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_title("Simulation")
        self.ax_sim.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def _run_simulation(self):
        """Run simulation from start position."""
        if self.synth is None:
            messagebox.showwarning("Warning", "Please run synthesis first")
            return
        
        try:
            start_x = float(self.start_x_var.get())
            start_y = float(self.start_y_var.get())
            
            if self.model_var.get() == "unicycle":
                start_theta = float(self.start_theta_var.get())
                start_pos = np.array([start_x, start_y, start_theta])
            else:
                start_pos = np.array([start_x, start_y])
            
            max_steps = int(self.max_steps_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid start position: {e}")
            return
        
        # Check if start is in winning set
        start_cell = self.abstraction.point_to_cell(start_pos)
        if start_cell == -1:
            messagebox.showerror("Error", "Start position is out of bounds")
            return
        
        if not self.synth.is_winning(start_cell):
            self._log("Warning: Start position is not in winning set!")
        
        self._log(f"\nSimulating from {start_pos}...")
        self._log(f"Start cell: {start_cell}")
        self._log(f"In winning set: {self.synth.is_winning(start_cell)}")
        
        try:
            trajectory = self._simulate(start_pos, max_steps)
            self.trajectory = trajectory
            self._log(f"Trajectory length: {len(trajectory)} steps")
            self._draw_simulation()
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {e}")
            self._log(f"ERROR: {e}")
    
    def _simulate(self, start_pos: np.ndarray, max_steps: int) -> np.ndarray:
        """Run simulation."""
        trajectory = [np.array(start_pos)]
        pos = np.array(start_pos)
        w_bound = float(self.w_bound_var.get())
        
        for step in range(max_steps):
            cell_idx = self.abstraction.point_to_cell(pos)
            if cell_idx == -1:
                self._log(f"Step {step}: Out of bounds at {pos}")
                break
            
            if self.synth.is_at_target(cell_idx):
                self._log(f"Step {step}: Reached target at {pos}")
                break
            
            u_idx = self.synth.get_control(cell_idx)
            if u_idx is None:
                self._log(f"Step {step}: No control at cell {cell_idx}")
                break
            
            u = self.dynamics.control_set[u_idx]
            w = np.random.uniform(-w_bound, w_bound, size=self.dynamics.disturbance_dim)
            next_pos = self.dynamics.step(pos, u, w)
            
            trajectory.append(next_pos.copy())
            pos = next_pos
        
        return np.array(trajectory)
    
    def _draw_simulation(self):
        """Draw the simulation trajectory."""
        if self.trajectory is None:
            return
        
        self.ax_sim.clear()
        
        x_min, x_max = self.abstraction.state_bounds[0]
        y_min, y_max = self.abstraction.state_bounds[1]
        
        # Draw obstacles and target
        for obs in self.obstacles:
            rect = Rectangle(
                (obs[0], obs[2]), obs[1] - obs[0], obs[3] - obs[2],
                facecolor=to_rgba('black', 0.7),
                edgecolor='black', linewidth=2
            )
            self.ax_sim.add_patch(rect)
        
        if self.target is not None:
            rect = Rectangle(
                (self.target[0], self.target[2]),
                self.target[1] - self.target[0],
                self.target[3] - self.target[2],
                facecolor=to_rgba('blue', 0.4),
                edgecolor='blue', linewidth=2
            )
            self.ax_sim.add_patch(rect)
        
        # Draw trajectory
        traj = self.trajectory
        self.ax_sim.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        self.ax_sim.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
        self.ax_sim.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End')
        
        # Draw heading arrows for unicycle
        if self.model_var.get() == "unicycle" and traj.shape[1] >= 3:
            step = max(1, len(traj) // 20)
            for i in range(0, len(traj), step):
                x, y, theta = traj[i, 0], traj[i, 1], traj[i, 2]
                dx = 0.3 * np.cos(theta)
                dy = 0.3 * np.sin(theta)
                self.ax_sim.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05,
                                 fc='blue', ec='blue', alpha=0.5)
        
        self.ax_sim.set_xlim(x_min, x_max)
        self.ax_sim.set_ylim(y_min, y_max)
        self.ax_sim.set_xlabel('X')
        self.ax_sim.set_ylabel('Y')
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_title(f"Simulation ({len(traj)} steps)")
        self.ax_sim.legend(loc='upper left')
        self.ax_sim.grid(True, alpha=0.3)
        
        self.canvas.draw()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SimpleSynthesisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
