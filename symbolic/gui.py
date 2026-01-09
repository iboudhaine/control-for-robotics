"""
Symbolic Control GUI - Interactive interface for controller synthesis and simulation.

This GUI provides a complete pipeline from discretization to simulation:
1. Select dynamics model (Integrator or Unicycle)
2. Configure grid parameters (bounds, per-dimension eta)
3. Define regions and regex specification
4. Run synthesis and visualize results
5. Simulate controller with interactive start position
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.colors import to_rgba
import threading
import io
import sys
import time
from typing import Dict, List, Optional, Tuple


class OutputRedirector:
    """Redirects stdout to a text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = io.StringIO()
        
    def write(self, text):
        self.buffer.write(text)
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()
        
    def flush(self):
        pass


class SymbolicControlGUI:
    """Main GUI application for symbolic control synthesis."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Symbolic Control Synthesis")
        self.root.geometry("1400x900")
        
        # State variables
        self.abstraction = None
        self.synth = None
        self.dynamics = None
        self.trajectory = None
        self.regions = {}
        self.synthesis_start_time = None
        
        # Region drawing state
        self.drawing_region = False
        self.draw_start_point = None
        self.temp_rect = None
        
        # Create main layout
        self._create_layout()
        self._create_parameter_panel()
        self._create_output_panel()
        self._create_visualization_panel()
        
        # Bind window resize
        self.root.bind('<Configure>', self._on_resize)
        
    def _create_layout(self):
        """Create the main layout with left panel and right visualization."""
        # Main paned window
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
        ttk.Radiobutton(model_frame, text="4D Manipulator", variable=self.model_var, 
                       value="manipulator", command=self._on_model_change).pack(anchor=tk.W)
        
        # === Dynamics Parameters ===
        dyn_frame = ttk.LabelFrame(self.param_frame, text="Dynamics Parameters", padding=10)
        dyn_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(dyn_frame, text="Sampling Period (τ):").grid(row=0, column=0, sticky=tk.W)
        self.tau_var = tk.StringVar(value="0.4")
        ttk.Entry(dyn_frame, textvariable=self.tau_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(dyn_frame, text="Disturbance Bound:").grid(row=1, column=0, sticky=tk.W)
        self.w_bound_var = tk.StringVar(value="0.01")
        ttk.Entry(dyn_frame, textvariable=self.w_bound_var, width=10).grid(row=1, column=1, padx=5)
        
        # === State Bounds ===
        bounds_frame = ttk.LabelFrame(self.param_frame, text="State Bounds", padding=10)
        bounds_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        # X bounds
        ttk.Label(bounds_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x_min_var = tk.StringVar(value="0")
        self.x_max_var = tk.StringVar(value="10")
        ttk.Entry(bounds_frame, textvariable=self.x_min_var, width=8).grid(row=0, column=1)
        ttk.Label(bounds_frame, text="to").grid(row=0, column=2)
        ttk.Entry(bounds_frame, textvariable=self.x_max_var, width=8).grid(row=0, column=3)
        
        # Y bounds
        ttk.Label(bounds_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        self.y_min_var = tk.StringVar(value="0")
        self.y_max_var = tk.StringVar(value="10")
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
        self.theta_bounds_row.grid_remove()  # Hidden by default
        
        # Manipulator bounds (θ₁, θ₂, θ̇₁, θ̇₂)
        self.manip_bounds_frame = ttk.Frame(bounds_frame)
        self.manip_bounds_frame.grid(row=3, column=0, columnspan=4, sticky="ew")
        
        # θ₁ bounds
        ttk.Label(self.manip_bounds_frame, text="θ₁:").grid(row=0, column=0, sticky=tk.W)
        self.theta1_min_var = tk.StringVar(value="-3.14159")
        self.theta1_max_var = tk.StringVar(value="3.14159")
        ttk.Entry(self.manip_bounds_frame, textvariable=self.theta1_min_var, width=8).grid(row=0, column=1)
        ttk.Label(self.manip_bounds_frame, text="to").grid(row=0, column=2)
        ttk.Entry(self.manip_bounds_frame, textvariable=self.theta1_max_var, width=8).grid(row=0, column=3)
        
        # θ₂ bounds
        ttk.Label(self.manip_bounds_frame, text="θ₂:").grid(row=1, column=0, sticky=tk.W)
        self.theta2_min_var = tk.StringVar(value="-3.14159")
        self.theta2_max_var = tk.StringVar(value="3.14159")
        ttk.Entry(self.manip_bounds_frame, textvariable=self.theta2_min_var, width=8).grid(row=1, column=1)
        ttk.Label(self.manip_bounds_frame, text="to").grid(row=1, column=2)
        ttk.Entry(self.manip_bounds_frame, textvariable=self.theta2_max_var, width=8).grid(row=1, column=3)
        
        # θ̇₁ bounds
        ttk.Label(self.manip_bounds_frame, text="θ̇₁:").grid(row=2, column=0, sticky=tk.W)
        self.dtheta1_min_var = tk.StringVar(value="-5")
        self.dtheta1_max_var = tk.StringVar(value="5")
        ttk.Entry(self.manip_bounds_frame, textvariable=self.dtheta1_min_var, width=8).grid(row=2, column=1)
        ttk.Label(self.manip_bounds_frame, text="to").grid(row=2, column=2)
        ttk.Entry(self.manip_bounds_frame, textvariable=self.dtheta1_max_var, width=8).grid(row=2, column=3)
        
        # θ̇₂ bounds
        ttk.Label(self.manip_bounds_frame, text="θ̇₂:").grid(row=3, column=0, sticky=tk.W)
        self.dtheta2_min_var = tk.StringVar(value="-5")
        self.dtheta2_max_var = tk.StringVar(value="5")
        ttk.Entry(self.manip_bounds_frame, textvariable=self.dtheta2_min_var, width=8).grid(row=3, column=1)
        ttk.Label(self.manip_bounds_frame, text="to").grid(row=3, column=2)
        ttk.Entry(self.manip_bounds_frame, textvariable=self.dtheta2_max_var, width=8).grid(row=3, column=3)
        
        self.manip_bounds_frame.grid_remove()  # Hidden by default
        
        # === Discretization (Eta) ===
        eta_frame = ttk.LabelFrame(self.param_frame, text="Discretization (η per dimension)", padding=10)
        eta_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(eta_frame, text="η_x:").grid(row=0, column=0, sticky=tk.W)
        self.eta_x_var = tk.StringVar(value="0.5")
        ttk.Entry(eta_frame, textvariable=self.eta_x_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(eta_frame, text="η_y:").grid(row=1, column=0, sticky=tk.W)
        self.eta_y_var = tk.StringVar(value="0.5")
        ttk.Entry(eta_frame, textvariable=self.eta_y_var, width=10).grid(row=1, column=1, padx=5)
        
        # Theta eta (for unicycle)
        self.eta_theta_row = ttk.Frame(eta_frame)
        self.eta_theta_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(self.eta_theta_row, text="η_θ:").grid(row=0, column=0, sticky=tk.W)
        self.eta_theta_var = tk.StringVar(value="0.5")
        ttk.Entry(self.eta_theta_row, textvariable=self.eta_theta_var, width=10).grid(row=0, column=1, padx=5)
        self.eta_theta_row.grid_remove()  # Hidden by default
        
        # Manipulator eta (4 dimensions)
        self.eta_manip_frame = ttk.Frame(eta_frame)
        self.eta_manip_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        
        ttk.Label(self.eta_manip_frame, text="η_θ₁:").grid(row=0, column=0, sticky=tk.W)
        self.eta_theta1_var = tk.StringVar(value="0.3")
        ttk.Entry(self.eta_manip_frame, textvariable=self.eta_theta1_var, width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.eta_manip_frame, text="η_θ₂:").grid(row=1, column=0, sticky=tk.W)
        self.eta_theta2_var = tk.StringVar(value="0.3")
        ttk.Entry(self.eta_manip_frame, textvariable=self.eta_theta2_var, width=8).grid(row=1, column=1, padx=5)
        
        ttk.Label(self.eta_manip_frame, text="η_θ̇₁:").grid(row=2, column=0, sticky=tk.W)
        self.eta_dtheta1_var = tk.StringVar(value="0.5")
        ttk.Entry(self.eta_manip_frame, textvariable=self.eta_dtheta1_var, width=8).grid(row=2, column=1, padx=5)
        
        ttk.Label(self.eta_manip_frame, text="η_θ̇₂:").grid(row=3, column=0, sticky=tk.W)
        self.eta_dtheta2_var = tk.StringVar(value="0.5")
        ttk.Entry(self.eta_manip_frame, textvariable=self.eta_dtheta2_var, width=8).grid(row=3, column=1, padx=5)
        
        self.eta_manip_frame.grid_remove()  # Hidden by default
        
        # === Control Discretization ===
        ctrl_frame = ttk.LabelFrame(self.param_frame, text="Control Discretization", padding=10)
        ctrl_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        # Integrator controls (u1 and u2 have same bounds)
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
        
        # Unicycle controls (v = linear velocity, omega = angular velocity)
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
        
        # Manipulator controls (torques τ₁ and τ₂)
        self.manip_ctrl_frame = ttk.Frame(ctrl_frame)
        
        tau1_row = ttk.Frame(self.manip_ctrl_frame)
        tau1_row.pack(fill=tk.X, pady=2)
        ttk.Label(tau1_row, text="τ₁ (torque 1):").pack(side=tk.LEFT)
        self.tau1_min_var = tk.StringVar(value="-10")
        self.tau1_max_var = tk.StringVar(value="10")
        self.tau1_num_var = tk.StringVar(value="5")
        ttk.Entry(tau1_row, textvariable=self.tau1_min_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(tau1_row, text="to").pack(side=tk.LEFT)
        ttk.Entry(tau1_row, textvariable=self.tau1_max_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(tau1_row, text="# pts:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(tau1_row, textvariable=self.tau1_num_var, width=4).pack(side=tk.LEFT, padx=2)
        
        tau2_row = ttk.Frame(self.manip_ctrl_frame)
        tau2_row.pack(fill=tk.X, pady=2)
        ttk.Label(tau2_row, text="τ₂ (torque 2):").pack(side=tk.LEFT)
        self.tau2_min_var = tk.StringVar(value="-10")
        self.tau2_max_var = tk.StringVar(value="10")
        self.tau2_num_var = tk.StringVar(value="5")
        ttk.Entry(tau2_row, textvariable=self.tau2_min_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(tau2_row, text="to").pack(side=tk.LEFT)
        ttk.Entry(tau2_row, textvariable=self.tau2_max_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(tau2_row, text="# pts:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(tau2_row, textvariable=self.tau2_num_var, width=4).pack(side=tk.LEFT, padx=2)
        
        # === Regions ===
        regions_frame = ttk.LabelFrame(self.param_frame, text="Regions (name: x_min, x_max, y_min, y_max)", padding=10)
        regions_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        self.regions_text = scrolledtext.ScrolledText(regions_frame, height=6, width=40)
        self.regions_text.pack(fill=tk.X)
        self.regions_text.insert(tk.END, "A: 0.5, 3.0, 0.5, 3.0\nB: 7.0, 9.5, 7.0, 9.5\nO: 1.0, 9.0, 4.0, 6.0")
        
        # Region drawing controls
        region_btn_row = ttk.Frame(regions_frame)
        region_btn_row.pack(fill=tk.X, pady=5)
        
        self.draw_region_var = tk.BooleanVar(value=False)
        self.draw_region_btn = ttk.Checkbutton(
            region_btn_row, text="Draw Region Mode", 
            variable=self.draw_region_var,
            command=self._toggle_draw_mode
        )
        self.draw_region_btn.pack(side=tk.LEFT)
        
        self.clear_regions_btn = ttk.Button(
            region_btn_row, text="Clear All", 
            command=self._clear_regions
        )
        self.clear_regions_btn.pack(side=tk.RIGHT, padx=2)
        
        self.delete_region_btn = ttk.Button(
            region_btn_row, text="Delete Last",
            command=self._delete_last_region
        )
        self.delete_region_btn.pack(side=tk.RIGHT, padx=2)
        
        # Region name entry
        region_name_row = ttk.Frame(regions_frame)
        region_name_row.pack(fill=tk.X, pady=2)
        ttk.Label(region_name_row, text="New region name:").pack(side=tk.LEFT)
        self.new_region_name_var = tk.StringVar(value="")
        self.new_region_entry = ttk.Entry(region_name_row, textvariable=self.new_region_name_var, width=10)
        self.new_region_entry.pack(side=tk.LEFT, padx=5)
        
        # === Specification ===
        spec_frame = ttk.LabelFrame(self.param_frame, text="Specification (RegEx)", padding=10)
        spec_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        self.spec_var = tk.StringVar(value="A[^O]*B")
        ttk.Entry(spec_frame, textvariable=self.spec_var, width=40).pack(fill=tk.X)
        ttk.Label(spec_frame, text="Examples: AB, A|B, A[^O]*B, (A|B)C", 
                 font=('TkDefaultFont', 8)).pack(anchor=tk.W)
        
        # === Start Position ===
        start_frame = ttk.LabelFrame(self.param_frame, text="Start Position", padding=10)
        start_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        pos_row = ttk.Frame(start_frame)
        pos_row.pack(fill=tk.X)
        
        ttk.Label(pos_row, text="X:").pack(side=tk.LEFT)
        self.start_x_var = tk.StringVar(value="1.5")
        ttk.Entry(pos_row, textvariable=self.start_x_var, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(pos_row, text="Y:").pack(side=tk.LEFT)
        self.start_y_var = tk.StringVar(value="1.5")
        ttk.Entry(pos_row, textvariable=self.start_y_var, width=8).pack(side=tk.LEFT, padx=2)
        
        self.start_theta_frame = ttk.Frame(start_frame)
        self.start_theta_frame.pack(fill=tk.X)
        ttk.Label(self.start_theta_frame, text="θ:").pack(side=tk.LEFT)
        self.start_theta_var = tk.StringVar(value="0.785")
        ttk.Entry(self.start_theta_frame, textvariable=self.start_theta_var, width=8).pack(side=tk.LEFT, padx=2)
        self.start_theta_frame.pack_forget()  # Hidden by default
        
        # Manipulator velocity start position (hidden by default)
        self.start_vel_frame = ttk.Frame(start_frame)
        self.start_vel_frame.pack(fill=tk.X)
        ttk.Label(self.start_vel_frame, text="θ̇₁:").pack(side=tk.LEFT)
        self.start_theta1_dot_var = tk.StringVar(value="0.0")
        ttk.Entry(self.start_vel_frame, textvariable=self.start_theta1_dot_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.start_vel_frame, text="θ̇₂:").pack(side=tk.LEFT)
        self.start_theta2_dot_var = tk.StringVar(value="0.0")
        ttk.Entry(self.start_vel_frame, textvariable=self.start_theta2_dot_var, width=8).pack(side=tk.LEFT, padx=2)
        self.start_vel_frame.pack_forget()  # Hidden by default
        
        # Click to set position
        self.click_to_set_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(start_frame, text="Click on plot to set start", 
                       variable=self.click_to_set_var).pack(anchor=tk.W)
        
        # === Simulation Parameters ===
        sim_frame = ttk.LabelFrame(self.param_frame, text="Simulation", padding=10)
        sim_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        row += 1
        
        ttk.Label(sim_frame, text="Max Steps:").pack(anchor=tk.W)
        self.max_steps_var = tk.StringVar(value="200")
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
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, width=40, 
                                                     state='normal', wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
    def _create_visualization_panel(self):
        """Create the matplotlib visualization panel."""
        # Create figure with two subplots
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.ax_grid = self.axes[0]
        self.ax_sim = self.axes[1]
        
        self.ax_grid.set_title("Grid Abstraction & Regions")
        self.ax_sim.set_title("Simulation")
        
        # Set default axis limits based on state bounds
        for ax in [self.ax_grid, self.ax_sim]:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
        
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
        
        # Draw initial regions
        self._draw_regions_preview()
        
    def _on_model_change(self):
        """Handle model selection change."""
        model = self.model_var.get()
        
        # Hide all model-specific UI elements first
        self.theta_bounds_row.grid_remove()
        self.manip_bounds_frame.grid_remove()
        self.eta_theta_row.grid_remove()
        self.eta_manip_frame.grid_remove()
        self.start_theta_frame.pack_forget()
        self.start_vel_frame.pack_forget()
        self.integrator_ctrl_frame.pack_forget()
        self.unicycle_ctrl_frame.pack_forget()
        self.manip_ctrl_frame.pack_forget()
        
        # Show elements specific to the selected model
        if model == "integrator":
            self.integrator_ctrl_frame.pack(fill=tk.X)
        elif model == "unicycle":
            self.theta_bounds_row.grid()
            self.eta_theta_row.grid()
            self.start_theta_frame.pack(fill=tk.X)
            self.unicycle_ctrl_frame.pack(fill=tk.X)
        elif model == "manipulator":
            self.manip_bounds_frame.grid()
            self.eta_manip_frame.grid()
            self.start_vel_frame.pack(fill=tk.X)
            self.manip_ctrl_frame.pack(fill=tk.X)
            # Note: For manipulator, X=θ₁, Y=θ₂ for 2D visualization
            
    def _on_resize(self, event):
        """Handle window resize."""
        pass
    
    def _toggle_draw_mode(self):
        """Toggle region drawing mode."""
        if self.draw_region_var.get():
            self.click_to_set_var.set(False)  # Disable start position click
            self._log("Region drawing mode enabled. Click and drag on the grid to draw a region.")
        else:
            self._log("Region drawing mode disabled.")
    
    def _clear_regions(self):
        """Clear all regions from the text widget."""
        self.regions_text.delete("1.0", tk.END)
        self._draw_regions_preview()
        
    def _delete_last_region(self):
        """Delete the last region from the text widget."""
        text = self.regions_text.get("1.0", tk.END).strip()
        lines = [l for l in text.split('\n') if l.strip()]
        if lines:
            lines = lines[:-1]
            self.regions_text.delete("1.0", tk.END)
            self.regions_text.insert(tk.END, '\n'.join(lines))
            self._draw_regions_preview()
    
    def _get_next_region_name(self) -> str:
        """Get the next available region name."""
        # If user specified a name, use it
        name = self.new_region_name_var.get().strip()
        if name:
            return name
        
        # Otherwise auto-generate
        existing = set()
        text = self.regions_text.get("1.0", tk.END).strip()
        for line in text.split('\n'):
            if ':' in line:
                existing.add(line.split(':')[0].strip())
        
        # Try single letters first
        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if c not in existing:
                return c
        
        # Then try numbered regions
        i = 1
        while f"R{i}" in existing:
            i += 1
        return f"R{i}"
        
    def _on_plot_click(self, event):
        """Handle clicks on the plot."""
        if event.inaxes not in [self.ax_grid, self.ax_sim]:
            return
        
        # Region drawing mode
        if self.draw_region_var.get():
            if event.button == 1:  # Left click
                self.drawing_region = True
                self.draw_start_point = (event.xdata, event.ydata)
                return
        
        # Start position mode
        if self.click_to_set_var.get():
            self.start_x_var.set(f"{event.xdata:.2f}")
            self.start_y_var.set(f"{event.ydata:.2f}")
            self._draw_start_position()
    
    def _on_plot_release(self, event):
        """Handle mouse release on the plot."""
        if not self.drawing_region or self.draw_start_point is None:
            return
        
        if event.inaxes not in [self.ax_grid, self.ax_sim]:
            # Cancel drawing if released outside
            self.drawing_region = False
            self.draw_start_point = None
            self._remove_temp_rect()
            return
        
        # Complete the region
        x1, y1 = self.draw_start_point
        x2, y2 = event.xdata, event.ydata
        
        # Calculate bounds
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Check minimum size
        if abs(x_max - x_min) < 0.1 or abs(y_max - y_min) < 0.1:
            self._log("Region too small, ignoring.")
            self.drawing_region = False
            self.draw_start_point = None
            self._remove_temp_rect()
            return
        
        # Get region name
        name = self._get_next_region_name()
        
        # Add to regions text
        current_text = self.regions_text.get("1.0", tk.END).strip()
        if current_text:
            current_text += "\n"
        new_region = f"{name}: {x_min:.2f}, {x_max:.2f}, {y_min:.2f}, {y_max:.2f}"
        self.regions_text.delete("1.0", tk.END)
        self.regions_text.insert(tk.END, current_text + new_region)
        
        self._log(f"Added region {name}: [{x_min:.2f}, {x_max:.2f}] x [{y_min:.2f}, {y_max:.2f}]")
        
        # Clear the name entry for next region
        self.new_region_name_var.set("")
        
        # Reset drawing state
        self.drawing_region = False
        self.draw_start_point = None
        self._remove_temp_rect()
        
        # Redraw regions
        self._draw_regions_preview()
    
    def _on_plot_motion(self, event):
        """Handle mouse motion on the plot for drawing preview."""
        if not self.drawing_region or self.draw_start_point is None:
            return
        
        if event.inaxes not in [self.ax_grid, self.ax_sim]:
            return
        
        x1, y1 = self.draw_start_point
        x2, y2 = event.xdata, event.ydata
        
        # Update temporary rectangle
        self._remove_temp_rect()
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        width = x_max - x_min
        height = y_max - y_min
        
        self.temp_rect = Rectangle(
            (x_min, y_min), width, height,
            facecolor=to_rgba('yellow', 0.3),
            edgecolor='orange', linewidth=2, linestyle='--'
        )
        self.temp_rect.temp_rect = True
        self.ax_grid.add_patch(self.temp_rect)
        self.canvas.draw_idle()
    
    def _remove_temp_rect(self):
        """Remove temporary drawing rectangle."""
        if self.temp_rect is not None:
            try:
                self.temp_rect.remove()
            except:
                pass
            self.temp_rect = None
        
        # Also remove any patches marked as temp
        for ax in [self.ax_grid, self.ax_sim]:
            for artist in list(ax.patches):
                if hasattr(artist, 'temp_rect'):
                    try:
                        artist.remove()
                    except:
                        pass
        self.canvas.draw_idle()
    
    def _draw_regions_preview(self):
        """Draw regions preview on the grid (before synthesis)."""
        # Clear axes
        self.ax_grid.clear()
        self.ax_sim.clear()
        
        # Get bounds from input
        try:
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
        except ValueError:
            x_min, x_max = 0, 10
            y_min, y_max = 0, 10
        
        # Draw grid lines
        for ax in [self.ax_grid, self.ax_sim]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        self.ax_grid.set_title("Grid & Regions (Preview)")
        self.ax_sim.set_title("Simulation")
        
        # Parse and draw regions
        regions = self._parse_regions()
        colors = {'A': 'green', 'B': 'blue', 'O': 'red', 'C': 'orange', 'D': 'purple',
                  'E': 'cyan', 'F': 'magenta', 'G': 'yellow', 'H': 'brown'}
        
        for name, bounds in regions.items():
            color = colors.get(name, 'gray')
            alpha = 0.3 if name == 'O' else 0.4
            
            x_lo, x_hi, y_lo, y_hi = bounds[0], bounds[1], bounds[2], bounds[3]
            width = x_hi - x_lo
            height = y_hi - y_lo
            
            for ax in [self.ax_grid, self.ax_sim]:
                rect = Rectangle(
                    (x_lo, y_lo), width, height,
                    facecolor=to_rgba(color, alpha),
                    edgecolor=color, linewidth=2
                )
                ax.add_patch(rect)
                ax.text(x_lo + width/2, y_lo + height/2, name,
                       ha='center', va='center', fontsize=14, fontweight='bold')
        
        self.canvas.draw()
        
    def _draw_start_position(self):
        """Draw the start position marker."""
        try:
            x = float(self.start_x_var.get())
            y = float(self.start_y_var.get())
        except ValueError:
            return
            
        # Clear previous markers
        for ax in [self.ax_grid, self.ax_sim]:
            for artist in ax.get_children():
                if hasattr(artist, 'start_marker'):
                    artist.remove()
        
        # Draw new marker
        for ax in [self.ax_grid, self.ax_sim]:
            marker = ax.plot(x, y, 'g*', markersize=15, label='Start')[0]
            marker.start_marker = True
            
            if self.model_var.get() == "unicycle":
                try:
                    theta = float(self.start_theta_var.get())
                    dx = 0.5 * np.cos(theta)
                    dy = 0.5 * np.sin(theta)
                    arrow = ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1, 
                                    fc='green', ec='green')
                    arrow.start_marker = True
                except ValueError:
                    pass
                    
        self.canvas.draw()
        
    def _parse_regions(self) -> Dict[str, List[float]]:
        """Parse regions from text input."""
        regions = {}
        text = self.regions_text.get("1.0", tk.END).strip()
        
        for line in text.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            name, bounds_str = line.split(':', 1)
            name = name.strip()
            bounds = [float(x.strip()) for x in bounds_str.split(',')]
            regions[name] = bounds
            
        return regions
        
    def _log(self, message: str):
        """Log a message to the output panel."""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.update_idletasks()
        
    def _clear_output(self):
        """Clear the output log."""
        self.output_text.delete("1.0", tk.END)
        
    def _run_synthesis(self):
        """Run the synthesis pipeline in a separate thread."""
        self.run_btn.state(['disabled'])
        self.sim_btn.state(['disabled'])
        self.progress_var.set(0)
        self.progress_label.config(text="Starting...")
        self.time_label.config(text="")
        self.synthesis_start_time = None
        
        # Run in thread to keep UI responsive
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
    
    def _update_progress(self, current, total, message):
        """Update progress bar and label from worker thread."""
        import time
        progress = (current / total) * 100 if total > 0 else 0
        
        # Estimate time remaining
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
            
    def _do_synthesis(self):
        """Execute the synthesis pipeline."""
        import time
        from symbolic_control import (
            IntegratorDynamics, UnicycleDynamics, ManipulatorDynamics, Abstraction,
            ProductSynthesis
        )
        
        self.synthesis_start_time = time.time()
        
        self.root.after(0, lambda: self._log("=" * 50))
        self.root.after(0, lambda: self._log("Starting synthesis pipeline..."))
        self.root.after(0, lambda: self._log("=" * 50))
        
        # Parse parameters
        tau = float(self.tau_var.get())
        w_bound = float(self.w_bound_var.get())
        
        model = self.model_var.get()
        
        # Create dynamics using bounds and number of points
        if model == "integrator":
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
            eta_x = float(self.eta_x_var.get())
            eta_y = float(self.eta_y_var.get())
            
            u_min = float(self.u_min_var.get())
            u_max = float(self.u_max_var.get())
            u_num = int(self.u_num_var.get())
            u_values = list(np.linspace(u_min, u_max, num=u_num))
            
            self.dynamics = IntegratorDynamics(tau=tau, w_bound=w_bound, u_values=u_values)
            state_bounds = np.array([[x_min, x_max], [y_min, y_max]])
            eta = [eta_x, eta_y]
            self.root.after(0, lambda: self._log("Model: 2D Integrator"))
            self.root.after(0, lambda: self._log(f"u values: {u_values}"))
            
        elif model == "unicycle":
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
            theta_min = float(self.theta_min_var.get())
            theta_max = float(self.theta_max_var.get())
            eta_x = float(self.eta_x_var.get())
            eta_y = float(self.eta_y_var.get())
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
            self.root.after(0, lambda: self._log(f"v values: {v_values}"))
            self.root.after(0, lambda: self._log(f"ω values: {omega_values}"))
            
        elif model == "manipulator":
            # 4D state: [θ₁, θ₂, θ̇₁, θ̇₂]
            theta1_min = float(self.theta1_min_var.get())
            theta1_max = float(self.theta1_max_var.get())
            theta2_min = float(self.theta2_min_var.get())
            theta2_max = float(self.theta2_max_var.get())
            dtheta1_min = float(self.dtheta1_min_var.get())
            dtheta1_max = float(self.dtheta1_max_var.get())
            dtheta2_min = float(self.dtheta2_min_var.get())
            dtheta2_max = float(self.dtheta2_max_var.get())
            
            eta_theta1 = float(self.eta_theta1_var.get())
            eta_theta2 = float(self.eta_theta2_var.get())
            eta_dtheta1 = float(self.eta_dtheta1_var.get())
            eta_dtheta2 = float(self.eta_dtheta2_var.get())
            
            tau1_min = float(self.tau1_min_var.get())
            tau1_max = float(self.tau1_max_var.get())
            tau1_num = int(self.tau1_num_var.get())
            tau1_values = list(np.linspace(tau1_min, tau1_max, num=tau1_num))
            
            tau2_min = float(self.tau2_min_var.get())
            tau2_max = float(self.tau2_max_var.get())
            tau2_num = int(self.tau2_num_var.get())
            tau2_values = list(np.linspace(tau2_min, tau2_max, num=tau2_num))
            
            # For now, use the same torque values for both joints (simplified)
            # The ManipulatorDynamics will build all combinations
            torque_values = tau1_values  # Using tau1 values for both
            
            self.dynamics = ManipulatorDynamics(
                tau=tau, w_bound=w_bound,
                torque_values=torque_values
            )
            state_bounds = np.array([
                [theta1_min, theta1_max],
                [theta2_min, theta2_max],
                [dtheta1_min, dtheta1_max],
                [dtheta2_min, dtheta2_max]
            ])
            eta = [eta_theta1, eta_theta2, eta_dtheta1, eta_dtheta2]
            self.root.after(0, lambda: self._log("Model: 4D Manipulator"))
            self.root.after(0, lambda: self._log(f"τ values: {torque_values}"))
            
        self.root.after(0, lambda: self._log(f"State bounds: {state_bounds.tolist()}"))
        self.root.after(0, lambda: self._log(f"Eta: {eta}"))
        self.root.after(0, lambda: self._log(f"Control set size: {len(self.dynamics.control_set)}"))
        
        # Create abstraction
        self._update_progress(0, 100, "Creating abstraction...")
        self.root.after(0, lambda: self._log("\nCreating abstraction..."))
        
        self.abstraction = Abstraction(
            dynamics=self.dynamics,
            state_bounds=state_bounds,
            eta=eta
        )
        
        self.root.after(0, lambda: self._log(f"Grid shape: {self.abstraction.grid_shape}"))
        self.root.after(0, lambda: self._log(f"Total cells: {self.abstraction.num_cells}"))
        
        # Build transitions with progress callback
        self._update_progress(5, 100, "Building transitions...")
        self.root.after(0, lambda: self._log("\nBuilding transitions..."))
        
        transition_start = time.time()
        self.abstraction.build_transitions(progress_callback=self._transition_progress)
        transition_time = time.time() - transition_start
        self.root.after(0, lambda: self._log(f"Transitions built in {transition_time:.1f}s"))
        
        # Parse regions
        self._update_progress(60, 100, "Parsing regions...")
        self.regions = self._parse_regions()
        self.root.after(0, lambda: self._log(f"\nRegions: {list(self.regions.keys())}"))
        
        # Create and run synthesis
        spec = self.spec_var.get()
        self.root.after(0, lambda: self._log(f"Specification: {spec}"))
        
        self._update_progress(65, 100, "Building product automaton...")
        self.root.after(0, lambda: self._log("\nBuilding product automaton..."))
        
        self.synth = ProductSynthesis(
            abstraction=self.abstraction,
            regions=self.regions,
            spec=spec
        )
        
        # Redirect stdout to capture synthesis output
        old_stdout = sys.stdout
        sys.stdout = OutputRedirector(self.output_text)
        
        try:
            self._update_progress(70, 100, "Running synthesis...")
            self.synth.run(verbose=True)
        finally:
            sys.stdout = old_stdout
        
        winning_cells = self.synth.get_winning_cells()
        self.root.after(0, lambda: self._log(f"\nWinning cells: {len(winning_cells)}"))
        
        total_time = time.time() - self.synthesis_start_time
        self.root.after(0, lambda: self._log(f"\nTotal synthesis time: {total_time:.1f}s"))
        
        self._update_progress(100, 100, "Synthesis complete!")
        
        # Update visualization
        self.root.after(0, self._draw_grid_visualization)
    
    def _transition_progress(self, current, total, message):
        """Progress callback for transition building."""
        # Map to 5-60% of overall progress
        progress = 5 + (current / total) * 55 if total > 0 else 5
        self._update_progress(progress, 100, message)
        
    def _draw_grid_visualization(self):
        """Draw the grid abstraction and regions."""
        self.ax_grid.clear()
        self.ax_sim.clear()
        
        if self.abstraction is None:
            self.canvas.draw()
            return
            
        # Get bounds
        x_min, x_max = self.abstraction.state_bounds[0]
        y_min, y_max = self.abstraction.state_bounds[1]
        
        # Draw grid
        eta_x = self.abstraction.eta[0]
        eta_y = self.abstraction.eta[1]
        
        for x in np.arange(x_min, x_max + eta_x, eta_x):
            self.ax_grid.axvline(x, color='lightgray', linewidth=0.5)
            self.ax_sim.axvline(x, color='lightgray', linewidth=0.5)
        for y in np.arange(y_min, y_max + eta_y, eta_y):
            self.ax_grid.axhline(y, color='lightgray', linewidth=0.5)
            self.ax_sim.axhline(y, color='lightgray', linewidth=0.5)
            
        # Draw regions
        colors = {'A': 'green', 'B': 'blue', 'O': 'red', 'C': 'orange', 'D': 'purple'}
        for name, bounds in self.regions.items():
            color = colors.get(name, 'gray')
            alpha = 0.3 if name == 'O' else 0.4
            
            x_lo, x_hi, y_lo, y_hi = bounds[0], bounds[1], bounds[2], bounds[3]
            width = x_hi - x_lo
            height = y_hi - y_lo
            
            for ax in [self.ax_grid, self.ax_sim]:
                rect = Rectangle((x_lo, y_lo), width, height, 
                                 facecolor=to_rgba(color, alpha),
                                 edgecolor=color, linewidth=2,
                                 label=f"Region {name}")
                ax.add_patch(rect)
                ax.text(x_lo + width/2, y_lo + height/2, name, 
                       ha='center', va='center', fontsize=14, fontweight='bold')
                       
        # Draw winning cells (if available)
        if self.synth is not None:
            winning_cells = self.synth.get_winning_cells()
            for cell_idx in winning_cells:
                lo, hi = self.abstraction.cell_to_bounds(cell_idx)
                x_lo, y_lo = lo[0], lo[1]
                width = self.abstraction.eta[0]
                height = self.abstraction.eta[1]
                rect = Rectangle((x_lo, y_lo), width, height,
                                 facecolor=to_rgba('cyan', 0.2),
                                 edgecolor='none')
                self.ax_grid.add_patch(rect)
                
        # Set limits and labels based on model
        for ax in [self.ax_grid, self.ax_sim]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            if self.model_var.get() == "manipulator":
                ax.set_xlabel('θ₁ (rad)')
                ax.set_ylabel('θ₂ (rad)')
            else:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            ax.set_aspect('equal')
            
        if self.model_var.get() == "manipulator":
            self.ax_grid.set_title(f"Joint Space Grid (η=[{eta_x:.2f}, {eta_y:.2f}])")
        else:
            self.ax_grid.set_title(f"Grid Abstraction (η=[{eta_x:.2f}, {eta_y:.2f}])")
        self.ax_sim.set_title("Simulation")
        
        self._draw_start_position()
        self.canvas.draw()
        
    def _run_simulation(self):
        """Run simulation from the start position."""
        if self.synth is None:
            messagebox.showwarning("Warning", "Please run synthesis first")
            return
            
        try:
            start_x = float(self.start_x_var.get())
            start_y = float(self.start_y_var.get())
            
            if self.model_var.get() == "unicycle":
                start_theta = float(self.start_theta_var.get())
                start_pos = np.array([start_x, start_y, start_theta])
            elif self.model_var.get() == "manipulator":
                # For manipulator: x=θ₁, y=θ₂, and get velocity components
                start_theta1_dot = float(self.start_theta1_dot_var.get())
                start_theta2_dot = float(self.start_theta2_dot_var.get())
                start_pos = np.array([start_x, start_y, start_theta1_dot, start_theta2_dot])
            else:
                start_pos = np.array([start_x, start_y])
                
            max_steps = int(self.max_steps_var.get())
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid start position: {e}")
            return
            
        self._log(f"\nSimulating from {start_pos}...")
        
        try:
            trajectory, nfa_trace = self.synth.simulate(start_pos, max_steps=max_steps, verbose=True)
            self.trajectory = trajectory
            
            self._log(f"Trajectory length: {len(trajectory)}")
            
            # Check if reached accepting state
            if len(nfa_trace) > 0 and self.synth.is_accepting(nfa_trace[-1]):
                self._log("SUCCESS: Reached accepting state!")
            else:
                self._log("Did not reach accepting state")
                
            self._draw_simulation()
            
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {e}")
            self._log(f"ERROR: {e}")
            
    def _draw_simulation(self):
        """Draw the simulation trajectory."""
        if self.trajectory is None:
            return
            
        self.ax_sim.clear()
        
        # Redraw regions
        x_min, x_max = self.abstraction.state_bounds[0]
        y_min, y_max = self.abstraction.state_bounds[1]
        
        # Draw grid
        eta_x = self.abstraction.eta[0]
        eta_y = self.abstraction.eta[1]
        
        for x in np.arange(x_min, x_max + eta_x, eta_x):
            self.ax_sim.axvline(x, color='lightgray', linewidth=0.5)
        for y in np.arange(y_min, y_max + eta_y, eta_y):
            self.ax_sim.axhline(y, color='lightgray', linewidth=0.5)
            
        # Draw regions
        colors = {'A': 'green', 'B': 'blue', 'O': 'red', 'C': 'orange', 'D': 'purple'}
        for name, bounds in self.regions.items():
            color = colors.get(name, 'gray')
            alpha = 0.3 if name == 'O' else 0.4
            
            x_lo, x_hi, y_lo, y_hi = bounds[0], bounds[1], bounds[2], bounds[3]
            width = x_hi - x_lo
            height = y_hi - y_lo
            
            rect = Rectangle((x_lo, y_lo), width, height,
                             facecolor=to_rgba(color, alpha),
                             edgecolor=color, linewidth=2)
            self.ax_sim.add_patch(rect)
            self.ax_sim.text(x_lo + width/2, y_lo + height/2, name,
                            ha='center', va='center', fontsize=14, fontweight='bold')
                            
        # Draw trajectory
        traj = self.trajectory
        self.ax_sim.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        self.ax_sim.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
        self.ax_sim.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End')
        
        # Draw arrows for heading (unicycle)
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
        
        # Set axis labels based on model type
        if self.model_var.get() == "manipulator":
            self.ax_sim.set_xlabel('θ₁ (rad)')
            self.ax_sim.set_ylabel('θ₂ (rad)')
        else:
            self.ax_sim.set_xlabel('X')
            self.ax_sim.set_ylabel('Y')
        
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_title(f"Simulation ({len(traj)} steps)")
        self.ax_sim.legend(loc='upper right')
        
        self.canvas.draw()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SymbolicControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
