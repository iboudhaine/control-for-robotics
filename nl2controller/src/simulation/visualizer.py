"""
Controller Visualizer

Provides visualization of synthesized controllers and execution traces.
Supports ASCII art, Graphviz DOT, and basic plotting.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    show_labels: bool = True
    show_actions: bool = True
    highlight_accepting: bool = True
    color_scheme: str = "default"


class ControllerVisualizer:
    """
    Visualizes synthesized controllers as state machines.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def to_ascii(self, controller: Dict[str, Any]) -> str:
        """
        Generate ASCII representation of controller.
        
        Args:
            controller: Synthesized controller dict
            
        Returns:
            ASCII string representation
        """
        states = controller.get("states", [])
        transitions = controller.get("transitions", [])
        
        # Handle states as dicts or strings
        if states and isinstance(states[0], dict):
            # Extract state names/ids from dicts
            initial_states = controller.get("initial_states", [])
            initial_ids = set(initial_states) if initial_states else set()
            accepting_states = controller.get("accepting_states", [])
            accepting_ids = set(accepting_states) if accepting_states else set()
        else:
            # Legacy format: states as strings
            initial = controller.get("initial_state", states[0] if states else "s0")
            initial_ids = {initial} if initial else set()
            accepting_states_list = controller.get("accepting_states", [])
            accepting_ids = set(accepting_states_list) if accepting_states_list else set()
        
        lines = [
            "╔══════════════════════════════════════╗",
            "║         CONTROLLER AUTOMATON         ║",
            "╠══════════════════════════════════════╣",
        ]
        
        # States
        lines.append("║ States:                              ║")
        for state in states:
            if isinstance(state, dict):
                state_id = state.get("id", state.get("name", "?"))
                state_name = state.get("name", f"s{state_id}")
                is_initial = state.get("is_initial", False) or state_id in initial_ids
                is_accepting = state_id in accepting_ids
                marker = "→" if is_initial else " "
                acc = "◉" if is_accepting else "○"
                display_name = f"{state_name} (id:{state_id})"
            else:
                # Legacy string format
                marker = "→" if state in initial_ids else " "
                acc = "◉" if state in accepting_ids else "○"
                display_name = str(state)
            lines.append(f"║   {marker} {acc} {display_name:<28} ║")
        
        lines.append("╠══════════════════════════════════════╣")
        lines.append("║ Transitions:                         ║")
        
        # Transitions
        for t in transitions:
            if isinstance(t, dict):
                src = t.get("from", t.get("source", "?"))
                tgt = t.get("to", t.get("target", "?"))
                action = t.get("action", t.get("label", ""))
                guard = t.get("guard", "")
            else:
                # Legacy format
                src = getattr(t, "source", "?")
                tgt = getattr(t, "target", "?")
                action = getattr(t, "action", "")
                guard = getattr(t, "guard", "")
            
            trans_str = f"{src} → {tgt}"
            if action:
                trans_str += f" [{action}]"
            if guard:
                trans_str += f" / {guard}"
            
            lines.append(f"║   {trans_str:<34} ║")
        
        lines.append("╚══════════════════════════════════════╝")
        
        # Add LTL specs if present
        specs = controller.get("ltl_specs", [])
        if specs:
            lines.append("")
            lines.append("LTL Specifications:")
            for spec in specs:
                lines.append(f"  • {spec}")
        
        return "\n".join(lines)
    
    def to_dot(self, controller: Dict[str, Any]) -> str:
        """
        Generate Graphviz DOT representation.
        
        Args:
            controller: Synthesized controller dict
            
        Returns:
            DOT format string
        """
        states = controller.get("states", [])
        transitions = controller.get("transitions", [])
        initial = controller.get("initial_state", states[0] if states else "s0")
        accepting = set(controller.get("accepting_states", []))
        
        lines = [
            "digraph Controller {",
            "    rankdir=LR;",
            "    node [shape=circle];",
            "",
        ]
        
        # Mark accepting states
        if accepting:
            acc_str = " ".join(f'"{s}"' for s in accepting)
            lines.append(f"    node [shape=doublecircle]; {acc_str};")
            lines.append("    node [shape=circle];")
            lines.append("")
        
        # Initial state marker
        lines.append('    __start__ [shape=point];')
        lines.append(f'    __start__ -> "{initial}";')
        lines.append("")
        
        # Transitions
        for t in transitions:
            src = t.get("from", t.get("source", "?"))
            tgt = t.get("to", t.get("target", "?"))
            action = t.get("action", t.get("label", ""))
            guard = t.get("guard", "")
            
            label_parts = []
            if guard:
                label_parts.append(guard)
            if action:
                label_parts.append(action)
            
            label = " / ".join(label_parts) if label_parts else ""
            
            lines.append(f'    "{src}" -> "{tgt}" [label="{label}"];')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def to_mermaid(self, controller: Dict[str, Any]) -> str:
        """
        Generate Mermaid diagram representation.
        
        Args:
            controller: Synthesized controller dict
            
        Returns:
            Mermaid format string (for markdown rendering)
        """
        states = controller.get("states", [])
        transitions = controller.get("transitions", [])
        initial = controller.get("initial_state", states[0] if states else "s0")
        accepting = set(controller.get("accepting_states", []))
        
        lines = [
            "```mermaid",
            "stateDiagram-v2",
        ]
        
        # Initial state
        lines.append(f"    [*] --> {initial}")
        
        # Transitions
        for t in transitions:
            src = t.get("from", t.get("source", "?"))
            tgt = t.get("to", t.get("target", "?"))
            action = t.get("action", t.get("label", ""))
            
            if action:
                lines.append(f"    {src} --> {tgt}: {action}")
            else:
                lines.append(f"    {src} --> {tgt}")
        
        # Accepting states
        for state in accepting:
            lines.append(f"    {state} --> [*]")
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def print_summary(self, controller: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the controller.
        
        Args:
            controller: Synthesized controller dict
            
        Returns:
            Summary string
        """
        states = controller.get("states", [])
        transitions = controller.get("transitions", [])
        accepting = controller.get("accepting_states", [])
        realizable = controller.get("realizable", True)
        specs = controller.get("ltl_specs", [])
        
        lines = [
            "=" * 50,
            "CONTROLLER SUMMARY",
            "=" * 50,
            "",
            f"Status: {'✅ Realizable' if realizable else '❌ Unrealizable'}",
            f"States: {len(states)}",
            f"Transitions: {len(transitions)}",
            f"Accepting states: {len(accepting)}",
            "",
        ]
        
        if specs:
            lines.append("LTL Specifications:")
            for spec in specs:
                lines.append(f"  φ: {spec}")
            lines.append("")
        
        if transitions:
            lines.append("Actions used:")
            actions = set()
            for t in transitions:
                action = t.get("action", t.get("label"))
                if action:
                    actions.add(action)
            for action in sorted(actions):
                lines.append(f"  • {action}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


class GridVisualizer:
    """
    Visualizes robot movement on a grid.
    """
    
    # Box drawing characters
    BOX_CHARS = {
        "horizontal": "─",
        "vertical": "│",
        "top_left": "┌",
        "top_right": "┐",
        "bottom_left": "└",
        "bottom_right": "┘",
        "t_down": "┬",
        "t_up": "┴",
        "t_right": "├",
        "t_left": "┤",
        "cross": "┼"
    }
    
    # Direction symbols
    DIRECTION_ARROWS = {
        "NORTH": "△",
        "SOUTH": "▽",
        "EAST": "▷",
        "WEST": "◁"
    }
    
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
    
    def render_trace(
        self,
        trace: List[Dict[str, Any]],
        obstacles: Optional[List[Tuple[int, int]]] = None,
        goals: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> str:
        """
        Render a movement trace on the grid.
        
        Args:
            trace: List of trace entries with position and direction
            obstacles: List of obstacle positions
            goals: Dict of goal names to positions
            
        Returns:
            ASCII grid visualization
        """
        obstacles = set(obstacles or [])
        goals = goals or {}
        
        # Extract path from trace
        path = []
        for entry in trace:
            pos = entry.get("position")
            if pos:
                path.append(tuple(pos))
        
        # Get final position and direction
        final_pos = path[-1] if path else (0, 0)
        final_dir = trace[-1].get("direction", "NORTH") if trace else "NORTH"
        
        lines = []
        
        # Top border
        lines.append("┌" + "─" * (self.width * 2 + 1) + "┐")
        
        for y in range(self.height - 1, -1, -1):
            row = "│ "
            for x in range(self.width):
                pos = (x, y)
                
                if pos == final_pos:
                    # Robot current position
                    row += self.DIRECTION_ARROWS.get(final_dir, "◯")
                elif pos in obstacles:
                    row += "█"
                elif pos in goals.values():
                    # Find goal name
                    for name, gpos in goals.items():
                        if gpos == pos:
                            row += name[0].upper()
                            break
                elif pos in path:
                    # Path trace
                    row += "·"
                else:
                    row += " "
                row += " "
            
            row += "│"
            lines.append(row)
        
        # Bottom border
        lines.append("└" + "─" * (self.width * 2 + 1) + "┘")
        
        # Legend
        lines.append("")
        lines.append("Legend:")
        lines.append(f"  {self.DIRECTION_ARROWS['NORTH']} Robot (facing north)")
        lines.append("  █ Obstacle")
        lines.append("  · Path taken")
        if goals:
            for name in goals:
                lines.append(f"  {name[0].upper()} Goal: {name}")
        
        return "\n".join(lines)
    
    def render_animation_frames(
        self,
        trace: List[Dict[str, Any]],
        obstacles: Optional[List[Tuple[int, int]]] = None,
        goals: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> List[str]:
        """
        Generate animation frames for each step.
        
        Args:
            trace: List of trace entries
            obstacles: Obstacle positions
            goals: Goal positions
            
        Returns:
            List of frame strings
        """
        obstacles = set(obstacles or [])
        goals = goals or {}
        
        frames = []
        path_so_far = []
        
        for i, entry in enumerate(trace):
            pos = tuple(entry.get("position", (0, 0)))
            direction = entry.get("direction", "NORTH")
            action = entry.get("action", "")
            
            path_so_far.append(pos)
            
            lines = [f"Step {entry.get('step', i)}: {action}", ""]
            
            # Grid
            lines.append("┌" + "─" * (self.width * 2 + 1) + "┐")
            
            for y in range(self.height - 1, -1, -1):
                row = "│ "
                for x in range(self.width):
                    cell_pos = (x, y)
                    
                    if cell_pos == pos:
                        row += self.DIRECTION_ARROWS.get(direction, "◯")
                    elif cell_pos in obstacles:
                        row += "█"
                    elif cell_pos in goals.values():
                        for name, gpos in goals.items():
                            if gpos == cell_pos:
                                row += name[0].upper()
                                break
                    elif cell_pos in path_so_far[:-1]:
                        row += "·"
                    else:
                        row += " "
                    row += " "
                
                row += "│"
                lines.append(row)
            
            lines.append("└" + "─" * (self.width * 2 + 1) + "┘")
            
            frames.append("\n".join(lines))
        
        return frames
    
    def render_ltl_trace(
        self,
        trace: List[Dict[str, Any]]
    ) -> str:
        """
        Render LTL proposition trace over time.
        
        Args:
            trace: Execution trace with propositions
            
        Returns:
            Text visualization of proposition changes
        """
        lines = ["LTL Proposition Trace:", ""]
        lines.append("Step | Position | Props")
        lines.append("-" * 40)
        
        for entry in trace:
            step = entry.get("step", "?")
            pos = entry.get("position", (0, 0))
            props = entry.get("props", [])
            
            props_str = ", ".join(props) if props else "∅"
            lines.append(f"{step:4} | {str(pos):8} | {props_str}")
        
        return "\n".join(lines)


def demo_visualization():
    """Demonstrate visualization capabilities."""
    
    # Sample controller
    controller = {
        "states": ["idle", "moving", "avoiding", "goal"],
        "initial_state": "idle",
        "accepting_states": ["goal"],
        "transitions": [
            {"from": "idle", "to": "moving", "action": "move_forward"},
            {"from": "moving", "to": "moving", "action": "move_forward", "guard": "!obstacle"},
            {"from": "moving", "to": "avoiding", "action": "turn_left", "guard": "obstacle"},
            {"from": "avoiding", "to": "moving", "action": "move_forward"},
            {"from": "moving", "to": "goal", "action": "stop", "guard": "at_goal"},
        ],
        "ltl_specs": [
            "G(!obstacle → X moving)",
            "G(obstacle → F avoiding)",
            "F goal"
        ],
        "realizable": True
    }
    
    # Sample trace
    trace = [
        {"step": 0, "action": "init", "position": (0, 0), "direction": "NORTH", "props": []},
        {"step": 1, "action": "move_forward", "position": (0, 1), "direction": "NORTH", "props": []},
        {"step": 2, "action": "move_forward", "position": (0, 2), "direction": "NORTH", "props": []},
        {"step": 3, "action": "turn_right", "position": (0, 2), "direction": "EAST", "props": []},
        {"step": 4, "action": "move_forward", "position": (1, 2), "direction": "EAST", "props": []},
        {"step": 5, "action": "move_forward", "position": (2, 2), "direction": "EAST", "props": ["at_goal"]},
    ]
    
    # Create visualizers
    ctrl_viz = ControllerVisualizer()
    grid_viz = GridVisualizer(width=5, height=5)
    
    print(ctrl_viz.to_ascii(controller))
    print()
    print(ctrl_viz.print_summary(controller))
    print()
    print(grid_viz.render_trace(trace, goals={"goal": (2, 2)}))
    print()
    print(grid_viz.render_ltl_trace(trace))


if __name__ == "__main__":
    demo_visualization()
