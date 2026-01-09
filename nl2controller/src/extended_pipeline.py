"""
Extended Pipeline with Darija Translation and Simulation

This module provides the complete "Objective 2" pipeline:
Darija Input → English Translation → LTL Formalization → Controller Synthesis → Simulation

This extends the base NL2ControllerPipeline with:
- Darija→English translation as the first stage
- Simulation and visualization of the synthesized controller
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .config import OpenAIConfig, PipelineConfig
from .pipeline import NL2ControllerPipeline, PipelineResult
from .translation import DarijaTranslator, TranslationResult
from .translation.translator import DemoTranslator
from .simulation import RobotSimulator, GridWorld, SimulationResult
from .simulation.visualizer import ControllerVisualizer, GridVisualizer
from .simulation.continuous_sim import ContinuousSimulator, SimulatorFactory
from .grounding.vocabulary import RobotVocabulary
from .synthesis.controller import Controller
from .synthesis.transition_system import TransitionSystem
from .lifting.engine import BaseLiftingEngine, LiftingResult, ExtractedEntity, MultiLiftingResult
from .lifting.patterns import get_pattern

logger = logging.getLogger(__name__)


class DemoLiftingEngine(BaseLiftingEngine):
    """Simple demo lifting engine that works without API."""

    def lift(self, nl_command: str) -> LiftingResult:
        """Extract pattern using simple rules."""
        cmd_lower = nl_command.lower()

        # Simple rule-based entity extraction
        # Look for common vocabulary words
        entity_text = "zone a"  # default fallback

        if "zone" in cmd_lower:
            if "a" in cmd_lower:
                entity_text = "zone a"
            elif "b" in cmd_lower:
                entity_text = "zone b"
            elif "c" in cmd_lower:
                entity_text = "zone c"
        elif any(word in cmd_lower for word in ["move", "go"]):
            entity_text = "move"
        elif "stop" in cmd_lower:
            entity_text = "stop"
        elif "patrol" in cmd_lower:
            entity_text = "patrol"
        elif "charge" in cmd_lower or "battery" in cmd_lower:
            entity_text = "charge"

        # Default: treat as "always_eventually" pattern
        pattern = get_pattern("always_eventually")
        entity = ExtractedEntity(text=entity_text, placeholder="p1")
        ltl = pattern.instantiate("p1")

        return LiftingResult(
            pattern=pattern,
            entities=[entity],
            abstract_ltl=ltl,
            confidence=0.8,
            original_command=nl_command
        )

    def lift_multi(self, nl_command: str):
        """Extract multiple patterns from complex command."""
        # For demo, just return single pattern wrapped in list
        result = self.lift(nl_command)
        return MultiLiftingResult(results=[result])


@dataclass
class FullPipelineResult:
    """
    Complete result of the extended Darija→Controller pipeline.
    
    Includes:
    - Translation result (Darija → English)
    - Pipeline result (English → Controller)
    - Simulation result (Controller execution)
    """
    # Input
    original_darija: str
    
    # Translation stage
    translation_result: Optional[TranslationResult] = None
    english_command: str = ""
    
    # NL2Controller pipeline stages
    pipeline_result: Optional[PipelineResult] = None
    abstract_ltl: str = ""
    grounded_ltl: str = ""
    
    # Controller
    controller: Optional[Controller] = None
    controller_dict: Optional[Dict[str, Any]] = None
    
    # Simulation
    simulation_result: Optional[SimulationResult] = None
    
    # Visualizations (as strings)
    controller_ascii: str = ""
    grid_visualization: str = ""
    controller_dot: str = ""
    
    # Status
    success: bool = False
    error_stage: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Timing
    translation_time: float = 0.0
    pipeline_time: float = 0.0
    simulation_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "original_darija": self.original_darija,
            "english_command": self.english_command,
            "abstract_ltl": self.abstract_ltl,
            "grounded_ltl": self.grounded_ltl,
            "controller": self.controller_dict,
            "simulation": {
                "success": self.simulation_result.success if self.simulation_result else None,
                "steps": self.simulation_result.steps if self.simulation_result else 0,
            } if self.simulation_result else None,
            "success": self.success,
            "error_stage": self.error_stage,
            "error_message": self.error_message,
            "timings": {
                "translation": self.translation_time,
                "pipeline": self.pipeline_time,
                "simulation": self.simulation_time,
                "total": self.total_time,
            }
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "FULL PIPELINE RESULT",
            "=" * 60,
            "",
            f"📝 Input (Darija): {self.original_darija}",
            f"🔄 Translation: {self.english_command}",
            f"📐 Abstract LTL: {self.abstract_ltl}",
            f"🎯 Grounded LTL: {self.grounded_ltl}",
            "",
        ]
        
        if self.success:
            lines.append("✅ Status: SUCCESS")
        else:
            lines.append(f"❌ Status: FAILED at {self.error_stage}")
            lines.append(f"   Error: {self.error_message}")
        
        lines.extend([
            "",
            f"⏱️ Timings:",
            f"   Translation: {self.translation_time:.3f}s",
            f"   Pipeline: {self.pipeline_time:.3f}s",
            f"   Simulation: {self.simulation_time:.3f}s",
            f"   Total: {self.total_time:.3f}s",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class DarijaPipeline:
    """
    Complete Darija-to-Controller Pipeline.
    
    Implements the full "Objective 2" flow:
    1. Translation: Darija → English (via LLM)
    2. Lifting: English → Abstract LTL
    3. Grounding: Abstract LTL → Grounded LTL
    4. Synthesis: Grounded LTL → Controller
    5. Simulation: Controller → Execution trace
    
    Example:
        from nl2controller.src.extended_pipeline import DarijaPipeline
        from nl2controller.src.config import OpenAIConfig
        
        config = OpenAIConfig(api_key="...", model="gpt-4o-mini")
        pipeline = DarijaPipeline(openai_config=config)
        
        result = pipeline.process("sir l lyamin u tjneb lhayt")
        print(result)
        print(result.grid_visualization)
    """
    
    def __init__(
        self,
        openai_config: Optional[OpenAIConfig] = None,
        config: Optional[PipelineConfig] = None,
        vocabulary_path: Optional[str] = None,
        vocabulary: Optional[RobotVocabulary] = None,
        lifting_engine: Optional[BaseLiftingEngine] = None,
        demo_mode: bool = False,
        grid_world: Optional[GridWorld] = None,
        use_continuous_dynamics: bool = True,
        model_type: str = "unicycle",
        disturbance_mode: str = "random",
    ):
        """
        Initialize the complete Darija pipeline.

        Args:
            openai_config: OpenAI API configuration
            config: Full pipeline configuration
            vocabulary_path: Path to robot vocabulary JSON
            vocabulary: Pre-loaded vocabulary
            lifting_engine: Custom lifting engine
            demo_mode: Use demo translator (no API calls for translation)
            grid_world: Custom grid world for simulation
            use_continuous_dynamics: Use continuous robot models (OBJECTIVE 2 integration)
            model_type: Robot model for continuous dynamics ("unicycle", "manipulator", "integrator")
            disturbance_mode: Disturbance handling ("random", "worst_case", "none")
        """
        # Determine OpenAI config
        if config and config.openai:
            openai_cfg = config.openai
        else:
            openai_cfg = openai_config
        
        # Initialize translator
        self.demo_mode = demo_mode
        if demo_mode:
            self.translator = DemoTranslator()
            logger.info("Using demo translator (no API)")
        elif openai_cfg:
            self.translator = DarijaTranslator.from_config(openai_cfg)
            logger.info("Using LLM translator")
        else:
            raise ValueError(
                "Either 'openai_config' or 'demo_mode=True' required for translation"
            )

        # Initialize base pipeline
        # In demo mode, provide a demo lifting engine
        if demo_mode and not lifting_engine:
            lifting_engine = DemoLiftingEngine()
            logger.info("Using demo lifting engine (no API)")

        self.base_pipeline = NL2ControllerPipeline(
            vocabulary_path=vocabulary_path,
            vocabulary=vocabulary,
            openai_config=openai_cfg if not demo_mode else None,
            config=config,
            lifting_engine=lifting_engine,
        )
        
        # Initialize simulation
        self.use_continuous_dynamics = use_continuous_dynamics
        self.model_type = model_type
        self.disturbance_mode = disturbance_mode

        if use_continuous_dynamics:
            # Use continuous dynamics (OBJECTIVE 2 integration)
            try:
                self.simulator = ContinuousSimulator(
                    model_type=model_type,
                    disturbance_mode=disturbance_mode,
                    enable_visualization=True
                )
                self.grid_world = None
                logger.info(f"✅ Using continuous dynamics: {model_type} model with {disturbance_mode} disturbances")
            except ImportError as e:
                logger.warning(f"⚠️  Continuous dynamics unavailable, falling back to grid-world: {e}")
                use_continuous_dynamics = False
                self.use_continuous_dynamics = False

        if not use_continuous_dynamics:
            # Fallback to grid-world
            self.grid_world = grid_world or self._create_default_world()
            self.simulator = RobotSimulator(self.grid_world)
            logger.info("Using grid-world simulation")

        # Initialize visualizers
        self.controller_viz = ControllerVisualizer()
        if not use_continuous_dynamics and self.grid_world:
            self.grid_viz = GridVisualizer(
                width=self.grid_world.width,
                height=self.grid_world.height
            )
        else:
            self.grid_viz = None
    
    def _create_default_world(self) -> GridWorld:
        """Create a default grid world for simulation."""
        world = GridWorld(width=8, height=8)
        
        # Add some obstacles
        obstacles = [(3, 3), (3, 4), (3, 5), (4, 3), (5, 3)]  # Wall
        for obs in obstacles:
            world.add_obstacle(*obs)
        
        # Add goals
        world.add_goal("goal", 7, 7)
        world.add_goal("zone_a", 6, 2)
        world.add_goal("zone_b", 2, 6)
        
        return world
    
    def process(
        self,
        darija_command: str,
        skip_simulation: bool = False,
        max_sim_steps: int = 100
    ) -> FullPipelineResult:
        """
        Process a Darija command through the complete pipeline.
        
        Args:
            darija_command: Command in Moroccan Arabic (Darija)
            skip_simulation: Skip simulation stage
            max_sim_steps: Maximum simulation steps
            
        Returns:
            FullPipelineResult with all stage outputs
        """
        total_start = time.time()
        result = FullPipelineResult(original_darija=darija_command)
        
        # Stage 1: Translation (Darija → English)
        logger.info(f"Stage 1 - Translation: '{darija_command}'")
        trans_start = time.time()
        
        try:
            translation = self.translator.translate(darija_command)
            result.translation_result = translation
            result.english_command = translation.english_translation
            result.translation_time = time.time() - trans_start
            logger.info(f"Translated to: '{result.english_command}'")
        except Exception as e:
            result.error_stage = "translation"
            result.error_message = str(e)
            result.total_time = time.time() - total_start
            logger.error(f"Translation failed: {e}")
            return result
        
        # Stages 2-4: NL2Controller pipeline (English → Controller)
        logger.info(f"Stages 2-4 - Pipeline: '{result.english_command}'")
        pipeline_start = time.time()
        
        try:
            pipeline_result = self.base_pipeline.process(result.english_command)
            result.pipeline_result = pipeline_result
            result.abstract_ltl = pipeline_result.abstract_ltl
            result.grounded_ltl = pipeline_result.grounded_ltl
            result.controller = pipeline_result.controller
            result.pipeline_time = time.time() - pipeline_start
            
            if not pipeline_result.success:
                result.error_stage = pipeline_result.error_stage
                result.error_message = pipeline_result.error_message
                result.total_time = time.time() - total_start
                return result
            
            # Convert controller to dict for visualization
            if result.controller:
                result.controller_dict = result.controller.to_dict()
                result.controller_ascii = self.controller_viz.to_ascii(result.controller_dict)
                result.controller_dot = self.controller_viz.to_dot(result.controller_dict)
            
        except Exception as e:
            result.error_stage = "pipeline"
            result.error_message = str(e)
            result.total_time = time.time() - total_start
            logger.error(f"Pipeline failed: {e}")
            return result
        
        # Stage 5: Simulation
        if not skip_simulation and result.controller_dict:
            logger.info("Stage 5 - Simulation")
            sim_start = time.time()

            try:
                if self.use_continuous_dynamics:
                    # Continuous simulator uses duration, not max_steps
                    sim_result = self.simulator.run_controller(
                        result.controller_dict,
                        duration=max_sim_steps * 0.1  # Convert steps to duration
                    )
                else:
                    # Grid simulator uses max_steps
                    sim_result = self.simulator.run_controller(
                        result.controller_dict,
                        max_steps=max_sim_steps
                    )
                result.simulation_result = sim_result
                result.simulation_time = time.time() - sim_start

                # Generate visualization
                if self.use_continuous_dynamics:
                    # Continuous dynamics visualization
                    try:
                        save_path = "objective2_continuous_result.png"
                        self.simulator.visualize(save_path=save_path)
                        result.grid_visualization = f"✅ Visualization saved: {save_path}"
                        logger.info(f"Continuous dynamics visualization saved: {save_path}")
                    except Exception as viz_error:
                        result.warnings.append(f"Visualization failed: {viz_error}")
                        logger.warning(f"Visualization failed: {viz_error}")
                else:
                    # Grid-world visualization
                    obstacles = list(self.grid_world.obstacles)
                    goals = self.grid_world.goals
                    result.grid_visualization = self.grid_viz.render_trace(
                        sim_result.trace,
                        obstacles=obstacles,
                        goals=goals
                    )
                
            except Exception as e:
                result.warnings.append(f"Simulation failed: {e}")
                logger.warning(f"Simulation failed: {e}")
        
        result.success = True
        result.total_time = time.time() - total_start
        
        return result
    
    def process_english(
        self,
        english_command: str,
        skip_simulation: bool = False,
        max_sim_steps: int = 100
    ) -> FullPipelineResult:
        """
        Process an English command (skip translation).
        
        Useful for testing or when input is already in English.
        
        Args:
            english_command: Command in English
            skip_simulation: Skip simulation
            max_sim_steps: Max simulation steps
            
        Returns:
            FullPipelineResult
        """
        result = FullPipelineResult(original_darija="[English input]")
        result.english_command = english_command
        result.translation_result = TranslationResult(
            original_darija="[N/A]",
            english_translation=english_command,
            confidence=1.0
        )
        
        # Run pipeline from English
        total_start = time.time()
        pipeline_start = time.time()
        
        try:
            pipeline_result = self.base_pipeline.process(english_command)
            result.pipeline_result = pipeline_result
            result.abstract_ltl = pipeline_result.abstract_ltl
            result.grounded_ltl = pipeline_result.grounded_ltl
            result.controller = pipeline_result.controller
            result.pipeline_time = time.time() - pipeline_start
            
            if not pipeline_result.success:
                result.error_stage = pipeline_result.error_stage
                result.error_message = pipeline_result.error_message
                result.total_time = time.time() - total_start
                return result
            
            if result.controller:
                result.controller_dict = result.controller.to_dict()
                result.controller_ascii = self.controller_viz.to_ascii(result.controller_dict)
                result.controller_dot = self.controller_viz.to_dot(result.controller_dict)
            
        except Exception as e:
            result.error_stage = "pipeline"
            result.error_message = str(e)
            result.total_time = time.time() - total_start
            return result
        
        # Simulation
        if not skip_simulation and result.controller_dict:
            sim_start = time.time()
            try:
                sim_result = self.simulator.run_controller(
                    result.controller_dict,
                    max_steps=max_sim_steps
                )
                result.simulation_result = sim_result
                result.simulation_time = time.time() - sim_start

                # Visualization
                if self.use_continuous_dynamics:
                    try:
                        save_path = "objective2_continuous_result.png"
                        self.simulator.visualize(save_path=save_path)
                        result.grid_visualization = f"✅ Visualization saved: {save_path}"
                    except Exception as viz_error:
                        result.warnings.append(f"Visualization: {viz_error}")
                else:
                    obstacles = list(self.grid_world.obstacles)
                    goals = self.grid_world.goals
                    result.grid_visualization = self.grid_viz.render_trace(
                        sim_result.trace,
                        obstacles=obstacles,
                        goals=goals
                    )
            except Exception as e:
                result.warnings.append(f"Simulation: {e}")
        
        result.success = True
        result.total_time = time.time() - total_start
        return result


def demo():
    """
    Demonstrate the complete Darija pipeline in demo mode.
    """
    print("=" * 60)
    print("DARIJA → CONTROLLER PIPELINE DEMO")
    print("=" * 60)
    print()
    
    # Demo mode doesn't require API
    # For production, use: DarijaPipeline(openai_config=config)
    
    # Create a minimal demo pipeline
    from .translation.translator import DemoTranslator
    
    translator = DemoTranslator()
    
    # Demo commands in Darija
    commands = [
        "sir l lyamin",  # go right
        "sir l qdam u tjneb lhayt",  # go forward and avoid the wall
        "dor l lisar u sir",  # turn left and go
        "wqef",  # stop
    ]
    
    print("Translation Examples:")
    print("-" * 40)
    
    for cmd in commands:
        result = translator.translate(cmd)
        print(f"  Darija: {cmd}")
        print(f"  English: {result.english_translation}")
        print(f"  Intent: {result.detected_intent}")
        print()
    
    print("-" * 40)
    print("Note: Full pipeline requires OpenAI API configuration.")
    print("Set OPENAI_API_KEY environment variable to use LLM features.")


if __name__ == "__main__":
    demo()
