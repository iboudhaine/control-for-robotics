"""
NL2Controller Pipeline - Main Orchestrator

This module provides the main entry point for the NLP-to-Controller
pipeline. It orchestrates the three stages:
1. Lifting: NL to abstract LTL patterns
2. Grounding: Abstract LTL to grounded LTL with robot variables
3. Synthesis: Grounded LTL to reactive controller

The pipeline ensures SOUNDNESS through:
- Constrained pattern templates (no LTL hallucination)
- Closed dictionary grounding (no variable hallucination)
- Formal GR(1) synthesis (correct-by-construction)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

from .lifting.engine import LiftingEngine, LiftingResult, MultiLiftingResult, BaseLiftingEngine
from .grounding.vocabulary import RobotVocabulary
from .grounding.grounding_filter import (
    DictionaryGroundingFilter, 
    GroundingResult,
    GroundingError,
    MultiGroundingResult
)
from .synthesis.synthesizer import Synthesizer, SynthesisResult, SynthesisError
from .synthesis.controller import Controller
from .synthesis.transition_system import TransitionSystem
from .config import PipelineConfig, OpenAIConfig


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Complete result of the NL2Controller pipeline.
    
    Contains all intermediate results and the final controller.
    """
    # Input
    original_command: str
    
    # Stage 1 results
    lifting_result: Optional[LiftingResult] = None
    abstract_ltl: str = ""
    
    # Stage 2 results
    grounding_result: Optional[GroundingResult] = None
    grounded_ltl: str = ""
    
    # Stage 3 results
    synthesis_result: Optional[SynthesisResult] = None
    controller: Optional[Controller] = None
    
    # Status
    success: bool = False
    error_stage: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Export result to dictionary."""
        return {
            "original_command": self.original_command,
            "abstract_ltl": self.abstract_ltl,
            "grounded_ltl": self.grounded_ltl,
            "controller": self.controller.to_dict() if self.controller else None,
            "success": self.success,
            "error_stage": self.error_stage,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "total_time": self.total_time,
        }


class NL2ControllerPipeline:
    """
    Main pipeline for converting natural language to controllers.
    
    This class orchestrates the three-stage pipeline:
    1. Lifting (NL → Abstract LTL)
    2. Grounding (Abstract LTL → Grounded LTL)
    3. Synthesis (Grounded LTL → Controller)
    
    Example:
        from nl2controller.src.config import OpenAIConfig
        
        config = OpenAIConfig(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini"
        )
        pipeline = NL2ControllerPipeline(
            vocabulary_path="data/robot_vocabulary.json",
            openai_config=config
        )
        result = pipeline.process("Patrol Zone A and avoid Zone C")
        if result.success:
            controller = result.controller
    """
    
    def __init__(
        self,
        vocabulary_path: Optional[str] = None,
        vocabulary: Optional[RobotVocabulary] = None,
        openai_config: Optional[OpenAIConfig] = None,
        config: Optional[PipelineConfig] = None,
        lifting_engine: Optional[BaseLiftingEngine] = None,
        strict_grounding: bool = True,
    ):
        """
        Initialize the pipeline.
        
        Args:
            vocabulary_path: Path to robot vocabulary JSON file
            vocabulary: Pre-loaded RobotVocabulary (alternative to path)
            openai_config: OpenAI API configuration for LLM lifting
            config: Full pipeline configuration (overrides openai_config)
            lifting_engine: Pre-configured lifting engine (for advanced use/testing)
            strict_grounding: Fail on unknown entities (recommended for safety)
        """
        # Initialize vocabulary
        if vocabulary:
            self.vocabulary = vocabulary
        elif vocabulary_path:
            self.vocabulary = RobotVocabulary(vocabulary_path)
        else:
            # Try default path
            default_path = Path(__file__).parent.parent / "data" / "robot_vocabulary.json"
            if default_path.exists():
                self.vocabulary = RobotVocabulary(str(default_path))
            else:
                self.vocabulary = RobotVocabulary()
                logger.warning("No vocabulary provided - grounding may fail")
        
        # Determine OpenAI config
        if config and config.openai:
            openai_cfg = config.openai
        else:
            openai_cfg = openai_config
        
        # Initialize lifting engine
        if lifting_engine:
            self.lifting_engine = lifting_engine
            logger.info("Using provided lifting engine")
        elif openai_cfg:
            self.lifting_engine = LiftingEngine.from_config(openai_cfg)
            logger.info(f"Using LLM lifting engine: {openai_cfg.model} @ {openai_cfg.base_url}")
        else:
            raise ValueError(
                "Either 'openai_config', 'config.openai', or 'lifting_engine' must be provided. "
                "Example: NL2ControllerPipeline(openai_config=OpenAIConfig(api_key='...'))"
            )
        
        # Initialize grounding filter
        self.grounding_filter = DictionaryGroundingFilter(
            vocabulary=self.vocabulary,
            strict_mode=strict_grounding
        )
        
        # Initialize synthesizer (uses TuLiP if available, otherwise basic)
        synthesis_config = config.synthesis if config else None
        self.synthesizer = Synthesizer(
            timeout=synthesis_config.timeout if synthesis_config else 60.0,
            verbose=synthesis_config.verbose if synthesis_config else False,
            solver=synthesis_config.solver if synthesis_config else "omega"
        )
        logger.info(f"Using synthesizer: TuLiP available = {self.synthesizer.tulip_available}")
    
    def process(
        self, 
        nl_command: str,
        transition_system: Optional[TransitionSystem] = None
    ) -> PipelineResult:
        """
        Process a natural language command through the full pipeline.
        
        Args:
            nl_command: Natural language specification
            transition_system: Optional robot transition system
            
        Returns:
            PipelineResult with controller (if successful) or error info
        """
        import time
        start_time = time.time()
        
        result = PipelineResult(original_command=nl_command)
        
        # Stage 1: Lifting
        logger.info(f"Stage 1 - Lifting: '{nl_command}'")
        try:
            lifting_result = self.lifting_engine.lift(nl_command)
            result.lifting_result = lifting_result
            result.abstract_ltl = lifting_result.abstract_ltl
            logger.info(f"  Pattern: {lifting_result.pattern.name}")
            logger.info(f"  Abstract LTL: {lifting_result.abstract_ltl}")
        except Exception as e:
            result.error_stage = "lifting"
            result.error_message = str(e)
            logger.error(f"Lifting failed: {e}")
            return result
        
        # Stage 2: Grounding
        logger.info("Stage 2 - Grounding")
        try:
            grounding_result = self.grounding_filter.ground(lifting_result)
            result.grounding_result = grounding_result
            result.grounded_ltl = grounding_result.grounded_ltl
            logger.info(f"  Grounded LTL: {grounding_result.grounded_ltl}")
        except GroundingError as e:
            result.error_stage = "grounding"
            result.error_message = str(e)
            result.warnings.append(f"Suggestions: {e.suggestions}")
            logger.error(f"Grounding failed: {e}")
            logger.error(f"  Ungrounded entities: {e.ungrounded_entities}")
            return result
        except Exception as e:
            result.error_stage = "grounding"
            result.error_message = str(e)
            logger.error(f"Grounding failed: {e}")
            return result
        
        # Stage 3: Synthesis
        logger.info("Stage 3 - Synthesis")
        try:
            synthesis_result = self.synthesizer.synthesize(
                grounding_result, 
                transition_system
            )
            result.synthesis_result = synthesis_result
            result.controller = synthesis_result.controller
            result.warnings.extend(synthesis_result.warnings)
            logger.info(f"  Controller: {synthesis_result.controller}")
            logger.info(f"  Synthesis time: {synthesis_result.synthesis_time:.3f}s")
        except SynthesisError as e:
            result.error_stage = "synthesis"
            result.error_message = str(e)
            if e.is_unrealizable:
                result.warnings.append("Specification is UNREALIZABLE")
            logger.error(f"Synthesis failed: {e}")
            return result
        except Exception as e:
            result.error_stage = "synthesis"
            result.error_message = str(e)
            logger.error(f"Synthesis failed: {e}")
            return result
        
        # Success
        result.success = True
        result.total_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {result.total_time:.3f}s")
        
        return result
    
    def process_multi(
        self, 
        nl_command: str,
        transition_system: Optional[TransitionSystem] = None
    ) -> PipelineResult:
        """
        Process a complex command that may contain multiple specifications.
        
        Args:
            nl_command: Natural language specification (may have multiple clauses)
            transition_system: Optional robot transition system
            
        Returns:
            PipelineResult with combined controller
        """
        import time
        start_time = time.time()
        
        result = PipelineResult(original_command=nl_command)
        
        # Stage 1: Multi-Lifting
        logger.info(f"Stage 1 - Multi-Lifting: '{nl_command}'")
        try:
            multi_lifting = self.lifting_engine.lift_multi(nl_command)
            if len(multi_lifting.results) == 1:
                result.lifting_result = multi_lifting.results[0]
            result.abstract_ltl = multi_lifting.combined_ltl
            logger.info(f"  Extracted {len(multi_lifting.results)} pattern(s)")
            logger.info(f"  Combined Abstract LTL: {multi_lifting.combined_ltl}")
        except Exception as e:
            result.error_stage = "lifting"
            result.error_message = str(e)
            logger.error(f"Multi-lifting failed: {e}")
            return result
        
        # Stage 2: Multi-Grounding
        logger.info("Stage 2 - Multi-Grounding")
        try:
            multi_grounding = self.grounding_filter.ground_multi(multi_lifting)
            if len(multi_grounding.results) == 1:
                result.grounding_result = multi_grounding.results[0]
            result.grounded_ltl = multi_grounding.combined_grounded_ltl
            logger.info(f"  Combined Grounded LTL: {multi_grounding.combined_grounded_ltl}")
        except GroundingError as e:
            result.error_stage = "grounding"
            result.error_message = str(e)
            result.warnings.append(f"Suggestions: {e.suggestions}")
            logger.error(f"Multi-grounding failed: {e}")
            return result
        
        # Stage 3: Synthesis (on combined formula)
        logger.info("Stage 3 - Synthesis")
        try:
            # Create a combined grounding result for synthesis
            combined_grounding = GroundingResult(
                grounded_ltl=multi_grounding.combined_grounded_ltl,
                grounded_entities=[
                    e for gr in multi_grounding.results 
                    for e in gr.grounded_entities
                ],
                original_abstract_ltl=multi_lifting.combined_ltl,
                original_command=nl_command
            )
            
            synthesis_result = self.synthesizer.synthesize(
                combined_grounding, 
                transition_system
            )
            result.synthesis_result = synthesis_result
            result.controller = synthesis_result.controller
            result.warnings.extend(synthesis_result.warnings)
            logger.info(f"  Controller: {synthesis_result.controller}")
        except SynthesisError as e:
            result.error_stage = "synthesis"
            result.error_message = str(e)
            logger.error(f"Synthesis failed: {e}")
            return result
        
        result.success = True
        result.total_time = time.time() - start_time
        logger.info(f"Multi-pipeline completed in {result.total_time:.3f}s")
        
        return result
    
    def validate_command(self, nl_command: str) -> Dict[str, Any]:
        """
        Validate a command without full synthesis.
        
        Useful for checking if a command can be processed before
        expensive synthesis.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "command": nl_command,
            "lifting_ok": False,
            "grounding_ok": False,
            "pattern": None,
            "abstract_ltl": None,
            "grounded_ltl": None,
            "ungrounded_entities": [],
            "suggestions": {},
        }
        
        # Try lifting
        try:
            lift_result = self.lifting_engine.lift(nl_command)
            validation["lifting_ok"] = True
            validation["pattern"] = lift_result.pattern.name
            validation["abstract_ltl"] = lift_result.abstract_ltl
            
            # Try grounding
            try:
                ground_result = self.grounding_filter.ground(lift_result)
                validation["grounding_ok"] = True
                validation["grounded_ltl"] = ground_result.grounded_ltl
            except GroundingError as e:
                validation["ungrounded_entities"] = e.ungrounded_entities
                validation["suggestions"] = e.suggestions
                
        except Exception as e:
            validation["error"] = str(e)
        
        return validation
    
    def get_available_phrases(self) -> List[str]:
        """Get all phrases known to the vocabulary."""
        return self.vocabulary.get_all_phrases()
    
    def get_vocabulary_categories(self) -> List[str]:
        """Get vocabulary categories."""
        return self.vocabulary.get_categories()
    
    def add_vocabulary_entry(
        self, 
        phrase: str, 
        system_var: str, 
        category: str = "custom"
    ) -> None:
        """
        Add a new entry to the vocabulary at runtime.
        
        Args:
            phrase: Natural language phrase
            system_var: System variable expression
            category: Category for the entry
        """
        self.vocabulary._phrase_to_var[phrase.lower()] = system_var
        logger.info(f"Added vocabulary entry: '{phrase}' -> '{system_var}'")


def create_pipeline(
    openai_api_key: str,
    vocabulary_path: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini"
) -> NL2ControllerPipeline:
    """
    Convenience function to create a pipeline.
    
    Args:
        openai_api_key: API key for LLM service
        vocabulary_path: Path to vocabulary JSON
        base_url: API endpoint (OpenAI, Ollama, Azure, etc.)
        model: Model name
        
    Returns:
        Configured NL2ControllerPipeline
    """
    from .config import OpenAIConfig
    
    config = OpenAIConfig(
        api_key=openai_api_key,
        base_url=base_url,
        model=model
    )
    
    return NL2ControllerPipeline(
        vocabulary_path=vocabulary_path,
        openai_config=config,
        strict_grounding=True
    )
