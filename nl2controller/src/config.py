"""
NL2Controller Configuration Module

Handles all configuration settings for the pipeline.
Supports environment variables, .env files, and programmatic configuration.

Configuration can be set via:
1. Environment variables (OPENAI_API_KEY, OPENAI_BASE_URL, etc.)
2. .env file in the project root
3. Programmatic configuration via create_config()
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class OpenAIConfig:
    """
    Configuration for OpenAI-compatible API.
    
    Supports any OpenAI-compatible endpoint:
    - OpenAI (default): https://api.openai.com/v1
    - Azure OpenAI: https://<resource>.openai.azure.com/
    - Local LLMs (Ollama, vLLM, etc.): http://localhost:8000/v1
    - Other providers (Together, Anyscale, etc.)
    """
    
    # API Key (required for most providers)
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    
    # Base URL - can be changed for different providers
    base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    
    # Model to use
    model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    
    # Temperature for generation (0 = deterministic)
    temperature: float = field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    )
    
    # Max tokens for response
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    )
    
    # Timeout in seconds
    timeout: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT", "30"))
    )
    
    # Organization ID (optional, for OpenAI)
    organization: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_ORG_ID")
    )
    
    def is_configured(self) -> bool:
        """Check if API key is provided."""
        return bool(self.api_key and self.api_key.strip())
    
    def validate(self) -> None:
        """
        Validate configuration, raise if invalid.
        
        Raises:
            ValueError: If API key is not configured
        """
        if not self.is_configured():
            raise ValueError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter.\n"
                "You can also create a .env file with:\n"
                "  OPENAI_API_KEY=your-api-key-here\n"
                "  OPENAI_BASE_URL=https://api.openai.com/v1  # optional\n"
                "  OPENAI_MODEL=gpt-4o-mini  # optional"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary (hides API key for security)."""
        return {
            "api_key": "***" if self.api_key else "",
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }


@dataclass
class GroundingConfig:
    """Configuration for the Grounding stage."""
    
    # Path to the robot vocabulary dictionary
    vocabulary_path: str = field(
        default_factory=lambda: os.getenv(
            "NL2C_VOCABULARY_PATH",
            str(Path(__file__).parent.parent / "data" / "robot_vocabulary.json")
        )
    )
    
    # Strict mode: fail if entity cannot be grounded (RECOMMENDED for safety)
    strict_mode: bool = field(
        default_factory=lambda: os.getenv("NL2C_STRICT_GROUNDING", "true").lower() == "true"
    )
    
    # Fuzzy matching threshold (only used if strict_mode is False)
    fuzzy_threshold: float = field(
        default_factory=lambda: float(os.getenv("NL2C_FUZZY_THRESHOLD", "0.6"))
    )


@dataclass
class SynthesisConfig:
    """Configuration for the Synthesis stage (TuLiP/GR1)."""
    
    # GR(1) solver backend
    solver: str = field(
        default_factory=lambda: os.getenv("NL2C_SOLVER", "gr1c")
    )
    
    # Timeout for synthesis (seconds)
    timeout: int = field(
        default_factory=lambda: int(os.getenv("NL2C_SYNTHESIS_TIMEOUT", "60"))
    )
    
    # Verbose output from solver
    verbose: bool = field(
        default_factory=lambda: os.getenv("NL2C_VERBOSE", "false").lower() == "true"
    )


@dataclass
class PipelineConfig:
    """
    Main configuration for the entire NL2Controller pipeline.
    
    Example usage:
        # From environment variables
        config = PipelineConfig()
        
        # Programmatic configuration
        config = create_config(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            model="gpt-4o"
        )
    """
    
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    
    # Logging settings
    log_level: str = field(
        default_factory=lambda: os.getenv("NL2C_LOG_LEVEL", "INFO")
    )
    log_file: Optional[str] = field(
        default_factory=lambda: os.getenv("NL2C_LOG_FILE")
    )
    
    # Project paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent
    )
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.openai.validate()
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            PipelineConfig instance
        """
        openai_cfg = config_dict.get("openai", {})
        grounding_cfg = config_dict.get("grounding", {})
        synthesis_cfg = config_dict.get("synthesis", {})
        
        return cls(
            openai=OpenAIConfig(
                api_key=openai_cfg.get("api_key", os.getenv("OPENAI_API_KEY", "")),
                base_url=openai_cfg.get("base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
                model=openai_cfg.get("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
                temperature=openai_cfg.get("temperature", 0.0),
                max_tokens=openai_cfg.get("max_tokens", 1000),
                timeout=openai_cfg.get("timeout", 30),
            ),
            grounding=GroundingConfig(
                vocabulary_path=grounding_cfg.get("vocabulary_path", ""),
                strict_mode=grounding_cfg.get("strict_mode", True),
                fuzzy_threshold=grounding_cfg.get("fuzzy_threshold", 0.6),
            ),
            synthesis=SynthesisConfig(
                solver=synthesis_cfg.get("solver", "gr1c"),
                timeout=synthesis_cfg.get("timeout", 60),
                verbose=synthesis_cfg.get("verbose", False),
            ),
            log_level=config_dict.get("log_level", "INFO"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "openai": self.openai.to_dict(),
            "grounding": {
                "vocabulary_path": self.grounding.vocabulary_path,
                "strict_mode": self.grounding.strict_mode,
                "fuzzy_threshold": self.grounding.fuzzy_threshold,
            },
            "synthesis": {
                "solver": self.synthesis.solver,
                "timeout": self.synthesis.timeout,
                "verbose": self.synthesis.verbose,
            },
            "log_level": self.log_level,
        }


def get_default_config() -> PipelineConfig:
    """Get the default pipeline configuration from environment."""
    return PipelineConfig.from_env()


def create_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    vocabulary_path: Optional[str] = None,
    strict_grounding: bool = True,
    **kwargs
) -> PipelineConfig:
    """
    Convenience function to create a configuration.
    
    Args:
        api_key: OpenAI API key (or compatible provider key)
        base_url: API base URL (defaults to OpenAI)
        model: Model name to use
        vocabulary_path: Path to robot vocabulary JSON
        strict_grounding: Whether to fail on unknown entities
        **kwargs: Additional configuration options
        
    Returns:
        Configured PipelineConfig
        
    Example:
        # Use OpenAI
        config = create_config(api_key="sk-...")
        
        # Use local Ollama
        config = create_config(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3"
        )
        
        # Use Azure OpenAI
        config = create_config(
            api_key="your-azure-key",
            base_url="https://your-resource.openai.azure.com/",
            model="gpt-4"
        )
    """
    config = PipelineConfig()
    
    if api_key:
        config.openai.api_key = api_key
    if base_url:
        config.openai.base_url = base_url
    if model:
        config.openai.model = model
    if vocabulary_path:
        config.grounding.vocabulary_path = vocabulary_path
    
    config.grounding.strict_mode = strict_grounding
    
    # Handle additional kwargs
    if "temperature" in kwargs:
        config.openai.temperature = kwargs["temperature"]
    if "max_tokens" in kwargs:
        config.openai.max_tokens = kwargs["max_tokens"]
    if "timeout" in kwargs:
        config.openai.timeout = kwargs["timeout"]
    if "synthesis_timeout" in kwargs:
        config.synthesis.timeout = kwargs["synthesis_timeout"]
    if "solver" in kwargs:
        config.synthesis.solver = kwargs["solver"]
    
    return config
