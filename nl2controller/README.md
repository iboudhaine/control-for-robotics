# NL2Controller: Sound NLP-to-Controller Pipeline

A robust software pipeline that accepts Natural Language (NLP) commands and automatically synthesizes a correct-by-construction feedback controller for robotic systems.

## Architecture: Constrained Lift + Dictionary Grounding

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NL2Controller Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   Stage 1    │    │   Stage 2    │    │      Stage 3         │   │
│  │   LIFTING    │───▶│  GROUNDING   │───▶│     SYNTHESIS        │   │
│  │  (nl2ltl)    │    │ (Dictionary) │    │      (TuLiP)         │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                      │
│  Input: NL Command   Pattern + Entities  Grounded LTL    Controller │
│  "Patrol Zone A"   → G(F(p1))          → G(F(at_zone_a))→ Automaton │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Soundness Guarantees

1. **Pattern Templates** (Stage 1): Constrains LLM output to Declare patterns - prevents LTL hallucination
2. **Closed Dictionary** (Stage 2): Maps entities to known variables only - prevents variable hallucination
3. **Formal Synthesis** (Stage 3): GR(1) game solving - guarantees realizable controllers

## Project Structure

```
nl2controller/
├── src/
│   ├── lifting/           # Stage 1: NL to abstract LTL patterns
│   ├── grounding/         # Stage 2: Dictionary-based variable mapping
│   ├── synthesis/         # Stage 3: TuLiP integration for controller synthesis
│   ├── simulation/        # Robot simulation and visualization
│   ├── translation/       # Darija (Moroccan dialect) translation support
│   ├── pipeline.py        # Main orchestrator
│   └── config.py          # Configuration management
├── tests/
│   ├── test_lifting.py
│   ├── test_grounding.py
│   ├── test_synthesis.py
│   ├── test_simulation.py
│   ├── test_translation.py
│   └── test_integration.py
├── data/
│   └── robot_vocabulary.json  # Robot variable dictionary
├── examples/
│   ├── simple_example.py      # No API key required
│   └── demo.py                # Full demo with OpenAI
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run example
python examples/simple_example.py
```

## Reproducibility Guide

Follow these steps to reproduce the project from scratch:

### 1. Clone the Repository

```bash
git clone https://github.com/SamiAGOURRAM/nl2controller.git
cd nl2controller/nl2controller
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Tests

```bash
pytest tests/ -v
```

Expected output: **161 tests passed**

### 5. Run Example (No API Key Required)

```bash
python examples/simple_example.py
```

This uses a keyword-based lifting engine that doesn't require an LLM API.

### 6. (Optional) Run with OpenAI API

```bash
export OPENAI_API_KEY="your-api-key"
python examples/demo.py
```

### Environment Details

- **Python**: 3.10+ (tested with 3.12)
- **Key Dependencies**: 
  - `tulip>=1.4.0` - GR(1) reactive synthesis
  - `openai>=1.0.0` - LLM integration
  - `pydantic>=2.0.0` - Data validation
  - `pytest>=8.0.0` - Testing

## Usage

```python
from src.pipeline import NL2ControllerPipeline, create_pipeline
from src.lifting.engine import BaseLiftingEngine, LiftingResult, ExtractedEntity
from src.lifting.patterns import get_pattern

# Option 1: Use with custom lifting engine (no API key needed)
class SimpleLiftingEngine(BaseLiftingEngine):
    def lift(self, nl_command: str) -> LiftingResult:
        pattern = get_pattern("always_eventually")  # G(F(x))
        entities = [ExtractedEntity(text="zone a", placeholder="p1", confidence=0.9)]
        return LiftingResult(
            pattern=pattern,
            entities=entities,
            abstract_ltl=pattern.ltl_template.format(*[e.placeholder for e in entities]),
            confidence=0.85,
            original_command=nl_command
        )

pipeline = create_pipeline(
    vocabulary_path="data/robot_vocabulary.json",
    lifting_engine=SimpleLiftingEngine()
)

# Process command
result = pipeline.process("Patrol Zone A")

if result.success:
    print(f"Grounded LTL: {result.grounded_ltl}")
    print(f"Controller states: {len(result.controller.states)}")
else:
    print(f"Failed at: {result.error_stage}")

# Option 2: Use with OpenAI API
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

pipeline = create_pipeline(vocabulary_path="data/robot_vocabulary.json")
result = pipeline.process("Patrol Zone A and avoid Zone C")
```

## Components

### Stage 1: Lifting (nl2ltl)
Extracts temporal patterns from natural language using Declare pattern templates.
Constrains LLM output to predefined patterns (e.g., `always_eventually`, `absence`, `response`).
Output: Abstract LTL formula with placeholders (e.g., `G(F(p1))`)

### Stage 2: Grounding (Dictionary Filter)
Maps extracted entities to robot variables using a closed dictionary (`robot_vocabulary.json`).
Uses atomic boolean propositions compatible with TuLiP (e.g., `at_zone_a`, `battery_low`).
Fails explicitly if entity cannot be grounded (safety feature).

### Stage 3: Synthesis (TuLiP)
Converts grounded LTL to GR(1) specification and synthesizes a reactive controller.
Returns a finite state automaton that guarantees the specification is satisfied.

## References

- [VLTL-Bench Paper](https://arxiv.org/html/2507.00877v1)
- [IBM nl2ltl](https://github.com/IBM/nl2ltl)
- [TuLiP Control](https://tulip-control.sourceforge.io/)
