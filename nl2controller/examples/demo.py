#!/usr/bin/env python3
"""
NL2Controller Demo

This script demonstrates the complete pipeline for converting
natural language commands to reactive controllers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import NL2ControllerPipeline


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def demo_basic_commands():
    """Demonstrate basic command processing."""
    print_separator("NL2CONTROLLER PIPELINE DEMONSTRATION")
    
    # Create pipeline with default vocabulary
    vocab_path = Path(__file__).parent.parent / "data" / "robot_vocabulary.json"
    pipeline = NL2ControllerPipeline(
        vocabulary_path=str(vocab_path),
        use_mock_lifting=True,
        use_mock_synthesis=True,
    )
    
    # Test commands
    test_commands = [
        "Patrol Zone A",
        "Avoid Zone C",
        "If low battery, go to charging station",
        "Patrol Zone A and avoid Zone C",
    ]
    
    print("\nPipeline Configuration:")
    print(f"  - Vocabulary entries: {len(pipeline.get_available_phrases())}")
    print(f"  - Categories: {pipeline.get_vocabulary_categories()}")
    print(f"  - Using mock lifting: True")
    print(f"  - Using mock synthesis: True")
    print(f"  - Strict grounding: True")
    
    for cmd in test_commands:
        print_separator(f"Command: '{cmd}'")
        
        # Process command
        result = pipeline.process(cmd)
        
        print(f"\n📝 Input: {result.original_command}")
        print(f"\n🔄 Stage 1 - LIFTING:")
        if result.lifting_result:
            print(f"   Pattern: {result.lifting_result.pattern.name}")
            print(f"   Entities: {[e.text for e in result.lifting_result.entities]}")
        print(f"   Abstract LTL: {result.abstract_ltl}")
        
        print(f"\n🎯 Stage 2 - GROUNDING:")
        if result.grounding_result:
            print(f"   Mappings:")
            for ge in result.grounding_result.grounded_entities:
                print(f"     '{ge.original_text}' → {ge.system_var}")
        print(f"   Grounded LTL: {result.grounded_ltl}")
        
        print(f"\n⚙️  Stage 3 - SYNTHESIS:")
        if result.success:
            ctrl = result.controller
            print(f"   Controller: {ctrl.num_states} states, {ctrl.num_transitions} transitions")
            print(f"   States: {[s.name for s in ctrl.states]}")
            print(f"   Initial: {ctrl.initial_states}")
        
        print(f"\n{'✅ SUCCESS' if result.success else '❌ FAILED'}")
        if not result.success:
            print(f"   Error at: {result.error_stage}")
            print(f"   Message: {result.error_message}")
        
        if result.warnings:
            print(f"   Warnings: {result.warnings}")
        
        print(f"\n   Total time: {result.total_time:.4f}s")


def demo_validation():
    """Demonstrate command validation."""
    print_separator("COMMAND VALIDATION DEMO")
    
    vocab_path = Path(__file__).parent.parent / "data" / "robot_vocabulary.json"
    pipeline = NL2ControllerPipeline(vocabulary_path=str(vocab_path))
    
    commands = [
        "Patrol Zone A",           # Valid
        "Patrol Zone X",           # Invalid - Zone X not in vocabulary
        "If battery low, charge",  # Valid
        "Fire the lasers",         # Invalid - not in vocabulary
    ]
    
    for cmd in commands:
        print(f"\nValidating: '{cmd}'")
        validation = pipeline.validate_command(cmd)
        
        lifting_status = "✅" if validation["lifting_ok"] else "❌"
        grounding_status = "✅" if validation["grounding_ok"] else "❌"
        
        print(f"  Lifting:   {lifting_status} (Pattern: {validation.get('pattern', 'N/A')})")
        print(f"  Grounding: {grounding_status}")
        
        if not validation["grounding_ok"]:
            print(f"  Ungrounded: {validation.get('ungrounded_entities', [])}")
            if validation.get('suggestions'):
                print(f"  Suggestions: {validation['suggestions']}")


def demo_controller_export():
    """Demonstrate controller export formats."""
    print_separator("CONTROLLER EXPORT DEMO")
    
    vocab_path = Path(__file__).parent.parent / "data" / "robot_vocabulary.json"
    pipeline = NL2ControllerPipeline(vocabulary_path=str(vocab_path))
    
    result = pipeline.process("Patrol Zone A")
    
    if result.success:
        print("\n📊 Controller Dictionary Export:")
        ctrl_dict = result.controller.to_dict()
        import json
        print(json.dumps(ctrl_dict, indent=2)[:500] + "...")
        
        print("\n📈 Controller DOT Export (for Graphviz):")
        dot_output = result.controller.to_dot()
        print(dot_output)


def demo_multi_pattern():
    """Demonstrate multi-pattern command processing."""
    print_separator("MULTI-PATTERN COMMAND DEMO")
    
    vocab_path = Path(__file__).parent.parent / "data" / "robot_vocabulary.json"
    pipeline = NL2ControllerPipeline(vocabulary_path=str(vocab_path))
    
    cmd = "Patrol Zone A and Zone B, but avoid Zone C"
    print(f"\n🎯 Complex Command: '{cmd}'")
    
    result = pipeline.process_multi(cmd)
    
    print(f"\n📝 Abstract LTL: {result.abstract_ltl}")
    print(f"🎯 Grounded LTL: {result.grounded_ltl}")
    
    if result.success:
        print(f"\n✅ Synthesized controller with {result.controller.num_states} states")


def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                      NL2CONTROLLER PIPELINE                          ║
║         Natural Language to Reactive Controller Synthesis            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Architecture: Constrained Lift + Dictionary Grounding               ║
║  Soundness: Pattern templates + Closed vocabulary                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    demo_basic_commands()
    demo_validation()
    demo_controller_export()
    demo_multi_pattern()
    
    print_separator("DEMO COMPLETE")
    print("\n✅ All demonstrations completed successfully!")
    print("\nNext steps:")
    print("  1. Add more entries to data/robot_vocabulary.json")
    print("  2. Test with real LLM by providing OPENAI_API_KEY")
    print("  3. Install TuLiP for real GR(1) synthesis")
    print("  4. Integrate with your robot's API")


if __name__ == "__main__":
    main()
