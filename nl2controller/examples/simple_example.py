#!/usr/bin/env python3
"""
Simple Example: Show what the pipeline outputs

This demonstrates the complete flow without requiring an LLM API.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import NL2ControllerPipeline
from src.lifting.engine import BaseLiftingEngine, LiftingResult, ExtractedEntity
from src.lifting.patterns import get_pattern
import re


class SimpleLiftingEngine(BaseLiftingEngine):
    """Simple keyword-based lifting (no LLM API needed)."""
    
    def lift(self, nl_command: str) -> LiftingResult:
        cmd_lower = nl_command.lower()
        
        # Pattern detection
        if "patrol" in cmd_lower:
            pattern = get_pattern("always_eventually")
        elif "avoid" in cmd_lower:
            pattern = get_pattern("absence")
        elif "if" in cmd_lower or "when" in cmd_lower:
            pattern = get_pattern("response")
        else:
            pattern = get_pattern("existence")
        
        # Extract entities (simple regex)
        entities = []
        
        # Find zones/locations
        for match in re.finditer(r"(zone\s+[a-z]|charging\s+station|zone\s+c)", cmd_lower, re.IGNORECASE):
            entities.append(ExtractedEntity(
                text=match.group(),
                placeholder=f"p{len(entities) + 1}",
                confidence=0.9
            ))
        
        # For response patterns, extract condition
        if pattern.arity == 2:
            if "battery" in cmd_lower and "low" in cmd_lower:
                entities.insert(0, ExtractedEntity(
                    text="low battery",
                    placeholder="p1",
                    confidence=0.9
                ))
                # Renumber others
                for i, e in enumerate(entities[1:], start=2):
                    e.placeholder = f"p{i}"
        
        if not entities:
            entities.append(ExtractedEntity(
                text=cmd_lower,
                placeholder="p1",
                confidence=0.5
            ))
        
        # Build abstract LTL
        placeholders = [e.placeholder for e in entities[:pattern.arity]]
        while len(placeholders) < pattern.arity:
            placeholders.append(f"p{len(placeholders) + 1}")
        
        abstract_ltl = pattern.instantiate(*placeholders[:pattern.arity])
        
        return LiftingResult(
            pattern=pattern,
            entities=entities,
            abstract_ltl=abstract_ltl,
            original_command=nl_command,
            confidence=0.8
        )
    
    def lift_multi(self, nl_command: str) -> "MultiLiftingResult":
        from src.lifting.engine import MultiLiftingResult
        return MultiLiftingResult(results=[self.lift(nl_command)])


def main():
    print("=" * 70)
    print("NL2Controller Simple Example (No LLM API Required)")
    print("=" * 70)
    print()
    
    # Create pipeline with simple lifting engine
    vocab_path = Path(__file__).parent.parent / "data" / "robot_vocabulary.json"
    
    pipeline = NL2ControllerPipeline(
        vocabulary_path=str(vocab_path),
        lifting_engine=SimpleLiftingEngine()
    )
    
    # Test commands
    commands = [
        "Patrol Zone A",
        "Avoid Zone C",
        "If low battery, go to charging station",
    ]
    
    for cmd in commands:
        print("\n" + "=" * 70)
        print(f"COMMAND: '{cmd}'")
        print("=" * 70)
        
        result = pipeline.process(cmd)
        
        if result.success:
            print(f"\n✅ SUCCESS")
            print(f"\n📐 Abstract LTL: {result.abstract_ltl}")
            print(f"🎯 Grounded LTL: {result.grounded_ltl}")
            
            print(f"\n🎮 Controller:")
            print(f"   States: {result.controller.num_states}")
            print(f"   Transitions: {result.controller.num_transitions}")
            
            if result.controller.states:
                print(f"\n   State List:")
                for state in result.controller.states:
                    init_marker = " [INITIAL]" if state.is_initial else ""
                    print(f"     - {state.name} (id={state.id}){init_marker}")
            
            if result.controller.transitions:
                print(f"\n   Transitions:")
                for trans in result.controller.transitions[:5]:  # Show first 5
                    print(f"     {trans.source} → {trans.target}: {trans.action}")
                if len(result.controller.transitions) > 5:
                    print(f"     ... and {len(result.controller.transitions) - 5} more")
            
            # Show controller as ASCII
            from src.simulation.visualizer import ControllerVisualizer
            viz = ControllerVisualizer()
            print(f"\n📊 Controller Visualization (ASCII):")
            print(viz.to_ascii(result.controller.to_dict())[:500])  # Truncate for demo
            
        else:
            print(f"\n❌ FAILED at stage: {result.error_stage}")
            print(f"   Error: {result.error_message}")
            if result.warnings:
                print(f"   Warnings: {result.warnings}")
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nTo use with real LLM:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Remove lifting_engine parameter")
    print("  3. Pipeline will use LLM automatically")


if __name__ == "__main__":
    main()

