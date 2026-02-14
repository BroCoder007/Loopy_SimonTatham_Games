#!/usr/bin/env python3
"""
Test script for the enhanced Advanced DP hint prioritization.
This script verifies that the new hint generation logic works correctly.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.game_state import GameState
from logic.graph import Graph

def test_enhanced_hints():
    """Test the enhanced hint prioritization system."""
    
    print("🧪 Testing Enhanced Advanced DP Hint Prioritization")
    print("=" * 60)
    
    # Create a simple 3x3 test puzzle
    rows, cols = 3, 3
    
    # Initialize game state
    game_state = GameState(rows, cols)
    
    # Add some simple clues (3x3 puzzle with a simple loop)
    clues = {
        (0, 0): 1, (0, 1): 3, (0, 2): 1,
        (1, 0): 3, (1, 1): 0, (1, 2): 3,
        (2, 0): 1, (2, 1): 3, (2, 2): 1
    }
    game_state.clues = clues
    
    # Set game mode for hint generation
    game_state.game_mode = "vs_cpu"
    game_state.turn = "Player 1 (Human)"
    
    # Create solver
    solver = AdvancedDPSolver(game_state)
    
    print("📊 Computing solution...")
    try:
        solver._compute_full_solution()
        print(f"✅ Solution computed with {len(solver._final_solution_edges)} edges")
    except Exception as e:
        print(f"❌ Solution computation failed: {e}")
        return False
    
    print("\n🎯 Testing Hint Generation Priorities:")
    print("-" * 40)
    
    # Test hint generation
    hint = solver.generate_hint()
    
    if hint["move"]:
        edge = hint["move"]
        strategy = hint["strategy"]
        explanation = hint["explanation"]
        
        print(f"✅ Hint Generated: {edge}")
        print(f"📋 Strategy: {strategy}")
        print(f"💡 Explanation: {explanation}")
        
        # Verify explanation contains key phrases
        key_phrases = [
            "DP + Divide & Conquer",
            "boundary",
            "merge",
            "configurations"
        ]
        
        found_phrases = [phrase for phrase in key_phrases if phrase.lower() in explanation.lower()]
        print(f"\n🔍 Explanation Quality Check:")
        print(f"   Found key phrases: {found_phrases}")
        
        if len(found_phrases) >= 2:
            print("✅ Explanation contains relevant technical details")
        else:
            print("⚠️  Explanation could be more detailed")
            
    else:
        print("ℹ️  No hint generated (puzzle may be complete)")
    
    print("\n🎨 Testing Different Hint Scenarios:")
    print("-" * 40)
    
    # Test different priority levels
    print("1️⃣  Testing Forced Inclusion Detection...")
    forced_inclusions = solver._detect_forced_inclusions(game_state)
    print(f"   Found {len(forced_inclusions)} forced inclusion edges")
    
    print("2️⃣  Testing Forced Exclusion Detection...")
    forced_exclusions = solver._detect_forced_exclusions(game_state)
    print(f"   Found {len(forced_exclusions)} forced exclusion edges")
    
    print("3️⃣  Testing Boundary Compatibility Detection...")
    boundary_forced = solver._detect_boundary_compatibility_forced_edges(game_state)
    print(f"   Found {len(boundary_forced)} boundary compatibility edges")
    
    print("4️⃣  Testing Pruning Forced Exclusions...")
    pruning_forced = solver._detect_pruning_forced_exclusions(game_state)
    print(f"   Found {len(pruning_forced)} pruning forced edges")
    
    print("\n📈 Merge Statistics:")
    print("-" * 40)
    
    merge_stats = solver._merge_stats
    print(f"Region Statistics: {len(merge_stats.get('region_stats', {}))} regions")
    print(f"Merge Details: {len(merge_stats.get('merge_details', []))} merge operations")
    
    if merge_stats.get('merge_details'):
        total_pruned = sum(m.get('pruned_count', 0) for m in merge_stats['merge_details'])
        total_candidates = sum(m.get('total_candidates', 0) for m in merge_stats['merge_details'])
        print(f"Total Pruned: {total_pruned} of {total_candidates} candidates")
        
        if total_candidates > 0:
            prune_rate = (total_pruned / total_candidates) * 100
            print(f"Prune Rate: {prune_rate:.1f}%")
    
    print("\n🎯 Presentation Mode Analysis:")
    print("-" * 40)
    
    # Analyze if hints are constructive vs reactive
    if hint["move"]:
        edge = hint["move"]
        current_edges = set(game_state.graph.edges)
        
        if edge in current_edges:
            print("⚠️  Hint suggests REMOVAL (reactive)")
        else:
            print("✅ Hint suggests ADDITION (constructive)")
            
        # Check if explanation mentions solution comparison
        if "solution" in hint["explanation"].lower() and "compare" in hint["explanation"].lower():
            print("⚠️  Explanation mentions solution comparison (weak)")
        else:
            print("✅ Explanation focuses on constraints (strong)")
    
    print("\n🏁 Test Complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_enhanced_hints()
        if success:
            print("✅ Enhanced hint prioritization test completed successfully!")
        else:
            print("❌ Test failed - check implementation")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
