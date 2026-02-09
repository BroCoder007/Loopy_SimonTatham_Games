"""
Solver Utilities
================
Core utilities for the Phase 2 "Super Solver".
Includes Zobrist Hashing for O(1) state comparison and Constraint Propagation for logical pruning.
"""

import random

class ZobristHasher:
    """
    Implements Zobrist Hashing for O(1) board state memoization.
    
    Assigns a random 64-bit integer to every possible edge.
    The hash of a board state is the XOR sum of the random integers of its active edges.
    """
    def __init__(self, num_edges):
        """
        Initialize the Zobrist table.
        
        Args:
            num_edges (int): Total number of possible edges in the grid.
                             For an RxC grid, this is roughly 2*R*C + R + C.
        """
        self.table = [random.getrandbits(64) for _ in range(num_edges)]
        
    def compute_hash(self, active_edge_indices):
        """
        Compute the hash for a configuration of active edges.
        
        Args:
            active_edge_indices (iter): An iterable of integer indices of currently active edges.
            
        Returns:
            int: The 64-bit hash of the board state.
        """
        current_hash = 0
        for idx in active_edge_indices:
            current_hash ^= self.table[idx]
        return current_hash


class ConstraintPropagator:
    """
    Applies cheap logical rules to prune the search space.
    
    This class assumes the existence of a 'SolverGrid' interface (Duck Typing)
    on the passed 'grid' object.
    
    Expected Interface for 'grid':
    - grid.rows: int
    - grid.cols: int
    - grid.get_clue(r, c): returns int or None
    - grid.get_neighbors(r, c): returns list of edge_indices
    - grid.get_edge_status(edge_index): returns 1 (ON), 0 (OFF), -1 (UNKNOWN)
    - grid.set_edge_status(edge_index, status): sets status, returns True if changed
    - grid.is_valid(): returns bool (checks for obvious contradictions like >2 edges at node)
    """

    @staticmethod
    def propagate(grid):
        """
        Iteratively applies logical rules until no more changes can be made.
        
        Args:
            grid: An object satisfying the SolverGrid interface.
            
        Returns:
            tuple: (is_valid, changes_made)
                   is_valid (bool): False if a contradiction is found.
                   changes_made (set): Set of edge indices that were changed (for undo/tracking).
        """
        changes_mode = set()
        keep_propagating = True
        
        while keep_propagating:
            keep_propagating = False
            current_changes = set()
            
            # Iterate over all cells with clues
            for r in range(grid.rows):
                for c in range(grid.cols):
                    clue = grid.get_clue(r, c)
                    if clue is None:
                        continue
                        
                    neighbors = grid.get_neighbors(r, c)
                    on_count = 0
                    off_count = 0
                    unknown_edges = []
                    
                    for edge_idx in neighbors:
                        status = grid.get_edge_status(edge_idx)
                        if status == 1:
                            on_count += 1
                        elif status == 0:
                            off_count += 1
                        else:
                            unknown_edges.append(edge_idx)
                            
                    # Rule 0: Contradiction check
                    if on_count > clue:
                        return False, changes_mode
                    if (len(neighbors) - off_count) < clue:
                        return False, changes_mode
                        
                    # Rule 1 (Clue 0): All neighbors must be OFF
                    if clue == 0:
                        for edge_idx in unknown_edges:
                            if grid.set_edge_status(edge_idx, 0):
                                current_changes.add(edge_idx)
                                keep_propagating = True
                                
                    # Rule 2 (Clue Satisfied): Remaining must be OFF
                    elif on_count == clue:
                        for edge_idx in unknown_edges:
                            if grid.set_edge_status(edge_idx, 0):
                                current_changes.add(edge_idx)
                                keep_propagating = True
                                
                    # Rule 3 (Must Fill): Remaining must be ON
                    elif (on_count + len(unknown_edges)) == clue:
                        for edge_idx in unknown_edges:
                            if grid.set_edge_status(edge_idx, 1):
                                current_changes.add(edge_idx)
                                keep_propagating = True
                                
                    # Rule 4 (Corner 3): A '3' in a corner means the two outer edges must be ON
                    # (This is a simplified version, robust general implementation would check
                    # if a cell is a corner of the grid)
                    if clue == 3:
                         # Check if (r, c) is a grid corner
                         is_corner = (r == 0 or r == grid.rows - 1) and (c == 0 or c == grid.cols - 1)
                         if is_corner:
                             # For a corner cell, the two edges touching the outer boundary must be ON.
                             # This requires knowing which edges are boundary edges.
                             # For now, we rely on the generic Rule 3 which already covers this if 
                             # the other 2 edges are internal.
                             # If we want to force it explicitly without knowing edge types, 
                             # we'd need more info from the grid interface.
                             pass 

            if current_changes:
                changes_mode.update(current_changes)
                
            # Perform a quick validity check on the grid state
            if not grid.is_valid():
                return False, changes_mode
                
        return True, changes_mode


class BoundaryProfile:
    """
    Represents the state of a grid boundary for Divide & Conquer or DP approaches.
    
    In the "Broken Profile DP" context, this tracks the connectivity or "crossing state"
    of the boundary line that separates processed cells from unprocessed cells.
    
    Attributes:
        profile_data (tuple): A tuple representing the state.
                              Could be a tuple of booleans (crossing/not crossing)
                              or component IDs (for connectivity tracking).
    """
    def __init__(self, profile_data):
        """
        Initialize the BoundaryProfile.
        
        Args:
            profile_data (iterable): The state data (converted to a tuple for immutability).
        """
        self.profile_data = tuple(profile_data)
        # Precompute hash since profiles are used as dict keys frequently
        self._hash = hash(self.profile_data)

    def __hash__(self):
        """
        Crucial for using this object as a key in DP memoization tables.
        
        Returns:
            int: The hash of the profile data.
        """
        return self._hash

    def __eq__(self, other):
        """
        Check for equality with another BoundaryProfile.
        
        Args:
            other (BoundaryProfile): The other profile to compare.
            
        Returns:
            bool: True if profile_data is identical.
        """
        if not isinstance(other, BoundaryProfile):
            return False
        return self.profile_data == other.profile_data

    def to_bitmask(self):
        """
        Convert the profile to an integer bitmask (if applicable).
        
        Useful for Zobrist hashing or compact storage if the profile consists
        of boolean values (e.g., edge ON/OFF).
        
        Returns:
            int: Integer representation of the profile.
        """
        mask = 0
        try:
            for i, val in enumerate(self.profile_data):
                if val:
                    mask |= (1 << i)
        except TypeError:
            # Fallback if profile data isn't boolean-like integers
            pass
        return mask

    def __repr__(self):
        return f"BoundaryProfile({self.profile_data})"
