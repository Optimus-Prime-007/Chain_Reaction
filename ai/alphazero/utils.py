from typing import Any, Dict
from ai.models import GameState, Position, GridSizeConfig # Assuming these are in ai.models

# Example: Maximum number of possible moves if policy is a flat vector.
# This needs to be defined based on game rules, e.g., rows * cols for a grid game.
# Or, if policy is a dict, this might not be strictly needed here.
# Let's assume for now the NN expects a flat policy vector for all possible cell selections.
# MAX_POLICY_INDEX = 7 * 6 # Example for a 7x6 grid, to be configured properly.

def encode_state(state: GameState) -> Any:
    """
    Encodes the game state into a format suitable for the neural network input.
    The actual representation (e.g., numpy array) depends on the NN architecture.
    To be implemented.
    """
    print(f"DEBUG: utils.encode_state called for state turn {state.turnNumber}")
    # Placeholder: return a simple representation or raise NotImplementedError
    # For now, returning a dict to simulate some encoding.
    # This needs to be a tensor (e.g., numpy array) for the actual NN.
    return {
        "grid": [[(c.player, c.orbs) for c in row] for row in state.grid],
        "currentPlayerId": state.currentPlayerId,
        "turn": state.turnNumber
    } # Must be implemented to return a format the NN expects (e.g., np.ndarray)

def get_move_to_policy_idx_map(grid_size: GridSizeConfig) -> Dict[Position, int]:
    """
    Creates a mapping from a game Position (row, col) to a unique integer index
    in the neural network's policy output vector.
    This is crucial for interpreting the NN's policy logits.
    Assumes a flat policy vector where each cell on the grid corresponds to an index.
    To be implemented based on how moves are represented in the policy vector.
    """
    print(f"DEBUG: utils.get_move_to_policy_idx_map called for grid size {grid_size.rows}x{grid_size.cols}")
    mapping = {}
    idx = 0
    for r in range(grid_size.rows):
        for c in range(grid_size.cols):
            mapping[Position(row=r, col=c)] = idx
            idx += 1
    return mapping # Must be implemented correctly based on NN policy head design

# Potentially, an inverse mapping might also be useful:
# def get_policy_idx_to_move_map(grid_size: GridSizeConfig) -> Dict[int, Position]:
#    pass
