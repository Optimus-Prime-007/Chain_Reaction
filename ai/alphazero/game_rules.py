from typing import List, Optional, Any
from ai.models import GameState, Position, PlayerId

# (Assuming GameState, Position, PlayerId are defined in ai.models)

def get_valid_moves(state: GameState) -> List[Position]:
    """
    Returns a list of all valid moves for the current player in the given state.
    To be implemented based on specific game rules.
    """
    print(f"DEBUG: game_rules.get_valid_moves called for state turn {state.turnNumber}, player {state.currentPlayerId}")
    # Placeholder: Return an empty list or a dummy move if needed for basic testing without full game logic
    # For now, to avoid breaking MCTS which expects moves, let's assume a dummy grid and return one move.
    # This will need to be properly implemented.
    if state.gridSize.rows > 0 and state.gridSize.cols > 0:
        # Example: return a corner if grid exists, otherwise empty list
        # return [Position(row=0, col=0)]
        pass # Placeholder, MCTS should ideally handle empty valid_moves if game ends.
    return [] # Must be implemented

def get_reward(state: GameState, root_player_id: PlayerId) -> float:
    """
    Returns the reward for the root_player_id given a terminal state.
    +1 for win, -1 for loss, 0 for draw.
    Perspective is crucial: reward for `root_player_id`.
    To be implemented based on specific game rules.
    """
    print(f"DEBUG: game_rules.get_reward called for state turn {state.turnNumber}, root_player {root_player_id}, winner {state.winnerId}")
    if state.winnerId is None:
        return 0.0  # Draw or ongoing (though should be terminal for reward)
    if state.winnerId == root_player_id:
        return 1.0  # root_player_id wins
    else:
        return -1.0 # root_player_id loses (opponent won)

def is_terminal(state: GameState) -> bool:
    """
    Checks if the given state is a terminal state (game over).
    To be implemented based on specific game rules.
    """
    print(f"DEBUG: game_rules.is_terminal called for state turn {state.turnNumber}, status {state.status}")
    return state.status == "finished" or state.status == "draw" # Based on GameState model

def apply_move(state: GameState, move: Position, player_id: PlayerId) -> GameState:
    """
    Applies the given move for the player_id to the state and returns a NEW GameState object.
    This function must not modify the input state (immutable states are preferred).
    To be implemented based on specific game rules.
    """
    print(f"DEBUG: game_rules.apply_move called: player {player_id} moves to {move.row},{move.col} in turn {state.turnNumber}")
    # Placeholder: Returns a copy of the state, possibly with the move made.
    # This is a critical function and needs full implementation.
    # For MCTS to work, it needs to simulate moves.
    new_state = state.copy(deep=True)
    
    # Basic example: Set cell, change current player, increment turn.
    # This is NOT the real game logic.
    if 0 <= move.row < new_state.gridSize.rows and 0 <= move.col < new_state.gridSize.cols:
        # new_state.grid[move.row][move.col].player = player_id # Example
        # new_state.grid[move.row][move.col].orbs = 1 # Example
        pass # Actual game logic for applying move is needed here

    # Determine next player (simple alternation for placeholder)
    # current_player_index = -1
    # for i, p_info in enumerate(new_state.players):
    #     if p_info.id == player_id:
    #         current_player_index = i
    #         break
    # if current_player_index != -1 and len(new_state.players) > 0:
    #     next_player_index = (current_player_index + 1) % len(new_state.players)
    #     new_state.currentPlayerId = new_state.players[next_player_index].id
    # else:
    #     new_state.currentPlayerId = None # Or handle error

    # new_state.turnNumber += 1
    # new_state.status = "active" # Or check for game end
    
    # For now, just return a deep copy. The MCTS expansion requires this to not fail.
    # The actual move application logic is complex and game-specific.
    return new_state # Must be implemented correctly

def get_next_player_id(state: GameState, current_player_id: PlayerId) -> Optional[PlayerId]:
    """
    Determines the ID of the next player.
    To be implemented based on specific game rules (e.g., simple alternation, skipping players).
    """
    print(f"DEBUG: game_rules.get_next_player_id called for current player {current_player_id} in turn {state.turnNumber}")
    player_ids = [p.id for p in state.players]
    if not player_ids:
        return None
    try:
        current_idx = player_ids.index(current_player_id)
        next_idx = (current_idx + 1) % len(player_ids)
        return player_ids[next_idx]
    except ValueError: # current_player_id not in list
        return None # Or raise error

# It's good practice to also define an initial state function if not already present.
# def get_initial_state(grid_size_config: Any, players: List[PlayerInfo]) -> GameState:
#     pass # To be implemented
