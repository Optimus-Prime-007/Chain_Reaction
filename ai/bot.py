import random
from typing import List, Optional

# Make sure models are imported from .models
from .models import GameState, Position, PlayerId, CellState

def get_random_ai_move(game_state: GameState) -> Optional[Position]:
    """
    Selects a random valid move for the AI player.
    A move is valid if the cell is empty or already belongs to the AI player.
    """
    if game_state.currentPlayerId is None:
        return None # Or raise an error, game should have a current player

    ai_player_id: PlayerId = game_state.currentPlayerId
    valid_moves: List[Position] = []
    
    for r_idx, row_data in enumerate(game_state.grid):
        for c_idx, cell in enumerate(row_data):
            if cell.player is None or cell.player == ai_player_id:
                valid_moves.append(Position(row=r_idx, col=c_idx))

    if not valid_moves:
        return None # No valid moves found (e.g., grid is full of opponent pieces)

    return random.choice(valid_moves)
