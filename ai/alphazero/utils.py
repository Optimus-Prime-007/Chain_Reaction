import torch
from typing import Optional # Required for Optional[PlayerId]

# Assuming models.py is in the parent directory 'ai'
# Adjust based on actual execution environment if this relative import fails
try:
    from ..models import GameState, PlayerId, CellState, Position # Added Position
except ImportError:
    # Fallback for environments where the relative import might not work as expected
    from ai.models import GameState, PlayerId, CellState, Position # Added Position


def encode_state(game_state: GameState, player_id: PlayerId) -> torch.Tensor:
  """
  Encodes the current game state into a PyTorch tensor from the perspective
  of the given player_id.

  The tensor has 5 channels:
  - Channel 0: Own orbs (normalized by 4.0).
  - Channel 1: Opponent orbs (normalized by 4.0).
  - Channel 2: Own player cells (binary).
  - Channel 3: Opponent player cells (binary).
  - Channel 4: Current turn plane (binary: 1.0 if current player's turn).

  Args:
    game_state: The current state of the game.
    player_id: The ID of the player for whom the state is being encoded.

  Returns:
    A PyTorch tensor of shape (5, rows, cols) representing the encoded state.

  Raises:
    ValueError: If the game is not a 2-player game or if player_id is not found.
  """
  if len(game_state.players) != 2:
    raise ValueError("State encoding currently assumes a 2-player game.")

  opponent_id: Optional[PlayerId] = None
  if game_state.players[0].id == player_id:
    opponent_id = game_state.players[1].id
  elif game_state.players[1].id == player_id:
    opponent_id = game_state.players[0].id
  else:
    # This case should ideally not be reached if player_id is validated before calling
    raise ValueError(f"Player ID {player_id} not found in game_state.players list.")

  rows = game_state.gridConfiguration.rows
  cols = game_state.gridConfiguration.cols
  
  # Initialize tensor: C=5 channels (own_orbs, opp_orbs, own_cells, opp_cells, turn_plane)
  encoded_tensor = torch.zeros((5, rows, cols), dtype=torch.float32)

  for r in range(rows):
    for c in range(cols):
      cell = game_state.grid[r][c]
      cell_owner = cell.player
      orbs = cell.orbs

      if cell_owner == player_id:
        encoded_tensor[0, r, c] = orbs / 4.0  # Own orbs
        encoded_tensor[2, r, c] = 1.0       # Own player cells
      elif cell_owner == opponent_id:
        encoded_tensor[1, r, c] = orbs / 4.0  # Opponent orbs
        encoded_tensor[3, r, c] = 1.0       # Opponent player cells
      # If cell_owner is None, orbs are implicitly 0 for channels 0 & 1,
      # and cells are 0 for channels 2 & 3 due to torch.zeros initialization.

  # Channel 4: Current Turn Plane
  if game_state.currentPlayerId == player_id:
    encoded_tensor[4, :, :] = 1.0

  return encoded_tensor

def convert_move_index_to_position(move_index: int, num_cols: int) -> Position:
  """
  Converts a flat move index into a Position object (row, col).

  Args:
    move_index: The flat index of the move in a row-major ordered grid.
                (e.g., 0 for (0,0), 1 for (0,1), ..., num_cols for (1,0)).
    num_cols: The number of columns in the grid.

  Returns:
    A Position object with the calculated row and column.
  
  Raises:
    ValueError: If move_index is negative or num_cols is not positive.
  """
  if move_index < 0:
    raise ValueError("move_index cannot be negative.")
  if num_cols <= 0:
    raise ValueError("num_cols must be a positive integer.")
    
  row = move_index // num_cols
  col = move_index % num_cols
  return Position(row=row, col=col)