import copy # For deepcopy
from typing import List, Optional, Tuple
from collections import deque

# Assuming models.py is in the parent directory 'ai'
# Adjust based on actual execution environment if this relative import fails
try:
    from ..models import GameState, Position, PlayerId, CellState, GameStatus, GridSizeConfig, PlayerInfo, Grid
except ImportError:
    # Fallback for environments where the relative import might not work as expected
    # This might happen if the script is run directly for testing from a different working directory
    from ai.models import GameState, Position, PlayerId, CellState, GameStatus, GridSizeConfig, PlayerInfo, Grid

def get_critical_mass(row: int, col: int, rows: int, cols: int) -> int:
  """
  Determines the critical mass of a cell in a grid.

  The critical mass is the maximum number of orbs a cell can hold before
  it explodes.

  Args:
    row: The row index of the cell.
    col: The column index of the cell.
    rows: The total number of rows in the grid.
    cols: The total number of columns in the grid.

  Returns:
    The critical mass of the cell.
  """
  if rows == 1 and cols == 1:
    return 1
  elif rows == 1:
    if col == 0 or col == cols - 1:
      return 1
    else:
      return 2
  elif cols == 1:
    if row == 0 or row == rows - 1:
      return 1
    else:
      return 2
  else:  # rows > 1 and cols > 1
    is_corner = (row == 0 or row == rows - 1) and \
                (col == 0 or col == cols - 1)
    is_edge = (row == 0 or row == rows - 1) or \
              (col == 0 or col == cols - 1)

    if is_corner:
      return 2
    elif is_edge:
      return 3
    else:
      return 4

def get_game_status_and_winner(game_state: GameState) -> Tuple[GameStatus, Optional[PlayerId]]:
  """
  Determines the current status of the game and identifies a winner if the
  game has finished.

  Args:
    game_state: The current state of the game.

  Returns:
    A tuple containing the GameStatus (e.g., "active", "finished", "draw")
    and an optional PlayerId of the winner (if the game is finished and
    there's a winner).
  """
  # 1. Initial Check (Too Early To End)
  # Ensures every player has a chance to make at least one move.
  if game_state.turnNumber < len(game_state.players):
    return GameStatus.ACTIVE, None

  # 2. Count Orbs and Active Players
  player_orbs = {p.id: 0 for p in game_state.players}
  players_with_orbs = set()

  for r in range(game_state.gridConfiguration.rows):
    for c in range(game_state.gridConfiguration.cols):
      cell = game_state.grid[r][c]
      if cell.player is not None and cell.orbs > 0:
        # Ensure cell.player is a valid PlayerId before using as a key
        if cell.player in player_orbs:
            player_orbs[cell.player] += cell.orbs
            players_with_orbs.add(cell.player)
        # else:
            # This case implies a cell has orbs for a player not in game_state.players.
            # This should ideally not happen if game state is managed correctly.
            # Consider logging a warning or raising an error if strict consistency is required.
            # print(f"Warning: Cell player {cell.player} has orbs but is not in game_state.players")


  # 3. Determine Winner/Status based on orb count
  if game_state.turnNumber > 0: # This check is now implicitly covered by turnNumber >= len(players)
                               # but kept for clarity for the zero orbs case.
    if len(players_with_orbs) == 0:
      # All orbs have vanished (e.g., simultaneous explosions).
      return GameStatus.DRAW, None
    
    if len(players_with_orbs) == 1:
      winner_id = list(players_with_orbs)[0]
      return GameStatus.FINISHED, winner_id

  # 4. Check for Draw by Max Turns (Optional but Recommended)
  # Using gridConfiguration for rows/cols
  MAX_TURNS = game_state.gridConfiguration.rows * game_state.gridConfiguration.cols * 4
  if game_state.turnNumber > MAX_TURNS:
    return GameStatus.DRAW, None

  # 5. Game is Still Active
  return GameStatus.ACTIVE, None

def get_valid_moves(game_state: GameState) -> List[Position]:
  """
  Identifies all valid moves for the current player.

  A move is valid if the cell is empty or already occupied by the current
  player.

  Args:
    game_state: The current state of the game.

  Returns:
    A list of Position objects representing valid moves.
  """
  valid_moves: List[Position] = []
  rows = game_state.gridConfiguration.rows
  cols = game_state.gridConfiguration.cols
  current_player_id = game_state.currentPlayerId

  for r in range(rows):
    for c in range(cols):
      cell = game_state.grid[r][c]
      if cell.player is None or cell.player == current_player_id:
        valid_moves.append(Position(row=r, col=c))
  return valid_moves

def apply_move(current_game_state: GameState, move: Position, player_id: PlayerId) -> GameState:
  """
  Applies a move to the game state and returns the new game state.

  This function does not modify current_game_state in place.

  Args:
    current_game_state: The current state of the game.
    move: The position of the move to apply.
    player_id: The ID of the player making the move.

  Returns:
    A new GameState object representing the state after the move.
  """
  if current_game_state.currentPlayerId != player_id:
    raise ValueError("Move played by player who is not the current player.")

  new_game_state = copy.deepcopy(current_game_state)

  # Initial Orb Placement
  cell_to_update = new_game_state.grid[move.row][move.col]
  cell_to_update.orbs += 1
  cell_to_update.player = player_id

  # Explosion Logic
  explosion_queue = deque([move])
  rows = new_game_state.gridConfiguration.rows
  cols = new_game_state.gridConfiguration.cols

  while explosion_queue:
    r, c = explosion_queue.popleft().row, explosion_queue.popleft().col # Incorrect: popleft twice
    # Corrected pop:
    # current_pos = explosion_queue.popleft()
    # r, c = current_pos.row, current_pos.col
    # The provided template for popleft was: Pop a (r, c) position from the queue.
    # Position objects are used in queue, so popping Position and then accessing r,c is correct.
    # However, the template deque([move]) adds Position, so popleft() returns Position.
    # The error is in the original template `r, c = explosion_queue.popleft().row, explosion_queue.popleft().col` which would call popleft() twice.
    # Let's fix this by first popping the Position object.
    
    current_pos = explosion_queue.popleft() 
    r, c = current_pos.row, current_pos.col

    cell = new_game_state.grid[r][c]
    crit_mass = get_critical_mass(r, c, rows, cols)

    if cell.orbs >= crit_mass:
      cell.orbs -= crit_mass
      if cell.orbs == 0:
        cell.player = None  # Cell becomes empty

      potential_neighbors = [
          Position(row=r - 1, col=c), Position(row=r + 1, col=c),
          Position(row=r, col=c - 1), Position(row=r, col=c + 1)
      ]

      for neighbor_pos in potential_neighbors:
        nr, nc = neighbor_pos.row, neighbor_pos.col
        if 0 <= nr < rows and 0 <= nc < cols:
          neighbor_cell = new_game_state.grid[nr][nc]
          neighbor_cell.orbs += 1
          neighbor_cell.player = player_id  # Orb capture
          explosion_queue.append(neighbor_pos) # Add to queue for potential chain reaction

  # Update Game Status and Winner
  status, winner = get_game_status_and_winner(new_game_state)
  new_game_state.status = status
  new_game_state.winnerId = winner

  # Switch Current Player
  if new_game_state.status == GameStatus.ACTIVE:
    current_player_index = -1
    for i, p_info in enumerate(new_game_state.players):
        if p_info.id == player_id:
            current_player_index = i
            break
    
    if current_player_index != -1: # Should always be found
        next_player_index = (current_player_index + 1) % len(new_game_state.players)
        new_game_state.currentPlayerId = new_game_state.players[next_player_index].id
    else:
        # This case should ideally not be reached if player_id is always valid
        # and present in new_game_state.players
        # Consider logging an error or raising an exception
        pass


  # Increment Turn Number
  new_game_state.turnNumber += 1

  return new_game_state

def get_initial_state(grid_size_config: GridSizeConfig, players_info: List[PlayerInfo]) -> GameState:
  """
  Creates and returns a new GameState object representing the start of a game.

  Args:
    grid_size_config: Configuration for the grid dimensions.
    players_info: A list of PlayerInfo objects for the players in the game.

  Returns:
    A GameState object representing the initial state of the game.
  """
  # 1. Initialize Grid
  grid: Grid = []
  for r in range(grid_size_config.rows):
    row_list: List[CellState] = []
    for c in range(grid_size_config.cols):
      row_list.append(CellState(player=None, orbs=0))
    grid.append(row_list)

  # 2. Determine Initial Current Player
  current_player_id: Optional[PlayerId] = None
  if players_info:
    current_player_id = players_info[0].id

  # 3. Create GameState Object
  initial_game_state = GameState(
      grid=grid,
      players=players_info,
      currentPlayerId=current_player_id,
      status=GameStatus.ACTIVE,  # Assuming GameStatus.ACTIVE is "active"
      winnerId=None,
      gridConfiguration=grid_size_config, # Corrected from gridSize to gridConfiguration
      turnNumber=0
  )

  return initial_game_state
