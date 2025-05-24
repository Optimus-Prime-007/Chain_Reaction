from typing import List, Tuple, Any, Optional

# Attempt to import AlphaZeroNet for type hinting.
# If this causes issues in a larger project structure due to circular dependencies,
# a string literal 'AlphaZeroNet' is a safe fallback.
try:
    from .neural_net import AlphaZeroNet
except ImportError:
    # This allows the type hint 'AlphaZeroNet' to be used as a forward reference
    # if direct import fails (e.g. during initial setup or if models are in different locations)
    pass

# Assuming models.py is in the same directory or accessible via PYTHONPATH
# Adjust the import path according to your project structure.
try:
    from ..models import GameState, PlayerId, Position
except ImportError:
    # Fallback for environments where the relative import might not work as expected
    # This might happen if the script is run directly for testing
    from ai.models import GameState, PlayerId, Position


def play_game(
    nn: 'AlphaZeroNet', 
    config: Any, 
    game_rules: Any, 
    mcts: Any, 
    utils: Any
) -> List[Tuple[Any, List[float], float]]:
  """
  Simulates a full game of self-play using MCTS and a neural network,
  collecting training data.

  Args:
    nn: The neural network (AlphaZeroNet instance) used by MCTS to evaluate
        states and provide initial policy estimates.
    config: A configuration object containing parameters like GRID_SIZE,
            PLAYERS_INFO, MCTS simulation counts, temperature settings, etc.
    game_rules: A module/object providing game-specific logic, including
                functions like get_initial_state and apply_move.
    mcts: A module/object providing MCTS functionality, including
          get_policy_for_state and select_move.
    utils: A module/object providing utility functions, including
           encode_state and convert_move_index_to_position.

  Returns:
    A list of processed training examples. Each example is a tuple:
    (encoded_state, policy_target, outcome), where:
      - encoded_state: The game state encoded from the perspective of the
                       player whose turn it was.
      - policy_target: The MCTS policy vector for that state.
      - outcome: The game outcome from the perspective of the player
                 (-1.0 for loss, 0.0 for draw, 1.0 for win).
  """
  # 1. Initialization
  # Assuming config.GRID_SIZE is a GridSizeConfig object
  # Assuming config.PLAYERS_INFO is a List[PlayerInfo]
  game_state: GameState = game_rules.get_initial_state(config.GRID_SIZE, config.PLAYERS_INFO)
  training_data_buffer: List[Tuple[Any, List[float], PlayerId]] = [] # Store (encoded_state, policy, player_id_for_perspective)

  # 2. Game Loop
  while game_state.status == "active": # Assuming "active" matches GameStatus.ACTIVE
    current_player_id: Optional[PlayerId] = game_state.currentPlayerId
    
    if current_player_id is None:
      # This case should ideally not happen in an active game with players defined.
      # If it does, it might indicate an issue with game_rules.apply_move or get_initial_state.
      print("Error: currentPlayerId is None during an active game. Breaking self-play loop.")
      break

    encoded_state_perspective = utils.encode_state(game_state, current_player_id)
    
    # mcts.get_policy_for_state runs MCTS simulations and returns a policy vector
    # The policy vector is a list of probabilities for all possible actions (num_actions long)
    mcts_policy_vector = mcts.get_policy_for_state(nn, game_state, config)
    
    training_data_buffer.append((encoded_state_perspective, mcts_policy_vector, current_player_id))
    
    # Move selection based on MCTS policy and potentially temperature (handled in select_move)
    move_index = mcts.select_move(mcts_policy_vector, game_state.turnNumber, config)
    
    # Convert flat move_index to Position(row, col)
    # Using game_state.gridConfiguration.cols as per GameState model
    move_position = utils.convert_move_index_to_position(move_index, game_state.gridConfiguration.cols)
    
    game_state = game_rules.apply_move(game_state, move_position, current_player_id)

  # 3. Determine Game Outcome & Process Buffer
  final_status = game_state.status
  winner_id = game_state.winnerId
  
  processed_training_examples: List[Tuple[Any, List[float], float]] = []
  
  for encoded_state, policy_target, player_id_for_state in training_data_buffer:
    outcome = 0.0  # Default to draw
    if final_status == "finished" and winner_id is not None: # Assuming "finished" matches GameStatus.FINISHED
      if winner_id == player_id_for_state:
        outcome = 1.0  # Win
      else:
        outcome = -1.0  # Loss
    # If final_status is "draw" (or any other non-"finished" status, or finished with no winner), outcome remains 0.0
    
    processed_training_examples.append((encoded_state, policy_target, outcome))

  # 4. Return
  return processed_training_examples
