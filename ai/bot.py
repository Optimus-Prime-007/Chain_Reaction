import random # Keep for get_random_ai_move
from typing import List, Optional, Tuple, Any # Ensure Tuple, Any are added
import torch # For device selection and model loading
import collections # For namedtuple

# Models from the parent 'ai' directory
from .models import GameState, Position, PlayerId, CellState # Adjust if path differs

# AlphaZero components (assuming they are in ai.alphazero)
try:
    from .alphazero.neural_net import AlphaZeroNet
    from .alphazero import mcts as mcts_module
    from .alphazero import utils as utils_module
    from .alphazero import game_rules as game_rules_module
    # az_config will be passed as an argument, typically imported from .alphazero.config
except ImportError as e:
    print(f"Error importing AlphaZero modules in bot.py: {e}. AlphaZero AI will not be available.")
    AlphaZeroNet = None # Define as None so type hints don't break if import fails
    mcts_module, utils_module, game_rules_module = None, None, None

# (Keep other existing imports)

def load_alphazero_model_and_dependencies(az_config: Any) -> Tuple[Optional['AlphaZeroNet'], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
  """
  Loads the AlphaZero neural network model and its dependencies.

  This function attempts to import AlphaZero components, initialize the
  neural network, and load its pre-trained weights.

  Args:
    az_config: The configuration object/module for AlphaZero, containing
               parameters like INPUT_CHANNELS, GRID_SIZE, and
               BEST_MODEL_PATH.

  Returns:
    A tuple containing:
      - nn (Optional[AlphaZeroNet]): The loaded AlphaZeroNet model, or None if loading failed.
      - mcts_module (Optional[Any]): The imported MCTS module, or None if import failed.
      - utils_module (Optional[Any]): The imported utils module, or None if import failed.
      - game_rules_module (Optional[Any]): The imported game_rules module, or None if import failed.
      - az_config (Optional[Any]): The passed AlphaZero configuration, returned for convenience.
                                   This will be None if the initial AlphaZero component import failed.
  """
  # 1. Handle Missing AlphaZero Components (from initial import attempt)
  if AlphaZeroNet is None or mcts_module is None or utils_module is None or game_rules_module is None:
    print("AlphaZeroNet: Critical AlphaZero components (model, mcts, utils, game_rules) failed to import. AlphaZero AI is disabled.")
    return None, None, None, None, None # az_config is not returned as it's tied to these components

  # 2. Determine Device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 3. Initialize Model
  try:
    # Ensure az_config has necessary attributes before using them
    if not all(hasattr(az_config, attr) for attr in ['INPUT_CHANNELS', 'GRID_SIZE', 'BEST_MODEL_PATH']):
        print("AlphaZeroNet: az_config is missing one or more required attributes (INPUT_CHANNELS, GRID_SIZE, BEST_MODEL_PATH).")
        return None, mcts_module, utils_module, game_rules_module, az_config # Return modules as they might be used elsewhere
    if not hasattr(az_config.GRID_SIZE, 'rows') or not hasattr(az_config.GRID_SIZE, 'cols'):
        print("AlphaZeroNet: az_config.GRID_SIZE must have 'rows' and 'cols' attributes.")
        return None, mcts_module, utils_module, game_rules_module, az_config

    nn = AlphaZeroNet(az_config.INPUT_CHANNELS, az_config.GRID_SIZE.rows, az_config.GRID_SIZE.cols).to(device)
  except Exception as e:
    print(f"AlphaZeroNet: Error initializing AlphaZeroNet model: {e}")
    return None, mcts_module, utils_module, game_rules_module, az_config


  # 4. Load Model Weights
  model_path = az_config.BEST_MODEL_PATH
  if not model_path: # Check if BEST_MODEL_PATH is None or empty
      print("AlphaZeroNet: BEST_MODEL_PATH is not configured in az_config. Cannot load model.")
      # Depending on desired behavior, could return nn (untrained) or None
      return None, mcts_module, utils_module, game_rules_module, az_config

  try:
    nn.load_model(model_path, device) # Assumes load_model is a method of AlphaZeroNet
    print(f"AlphaZeroNet: Successfully loaded model from {model_path} to {device}")
  except FileNotFoundError:
    print(f"AlphaZeroNet: Model file not found at {model_path}. AI will not use AlphaZeroNet.")
    return None, mcts_module, utils_module, game_rules_module, az_config
  except Exception as e:
    print(f"AlphaZeroNet: Error loading model from {model_path}: {e}")
    return None, mcts_module, utils_module, game_rules_module, az_config

  # 5. Return Loaded Components
  return nn, mcts_module, utils_module, game_rules_module, az_config

# New main AI move function
def get_ai_player_move(game_state: GameState) -> Optional[Position]:
    """
    Determines the AI player's move, deciding between AlphaZero and random AI
    based on player configuration and loaded components.

    Args:
        game_state: The current state of the game.

    Returns:
        The selected Position for the AI's move, or None if a move
        cannot be determined or if it's not an AI player's turn.
    """
    if game_state.currentPlayerId is None:
        print("Error in get_ai_player_move: currentPlayerId is None.")
        return None

    current_player_info = next((p for p in game_state.players if p.id == game_state.currentPlayerId), None)

    if not current_player_info:
        print(f"Error in get_ai_player_move: Player info for ID {game_state.currentPlayerId} not found.")
        return None

    use_alphazero = False
    # Check for 'ai_type' first (preferred)
    if hasattr(current_player_info, 'ai_type') and current_player_info.ai_type == 'alphazero':
        if LOADED_AZ_MODEL and AZ_CONFIG_MODULE and LOADED_MCTS and LOADED_UTILS and LOADED_GAME_RULES:
            print(f"Player {current_player_info.id} is configured as 'alphazero'. Attempting AlphaZero move.")
            use_alphazero = True
        else:
            print(f"Player {current_player_info.id} is 'alphazero', but components are not loaded. Fallback needed.")
    # Fallback: if player isAI and AZ model is loaded, and no specific 'ai_type' denied it.
    elif current_player_info.isAI and LOADED_AZ_MODEL and AZ_CONFIG_MODULE and LOADED_MCTS and LOADED_UTILS and LOADED_GAME_RULES:
         print(f"Player {current_player_info.id} is AI, AlphaZero model available. Attempting AlphaZero move.")
         use_alphazero = True

    if use_alphazero:
        # Ensure all components are truly loaded one last time before calling
        if not (LOADED_AZ_MODEL and AZ_CONFIG_MODULE and LOADED_MCTS and LOADED_UTILS and LOADED_GAME_RULES):
            print(f"Critical AlphaZero components missing for player {current_player_info.id}. Falling back to random.")
            return get_random_ai_move(game_state)

        print(f"Using AlphaZero for player {current_player_info.id}")
        move = get_alphazero_move(game_state, LOADED_AZ_MODEL, AZ_CONFIG_MODULE, LOADED_MCTS, LOADED_UTILS, LOADED_GAME_RULES)
        if move is None:
            print(f"AlphaZero returned no move for player {current_player_info.id}, falling back to random.")
            return get_random_ai_move(game_state) # Fallback
        return move
    elif current_player_info.isAI: # If it's an AI but not using AlphaZero (or AZ failed to load)
        print(f"Using random AI for player {current_player_info.id}")
        return get_random_ai_move(game_state)
    else: # Not an AI player
        print(f"Warning: get_ai_player_move called for non-AI player {current_player_info.id}")
        return None

def get_alphazero_move(
    game_state: GameState, 
    nn: 'AlphaZeroNet', 
    config: Any,  # The main AlphaZero config, renamed from az_config
    mcts: Any, 
    utils: Any,
    game_rules: Any # Added game_rules to match signature
) -> Optional[Position]:
  """
  Determines the AI's move using the AlphaZero model and MCTS.

  Args:
    game_state: The current state of the game.
    nn: The loaded AlphaZeroNet model.
    config: The AlphaZero configuration module/object.
    mcts: The MCTS module.
    utils: The utility module.
    game_rules: The game rules module (not directly used here but part of signature).

  Returns:
    The selected Position for the AI's move, or None if a move
    cannot be determined.
  """
  try:
    # 1. Set the neural network to evaluation mode
    nn.eval()

    # 2. Get the MCTS policy vector
    # This policy_vector is assumed to be a list/array of probabilities for all actions.
    policy_vector = mcts.get_policy_for_state(nn, game_state, config) # Use 'config'

    if policy_vector is None:
      print("get_alphazero_move: MCTS returned no policy vector. Falling back.")
      return None

    # 3. Select the move deterministically using MCTS
    # For deterministic selection, temperature should be very low (e.g., 0.01).
    # We create a temporary config-like object for this purpose.
    # The MCTS's select_move function is assumed to call a get_temperature function
    # that can use this temporary object.
    DeterministicConfig = collections.namedtuple(
        'DeterministicConfig', 
        ['INITIAL_TEMPERATURE', 'TEMPERATURE_DECAY_TURN', 'FINAL_TEMPERATURE']
    )
    # Using config.TEMPERATURE_DECAY_TURN (if it were used by a general get_temperature)
    # Forcing deterministic by setting low values for initial and final.
    deterministic_temp_params = DeterministicConfig(
        INITIAL_TEMPERATURE=0.01, 
        TEMPERATURE_DECAY_TURN=0, # Ensures that effectively FINAL_TEMPERATURE is used if logic relies on this.
        FINAL_TEMPERATURE=0.01
    )
    
    move_index = mcts.select_move(policy_vector, game_state.turnNumber, deterministic_temp_params)

    # 4. Convert move index to position
    # Use game_state.gridConfiguration.cols as established previously
    num_cols = game_state.gridConfiguration.cols
    move_position = utils.convert_move_index_to_position(move_index, num_cols)

    # 5. Return move_position
    return move_position

  except Exception as e:
    print(f"get_alphazero_move: An error occurred - {e}. Falling back.")
    # Depending on the desired robustness, you might want to log the error
    # or return a default move (like random) instead of None.
    # For now, returning None indicates failure to produce an AlphaZero move.
    return None


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

# --- Module-Level Initialization ---
# Attempt to import the AlphaZero config
AZ_CONFIG_MODULE = None
try:
    from .alphazero import config as alphazero_main_config
    AZ_CONFIG_MODULE = alphazero_main_config
    print("AlphaZero: Successfully imported AlphaZero config.")
except ImportError:
    print("AlphaZero: Failed to import AlphaZero config. AlphaZero AI will be disabled.")
except Exception as e: # Catch other potential errors during config import
    print(f"AlphaZero: An unexpected error occurred during AlphaZero config import: {e}. AlphaZero AI will be disabled.")


# Load AlphaZero model and dependencies using the function already defined in this file
if AZ_CONFIG_MODULE:
    LOADED_AZ_MODEL, LOADED_MCTS, LOADED_UTILS, LOADED_GAME_RULES, _ = load_alphazero_model_and_dependencies(AZ_CONFIG_MODULE)
    # We discard the returned config as AZ_CONFIG_MODULE is already the one we want to use.
else:
    LOADED_AZ_MODEL, LOADED_MCTS, LOADED_UTILS, LOADED_GAME_RULES = None, None, None, None

if LOADED_AZ_MODEL:
    print("AlphaZero: Model and dependencies loaded successfully for use in bot.")
else:
    print("AlphaZero: Model and/or dependencies NOT loaded. AlphaZero AI will be disabled or use fallback.")
