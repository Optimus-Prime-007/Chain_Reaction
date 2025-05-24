from typing import List, Any, Optional # Ensured Optional is imported

# Assuming models are in ai.models or accessible via the parent directory
try:
    from ..models import GridSizeConfig, PlayerInfo
except ImportError:
    # Fallback for environments where the relative import might not work as expected
    # This might happen if the script is run directly for testing
    from ai.models import GridSizeConfig, PlayerInfo

# MCTS and Self-Play Configuration Parameters
MCTS_SIMULATIONS: int = 100  # Number of MCTS simulations per move
MCTS_SIMULATIONS_EVAL: int = 50  # Number of MCTS simulations to use during evaluation games.
C_PUCT: float = 1.41  # Exploration constant in UCB1 formula for MCTS (sqrt(2) is common)

# Temperature settings for move selection in MCTS
INITIAL_TEMPERATURE: float = 1.0  # Initial exploration temperature
TEMPERATURE_DECAY_TURN: int = 15  # Turn number after which temperature reduces
FINAL_TEMPERATURE: float = 0.1  # Temperature after decay, for more deterministic moves

# --- Neural Network Configuration ---
INPUT_CHANNELS: int = 5  # Number of channels for the NN input (e.g., from encode_state)

# --- Game-Specific Configuration ---
GRID_SIZE: GridSizeConfig = GridSizeConfig(rows=6, cols=6)  # Default grid size
GRID_ROWS: int = GRID_SIZE.rows # For direct access in some modules, derived from GRID_SIZE.
GRID_COLS: int = GRID_SIZE.cols # For direct access in some modules, derived from GRID_SIZE.

# Derived Configurations (should be defined after their dependencies)
NUM_ACTIONS: int = GRID_SIZE.rows * GRID_SIZE.cols  # Total number of possible actions/cells

# --- Training Loop Configuration ---
LEARNING_RATE: float = 0.001
REPLAY_BUFFER_SIZE: int = 50000  # Max number of game states (or full games) to store
MIN_BUFFER_FOR_TRAINING: int = 1000 # Minimum examples in buffer before training starts
BATCH_SIZE: int = 64  # Batch size for training the neural network
NUM_ITERATIONS: int = 1000  # Total number of training iterations
NUM_SELF_PLAY_GAMES_PER_ITER: int = 50  # Number of self-play games per iteration
NUM_EPOCHS_PER_ITERATION: int = 10  # Number of training epochs on sampled data per iteration
VALUE_LOSS_WEIGHT: float = 1.0  # Weight for the value loss component in total loss

# --- Checkpoint/Saving Configuration ---
# CHECKPOINT_SAVE_PATH_FORMAT is used by the main_training_loop to save iteration checkpoints
CHECKPOINT_SAVE_PATH_FORMAT: str = "ai/trained_models/checkpoints/alphazero_iter_{iteration}.pth" # Updated path
# Set to a specific path like "ai/trained_models/checkpoints/alphazero_iter_X.pth" to resume training
LOAD_CHECKPOINT_PATH: Optional[str] = None 
BEST_MODEL_PATH: str = "ai/trained_models/best_model.pth"  # Path to the chosen production model (updated path)
CHECKPOINT_DIR: str = "ai/trained_models/checkpoints" # Directory for periodic checkpoints
SAVE_CHECKPOINT_EVERY_N_ITERATIONS: int = 10 # Save a checkpoint every N training iterations

# --- Logging and Monitoring ---
LOG_DIR: str = "ai/logs"  # Base directory for all logs
TENSORBOARD_DIR: str = f"{LOG_DIR}/tensorboard" # For TensorBoard event files
TRAINING_LOG_FILE: str = f"{LOG_DIR}/training.log" # For general text logs
LOG_LEVEL: str = "INFO" # Default logging level (DEBUG, INFO, WARNING, ERROR)

# --- Evaluation ---
EVALUATION_ENABLED: bool = True # Master switch for evaluation phase
EVALUATE_EVERY_N_ITERATIONS: int = 20 # Run evaluation every N training iterations
EVALUATION_GAMES_PER_OPPONENT: int = 50 # Number of games to play against each opponent
EVALUATION_OPPONENTS: List[dict] = [ # List of opponents to evaluate against
    {"type": "random", "name": "RandomBot"},
    # Example: {"type": "heuristic", "name": "HeuristicBot", "path": "path/to/heuristic_logic.py"},
    {"type": "previous_best", "name": "PreviousBest"}, # Special keyword for current best_model.pth
    # Example: {"type": "specific_checkpoint", "name": "Checkpoint_Iter50", "path": f"{CHECKPOINT_DIR}/alphazero_iter_050.pth"}
]
WIN_RATIO_THRESHOLD_FOR_BEST_MODEL: float = 0.55 # New model needs to win >55% to become the new best


# --- Player Information (Example for 2 players) ---
# (Used for UI or if game logic needs player-specific details beyond ID)
# These colors are just examples and might be used by a UI.
# The core game logic typically only uses PlayerId.
PLAYERS_INFO: List[PlayerInfo] = [
    PlayerInfo(id=1, name="Player 1", colorClass="bg-rose-600", textColorClass="text-rose-500", orbColorClass="bg-rose-400", primaryHex="#E11D48"),
    PlayerInfo(id=2, name="Player 2", colorClass="bg-sky-600", textColorClass="text-sky-500", orbColorClass="bg-sky-400", primaryHex="#0284C7")
]

# Function to calculate temperature for MCTS move selection
def get_temperature(turn_number: int, config: Any) -> float:
    """
    Calculates the temperature for move selection based on the turn number.
    Temperature is higher initially for exploration, lower later for exploitation.

    Args:
        turn_number: The current turn number in the game.
        config: The configuration module itself (to access its variables like
                TEMPERATURE_DECAY_TURN, INITIAL_TEMPERATURE, FINAL_TEMPERATURE).
                This allows the function to be part of the config module and still
                access its own variables, which is a common pattern if this function
                were, for example, called from outside with `config.get_temperature(turn, config)`.

    Returns:
        The temperature value.
    """
    if turn_number < config.TEMPERATURE_DECAY_TURN:
        return config.INITIAL_TEMPERATURE
    else:
        return config.FINAL_TEMPERATURE

# Example of how to potentially use this config module (not part of the config itself)
# if __name__ == '__main__':
#     print(f"MCTS Simulations: {MCTS_SIMULATIONS}")
#     print(f"Grid Size: {GRID_SIZE.rows}x{GRID_SIZE.cols}")
#     print(f"Number of Actions: {NUM_ACTIONS}")
#     print(f"Player 1 Info: {PLAYERS_INFO[0]}")
#
#     # To use get_temperature, you'd typically pass the module itself if calling from outside
#     # or directly access variables if calling from within a context where 'config' is this module.
#     # For example, if another module `game_logic.py` imported `config`, it might do:
#     # current_temp = config.get_temperature(turn_number=5, config=config)
#     # print(f"Temperature at turn 5: {current_temp}")
#     # current_temp = config.get_temperature(turn_number=20, config=config)
#     # print(f"Temperature at turn 20: {current_temp}")
