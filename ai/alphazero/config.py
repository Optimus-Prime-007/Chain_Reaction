# MCTS Hyperparameters
C_PUCT: float = 1.0  # Exploration constant for PUCT
NUM_MCTS_SIMULATIONS: int = 800  # Number of MCTS simulations per move

# Add any other game-specific or NN-specific configs here if known,
# but for now, just these two for MCTS.

# Example of other possible configs (can be added later):
# GRID_ROWS: int = 6
# GRID_COLS: int = 7
# LEARNING_RATE: float = 0.001
# BATCH_SIZE: int = 64
