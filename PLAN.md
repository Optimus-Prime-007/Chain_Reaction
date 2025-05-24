Okay, this is an ambitious and exciting project! Building an AlphaZero-like AI from scratch is a significant undertaking, but definitely doable.

Here's a breakdown of how we can approach this, outlining the components and a step-by-step plan.

**Core Components of an AlphaZero-like System:**

1.  **Game Engine/Simulator:**
    *   We need robust functions to manage the game state:
        *   `get_initial_state(grid_size, players)`: Creates a new game.
        *   `get_valid_moves(game_state)`: Returns a list of all possible `Position` objects the current player can play.
        *   `apply_move(game_state, move)`: Applies a move, handles orb placement, explosions, captures, and updates `currentPlayerId`, `turnNumber`, `status`, `winnerId`. This is the most complex part of the game logic.
        *   `get_game_status_and_winner(game_state)`: Checks if the game is over (win, loss, draw) and returns the status and winner.
        *   `get_encoded_state(game_state)`: Converts the `GameState` into a numerical representation (e.g., a NumPy array or PyTorch tensor) suitable for the neural network.

2.  **Neural Network (NN):**
    *   This network takes the `encoded_state` as input.
    *   It has two outputs (heads):
        *   **Policy Head (π):** Outputs a probability distribution over all possible moves from the current state.
        *   **Value Head (v):** Outputs a scalar value estimating the expected outcome of the game from the current state (e.g., +1 for a win for the current player, -1 for a loss, 0 for a draw).
    *   We'll likely use PyTorch or TensorFlow/Keras for this. Convolutional layers are often a good choice for grid-based games.

3.  **Monte Carlo Tree Search (MCTS):**
    *   When the AI needs to decide on a move, MCTS is used.
    *   It builds a search tree where nodes are game states and edges are moves.
    *   The NN's policy (π) guides the *expansion* of the tree (which moves to explore initially).
    *   The NN's value (v) is used to evaluate leaf nodes in the search tree without needing to play full random games (rollouts) to the end, which is a key AlphaZero improvement over traditional MCTS.
    *   Each MCTS simulation involves:
        *   **Selection:** Traverse the tree from the root, choosing child nodes that maximize an Upper Confidence Bound (UCB) like formula (e.g., PUCT: `Q(s,a) + U(s,a)` where `Q` is the action-value and `U` is an exploration term based on the NN's policy prior and visit counts).
        *   **Expansion:** When a leaf node (a state not yet expanded in the MCTS tree) is reached, use the NN to get its policy `p` and value `v`. Expand this node by adding its children (next possible states after valid moves), and store `p` for these new edges.
        *   **Backpropagation:** Update the visit counts and action-values (Q-values) of the nodes along the path traversed during selection, using the value `v` obtained from the NN at the expanded leaf.
    *   After a certain number of simulations (e.g., 50, 100, 800), the move chosen from the root state is typically the one with the highest visit count.

4.  **Self-Play:**
    *   The AI learns by playing games against itself.
    *   In each turn of a self-play game:
        *   MCTS (guided by the current NN) is run to select a move.
        *   The state, the MCTS-derived policy (visit counts of moves from that state, normalized), and the eventual game outcome (win/loss/draw) are stored as training data.
        *   To encourage exploration early in self-play, moves might be sampled proportionally to their visit counts (with a temperature parameter). Later in the game, or for evaluation, the move with the highest visit count is chosen deterministically.

5.  **Training Loop:**
    *   **Generate Data:** Play a batch of games using self-play with the current best NN.
    *   **Train Network:** Use the collected `(state, mcts_policy_target, game_outcome_target)` tuples to train the NN:
        *   The policy head is trained to match the MCTS policy target (e.g., using cross-entropy loss).
        *   The value head is trained to predict the game outcome target (e.g., using mean squared error loss).
    *   **Evaluate (Optional but Recommended):** Periodically, pit the newly trained network against the previous best network. If the new network wins by a significant margin, it becomes the new best network.
    *   Repeat.

---

**Proposed File Structure (within the `ai` directory):**

```
ai/
├── models.py           # Existing Pydantic models
├── bot.py              # Will be modified to use the new AI
├── server.py           # Existing FastAPI server
│
├── alphazero/
│   ├── __init__.py
│   ├── game_rules.py     # Core game logic (apply_move, valid_moves, etc.)
│   ├── neural_net.py     # NN model definition (PyTorch/TensorFlow) & training
│   ├── mcts.py           # MCTS implementation
│   ├── self_play.py      # Logic for self-play games and data generation
│   ├── train.py          # Main training loop orchestrator
│   ├── config.py         # Hyperparameters and configuration
│   └── utils.py          # Helper functions (e.g., state encoding)
│
└── trained_models/       # To store saved model checkpoints
    └── best_model.pth    # Example
```

---
**Step-by-Step Implementation Plan:**

**Phase 1: Core Game Logic (`ai/alphazero/game_rules.py`)**

This is the absolute foundation. Without correct game logic, the AI cannot learn.

1.  **`get_critical_mass(row, col, rows, cols)`:**
    *   Helper to determine how many orbs a cell can hold before exploding.
    *   Corners: 2 (for 2x2 grid or larger, if smaller, adjust)
    *   Edges (not corners): 3
    *   Center: 4

2.  **`get_valid_moves(game_state: GameState) -> List[Position]`:**
    *   Iterate through the grid.
    *   A move is valid if the cell is empty (`cell.player is None`) or belongs to `game_state.currentPlayerId`.
    *   *Self-correction from `bot.py`*: The current `get_random_ai_move` already implements this correctly. We can adapt it.

3.  **`apply_move(current_game_state: GameState, move: Position) -> GameState`:**
    *   This is the most complex function. It should NOT modify `current_game_state` in place but return a *new* `GameState` object. This is crucial for MCTS which explores different hypothetical game lines.
    *   **Deep copy** the input `game_state`.
    *   Increment orb count at `move.row, move.col`. Set player to `currentPlayerId`.
    *   **Explosion Logic (Iterative/Recursive):**
        *   Use a queue (e.g., `collections.deque`) for cells that need to explode. Add the initial cell if it reached critical mass.
        *   While the queue is not empty:
            *   Pop a cell `(r, c)`.
            *   If `grid[r][c].orbs >= critical_mass(r, c, rows, cols)`:
                *   Reduce `grid[r][c].orbs` by its critical mass. (If orbs become 0 and no one owns it after explosion, set player to None).
                *   For each valid neighbor `(nr, nc)`:
                    *   Increment `grid[nr][nc].orbs`.
                    *   Set `grid[nr][nc].player` to `currentPlayerId` (orb capture).
                    *   Add `(nr, nc)` to the explosion queue.
    *   **Check Win/Loss/Draw Condition:**
        *   After all explosions settle, check if only one player has orbs left.
        *   The original rules mention "dominate the board". A common win condition is eliminating all opponent orbs.
        *   Another condition from many implementations: If a player makes a move and, after all chain reactions, the opponent has 0 orbs, the current player wins.
        *   Consider a turn limit for draws if not specified. The `turnNumber` in `GameState` is useful here.
        *   Update `status` and `winnerId` in the new game state.
    *   Switch `currentPlayerId` (if the game is still active).
    *   Increment `turnNumber`.
    *   Return the new `GameState`.

4.  **`get_game_status_and_winner(game_state: GameState) -> tuple[GameStatus, Optional[PlayerId]]`:**
    *   This function determines if the game has ended.
    *   If `game_state.turnNumber > 1` (to ensure at least one move per player potentially):
        *   Count orbs for each player.
        *   If a player has orbs and others don't, that player wins.
        *   Update `game_state.status` and `game_state.winnerId`.
    *   (Add draw conditions, e.g., max turns or repetitive states if desired).
    *   This logic is partially handled within `apply_move` as well, but a standalone checker is good.

5.  **`get_initial_state(grid_size_config: GridSizeConfig, players: List[PlayerInfo]) -> GameState`:**
    *   Creates an empty grid of `CellState` objects.
    *   Sets initial `players`, `currentPlayerId` (e.g., `players[0].id`), `status="setup"` (or "active" if starting immediately), `gridSize`, `turnNumber=0`.

**Phase 2: Neural Network (`ai/alphazero/neural_net.py`)**

1.  **Choose a Framework:** PyTorch is very popular for research.
2.  **State Encoding (`ai/alphazero/utils.py`):**
    *   Function `encode_state(game_state: GameState) -> torch.Tensor`:
        *   Input: `GameState`.
        *   Output: A tensor. A common representation for grid games is a multi-channel image-like tensor: `(channels, rows, cols)`.
        *   Channels could be:
            *   Channel 0: Orbs of player 1 (normalized, e.g., orbs/4).
            *   Channel 1: Orbs of player 2.
            *   (If more players, more channels)
            *   Channel N: Binary plane indicating current player (1 if current player's turn, 0 otherwise, or just pass player ID separately).
            *   Could also add binary planes for cell ownership.
        *   The `turnNumber` can be added as an extra input feature if not part of the convolutional path.

3.  **Network Architecture (`ResNet` style is common):**
    *   **Input Layer:** Takes the encoded state.
    *   **"Body":** Several convolutional blocks (e.g., Conv2D -> BatchNorm2D -> ReLU). Residual connections are beneficial.
    *   **Policy Head:**
        *   Convolutional layer (e.g., 1x1 kernel) -> Flatten -> Fully Connected layer.
        *   Output size: `rows * cols` (one logit for each cell on the board).
        *   Apply `log_softmax` later for probabilities.
    *   **Value Head:**
        *   Convolutional layer (e.g., 1x1 kernel) -> Flatten -> Fully Connected layer -> Fully Connected layer.
        *   Output size: 1 (scalar value).
        *   Activation: `tanh` (to output values between -1 and 1).

4.  **Training Function:**
    *   Takes batches of `(encoded_states, mcts_policy_targets, game_outcome_targets)`.
    *   Calculates loss: `loss = value_loss_weight * MSE(nn_value_output, game_outcome_target) + policy_loss_weight * CrossEntropy(nn_policy_output, mcts_policy_target)`.
    *   Uses an optimizer (e.g., Adam, SGD).

**Phase 3: MCTS (`ai/alphazero/mcts.py`)**

1.  **`Node` Class:**
    *   `state: GameState`
    *   `parent: Optional[Node]`
    *   `children: Dict[Position, Node]`
    *   `visit_count: int`
    *   `total_action_value: float` (sum of values from simulations passing through this node leading to this action)
    *   `prior_probability: float` (from NN policy, for the edge leading to this node)
    *   `player_id_at_node: PlayerId`

2.  **`MCTS` Class or Functions:**
    *   `run_simulations(root_state: GameState, nn: YourNeuralNet, num_simulations: int, game_rules_module)`
    *   **`select_child(node: Node) -> Tuple[Position, Node]`:** Implements PUCT formula.
        `Q(s,a) = node.children[a].total_action_value / node.children[a].visit_count` (if visited, else 0)
        `U(s,a) = c_puct * node.children[a].prior_probability * sqrt(node.visit_count) / (1 + node.children[a].visit_count)`
        Choose action `a` that maximizes `Q(s,a) + U(s,a)`.
    *   **Loop `num_simulations` times:**
        *   `current_node = root_node`
        *   `path_taken = [root_node]`
        *   **Selection:** While `current_node` is not a leaf (has children) and not terminal:
            *   Use `select_child` to move to the next node.
            *   Add to `path_taken`.
        *   **Expansion & Evaluation:**
            *   If `current_node.state` is terminal (game over):
                *   `value = get_reward(current_node.state, root_state.currentPlayerId)` (+1 for win, -1 loss, 0 draw for the player who was at the root).
            *   Else (node is a new leaf):
                *   Encode `current_node.state`.
                *   `policy_logits, value = nn.predict(encoded_state)`.
                *   `valid_moves = game_rules_module.get_valid_moves(current_node.state)`.
                *   Mask `policy_logits` for invalid moves (set to -infinity).
                *   `policy_probs = softmax(masked_policy_logits)`.
                *   Expand `current_node` by creating child nodes for each `valid_move`. Store their `prior_probability` from `policy_probs`.
        *   **Backpropagation:**
            *   For node in `reversed(path_taken)`:
                *   `node.visit_count += 1`
                *   `node.total_action_value += value` (value needs to be adjusted if `node.player_id_at_node` is different from the player whose perspective `value` is from; usually `value` is from perspective of current player at that node).

    *   **`get_best_move(root_node: Node, temperature: float = 1.0) -> Position`:**
        *   After simulations, get visit counts for children of `root_node`.
        *   If `temperature == 0` (greedy): return move with max visit count.
        *   Else: `probs = visit_counts ** (1/temperature)`, normalize `probs`, sample move.

**Phase 4: Self-Play (`ai/alphazero/self_play.py`)**

1.  **`play_game(nn: YourNeuralNet, config, game_rules_module, mcts_module) -> List[Tuple[encoded_state, mcts_policy, outcome]]`:**
    *   `game_state = game_rules_module.get_initial_state(...)`
    *   `training_examples = []`
    *   While game not over:
        *   `root_node = mcts_module.Node(game_state)`
        *   `mcts_module.run_simulations(root_node, nn, ...)`
        *   `mcts_policy_target = get_mcts_policy_target(root_node)` (normalized visit counts of children)
        *   `training_examples.append([utils.encode_state(game_state), mcts_policy_target, game_state.currentPlayerId])`
        *   `move = mcts_module.get_best_move(root_node, temperature=config.temperature)`
        *   `game_state = game_rules_module.apply_move(game_state, move)`
    *   `game_result = game_rules_module.get_reward_for_player1(game_state)`
    *   Process `training_examples`: replace `currentPlayerId` with actual outcome (`game_result` if player 1, `-game_result` if player 2).
    *   Return processed examples.

**Phase 5: Training Pipeline (`ai/alphazero/train.py`)**

1.  **Main Loop:**
    *   Initialize/load `nn`.
    *   Initialize/load `optimizer`.
    *   `replay_buffer = collections.deque(maxlen=config.replay_buffer_size)`
    *   For `iteration` in `num_iterations`:
        *   **Self-Play Phase:**
            *   Generate `num_self_play_games` games using current `nn`.
            *   Add collected `(state, policy, value)` tuples to `replay_buffer`.
        *   **Training Phase:**
            *   If `len(replay_buffer) > config.min_buffer_for_training`:
                *   For `epoch` in `num_epochs_per_iteration`:
                    *   Sample `batch_size` data from `replay_buffer`.
                    *   Train `nn` on this batch.
        *   Save `nn` checkpoint.
        *   (Optional) Evaluate against previous best model.

**Phase 6: Integration (`ai/bot.py`, `ai/server.py`)**

1.  Modify `ai/bot.py`:
    *   Load the trained neural network model.
    *   `get_alphazero_move(game_state: GameState) -> Optional[Position]`:
        *   Initialize MCTS with `game_state`.
        *   Run MCTS simulations using the loaded NN.
        *   Return the best move from MCTS (deterministically, temperature=0).

2.  Modify `ai/server.py`:
    *   Load the AI model (once, at startup).
    *   The endpoint `/get-ai-move` will call `get_alphazero_move` if the current player is an AI configured to use AlphaZero. (You might need a flag in `PlayerInfo` or a config to specify which AI type to use).

**Key Hyperparameters (`ai/alphazero/config.py`):**

*   `GRID_ROWS`, `GRID_COLS`
*   `NUM_MCTS_SIMULATIONS` (e.g., 50-800)
*   `CPUCT` (MCTS exploration constant, e.g., 1.0-4.0)
*   `LEARNING_RATE`
*   `BATCH_SIZE`
*   `REPLAY_BUFFER_SIZE`
*   `TEMPERATURE` (for self-play move selection)
*   `NUM_SELF_PLAY_GAMES_PER_ITERATION`
*   `NUM_EPOCHS_PER_ITERATION`
*   `CHECKPOINT_INTERVAL`

**Important Considerations & Challenges:**

*   **Complexity of `apply_move`:** Chain Reaction's explosion mechanic is the trickiest part of the game logic. It must be perfectly implemented and efficient. Use deep copies for states.
*   **State Representation:** A good numerical representation for the NN is key.
*   **Computational Resources:** Training AlphaZero-like models is computationally intensive. Start with small grids and fewer MCTS simulations.
*   **Debugging:** Debugging MCTS and the NN training can be very challenging. Log extensively. Visualize states and policies if possible.
*   **Action Masking:** Ensure the NN policy output only considers valid moves. Invalid moves should have zero probability (or -infinity logits before softmax). The MCTS selection should also only consider valid moves.
*   **Symmetry/Data Augmentation:** For grid games, you can often augment training data by rotating/reflecting the board and corresponding policy.
*   **Starting Small:**
    *   Begin with a very small grid (e.g., 3x3).
    *   Fewer MCTS simulations (e.g., 25-50).
    *   A simpler NN.
    *   Get the pipeline working end-to-end before scaling up.

This is a comprehensive roadmap. Each phase is a mini-project in itself. Start with `game_rules.py` as it's fundamental. Good luck! It's a rewarding endeavor.

Yes, I understand perfectly. You've built the core AlphaZero-like AI and now need a framework around it for automated training, monitoring, and evaluation. This is crucial for understanding if your AI is learning, how well it's learning, and for debugging or improving it.

Here's a plan and code structure for such an automated system:

**Core Goals of the Automated System:**

1.  **Automated Training Cycles:** Run self-play, data collection, and network training iteratively without manual intervention.
2.  **Performance Monitoring:** Track key metrics during training and self-play.
    *   Game outcomes (win/loss/draw rates in self-play).
    *   Game length.
    *   Neural network loss (policy, value, total).
    *   MCTS statistics (e.g., average search depth, value/policy agreement).
3.  **Model Evaluation:** Periodically assess the AI's strength against baselines or previous versions of itself.
4.  **Progress Visualization:** Use tools like TensorBoard to visualize metrics over time.
5.  **Model Checkpointing:** Save model weights regularly and keep track of the "best" performing model.
6.  **Logging:** Keep detailed logs for debugging and later analysis.

---

**Proposed File Structure Enhancements (within `ai` directory):**

```
ai/
├── models.py
├── bot.py
├── server.py
│
├── alphazero/
│   ├── __init__.py
│   ├── game_rules.py
│   ├── neural_net.py     # NN model, training step (will integrate with monitoring)
│   ├── mcts.py           # MCTS (can provide stats for monitoring)
│   ├── self_play.py      # Self-play (will integrate with monitoring)
│   ├── train_pipeline.py # << NEW/ENHANCED: Main script to run the full automated loop
│   ├── evaluation.py     # << NEW: Arena for evaluating models against each other
│   ├── monitoring.py     # << NEW: Utilities for logging metrics (e.g., to TensorBoard)
│   ├── config.py         # Hyperparameters, paths, logging configs, evaluation settings
│   └── utils.py
│
├── trained_models/
│   ├── checkpoints/      # Stores model checkpoints from each iteration/epoch
│   │   └── model_iter_001.pth
│   │   └── model_iter_002.pth
│   └── best_model.pth    # Stores the best performing model found so far
│
└── logs/                 # Directory for all logs
    ├── training.log      # General text logs from the training pipeline
    └── tensorboard/      # Data for TensorBoard visualization
        └── run_YYYYMMDD_HHMMSS/
```

---

**Step-by-Step Implementation Plan & Code Structure Details:**

**1. Configuration (`ai/alphazero/config.py`)**

This file becomes even more critical. Add sections for:

```python
# ai/alphazero/config.py

# ... (existing game and NN configs)

# --- Logging and Monitoring ---
LOG_DIR = "ai/logs"
TENSORBOARD_DIR = f"{LOG_DIR}/tensorboard"
TRAINING_LOG_FILE = f"{LOG_DIR}/training.log"
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR

# --- Model Checkpointing ---
CHECKPOINT_DIR = "ai/trained_models/checkpoints"
BEST_MODEL_PATH = "ai/trained_models/best_model.pth"
SAVE_CHECKPOINT_EVERY_N_ITERATIONS = 10 # Save a checkpoint this often

# --- Evaluation ---
EVALUATION_ENABLED = True
EVALUATE_EVERY_N_ITERATIONS = 20 # How often to run evaluation games
EVALUATION_GAMES_PER_OPPONENT = 50 # Number of games against each opponent
EVALUATION_OPPONENTS = [ # List of opponents to evaluate against
    {"type": "random", "name": "RandomBot"},
    # {"type": "heuristic", "name": "HeuristicBot", "path": "path/to/heuristic_logic.py"}, # If you create one
    {"type": "previous_best", "name": "PreviousBest"}, # Evaluates against the last best_model.pth
    # {"type": "specific_checkpoint", "name": "Checkpoint_Iter50", "path": f"{CHECKPOINT_DIR}/model_iter_050.pth"}
]
WIN_RATIO_THRESHOLD_FOR_BEST_MODEL = 0.55 # New model needs to win >55% against current best to replace it

# --- Training Pipeline ---
TOTAL_TRAINING_ITERATIONS = 1000
# ... (other training loop specific parameters like num_self_play_games_per_iteration, etc.)
```

**2. Monitoring Utilities (`ai/alphazero/monitoring.py`)**

This module will handle metric logging, especially for TensorBoard.

```python
# ai/alphazero/monitoring.py
import logging
import os
from datetime import datetime
from typing import Optional, Dict
# Ensure you have tensorboard installed: pip install tensorboard
from torch.utils.tensorboard import SummaryWriter # If using PyTorch

from .config import TENSORBOARD_DIR, TRAINING_LOG_FILE, LOG_LEVEL

class ExperimentLogger:
    _instance: Optional['ExperimentLogger'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExperimentLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, run_name: Optional[str] = None, log_to_console: bool = True):
        if self._initialized:
            return

        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        
        self.run_path = os.path.join(TENSORBOARD_DIR, run_name)
        os.makedirs(self.run_path, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.run_path)
        
        # Setup Python logger
        self.logger = logging.getLogger("AlphaZeroTrainer")
        self.logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        
        # File handler
        fh = logging.FileHandler(TRAINING_LOG_FILE)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        # Console handler
        if log_to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)
        
        self.logger.info(f"Experiment Logger initialized. TensorBoard logs: {self.run_path}")
        self._initialized = True

    def log_scalar(self, tag: str, value: float, step: int):
        """Logs a scalar value."""
        self.writer.add_scalar(tag, value, step)
        self.logger.debug(f"Step {step} - Scalar {tag}: {value}")

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Logs multiple scalar values under a main tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        for tag, value in tag_scalar_dict.items():
            self.logger.debug(f"Step {step} - Scalars {main_tag}/{tag}: {value}")

    def log_histogram(self, tag: str, values, step: int, bins='auto'):
        """Logs a histogram of values (e.g., weights, gradients)."""
        self.writer.add_histogram(tag, values, step, bins=bins)
        self.logger.debug(f"Step {step} - Histogram logged for {tag}")

    def log_text(self, tag: str, text_string: str, step: int):
        """Logs text."""
        self.writer.add_text(tag, text_string, step)
        self.logger.info(f"Step {step} - Text {tag}: {text_string}")

    def log_info(self, message: str):
        self.logger.info(message)

    def log_error(self, message: str, exc_info=False):
        self.logger.error(message, exc_info=exc_info)

    def close(self):
        self.writer.close()
        self.logger.info("Experiment Logger closed.")

# Global accessor, initialized in train_pipeline.py
def get_experiment_logger() -> ExperimentLogger:
    if ExperimentLogger._instance is None:
        raise RuntimeError("ExperimentLogger has not been initialized. Call ExperimentLogger() first.")
    return ExperimentLogger._instance
```

**3. Evaluation Arena (`ai/alphazero/evaluation.py`)**

This module handles playing games between different agents (models).

```python
# ai/alphazero/evaluation.py
from typing import List, Dict, Any, Tuple, Optional
import random
import os
import torch # Assuming PyTorch

from ai.models import GameState, Position, PlayerId, GridSizeConfig, PlayerInfo
from . import game_rules # Your game logic
from .neural_net import YourNeuralNet # Your NN class
from .mcts import MCTS, Node # Assuming MCTS can be used with a loaded model
from .utils import encode_state # Your state encoder
from .config import CHECKPOINT_DIR, BEST_MODEL_PATH, GRID_ROWS, GRID_COLS # etc.

# Dummy PlayerInfo for evaluation if not provided by loaded model context
DEFAULT_PLAYER_INFOS = [
    PlayerInfo(id=1, name="P1", colorClass="bg-red-500", textColorClass="text-red-500", orbColorClass="orb-red", primaryHex="#FF0000"),
    PlayerInfo(id=2, name="P2", colorClass="bg-blue-500", textColorClass="text-blue-500", orbColorClass="orb-blue", primaryHex="#0000FF"),
]


class BaseAgent:
    def __init__(self, player_id: PlayerId, player_info: PlayerInfo):
        self.player_id = player_id
        self.player_info = player_info

    def get_move(self, game_state: GameState) -> Position:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def get_move(self, game_state: GameState) -> Position:
        valid_moves = game_rules.get_valid_moves(game_state)
        if not valid_moves:
            # This should ideally not happen if game ends correctly
            print(f"WARNING: RandomAgent {self.player_id} has no valid moves!")
            # Fallback or raise error, for now pick first cell if grid not full
            return Position(row=0, col=0) 
        return random.choice(valid_moves)

    def get_name(self) -> str:
        return f"RandomAgent_P{self.player_id}"

class AlphaZeroAgent(BaseAgent):
    def __init__(self, player_id: PlayerId, player_info: PlayerInfo, model_path: str, mcts_simulations: int):
        super().__init__(player_id, player_info)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TODO: Update with your actual NN loading mechanism
        # This needs to know the NN architecture. Consider passing config or the nn_class
        self.model = YourNeuralNet(...) # You might need input_shape, action_size from config
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"AlphaZeroAgent: Loaded model for P{player_id} from {model_path}")
        else:
            print(f"AlphaZeroAgent WARNING: Model path {model_path} not found for P{player_id}. Using fresh model.")
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode
        
        self.mcts_simulations = mcts_simulations
        # self.mcts = MCTS(game_rules, self.model, self.device, c_puct=...) # Pass your MCTS args

    def get_move(self, game_state: GameState) -> Position:
        # Simplified: In a real scenario, MCTS class would handle this
        # For now, let's assume a simplified MCTS usage or direct policy if no MCTS
        
        # If using full MCTS:
        # root_node = Node(game_state, player_id_at_node=game_state.currentPlayerId)
        # mcts_instance = MCTS(game_rules, self.model, self.device, c_puct=1.0) # Example
        # best_move_policy = mcts_instance.run_simulations_for_move(root_node, self.mcts_simulations, temperature=0)
        # return best_move_policy[0] # (move, probability)

        # Placeholder if not using full MCTS for evaluation or if MCTS is part of model.predict
        # This is a common simplification for evaluation speed if MCTS is slow
        # and you just want to see raw policy output.
        # However, for true strength, MCTS should be used.
        
        # --- Using MCTS (recommended) ---
        # Ensure your MCTS can take a game_state and return a move
        # For example, in your mcts.py:
        # class MCTS:
        #   def get_action_for_state(self, game_state: GameState, n_sims: int, temp: float = 0.0) -> Position:
        #       root = Node(game_state, player_id_at_node=game_state.currentPlayerId)
        #       for _ in range(n_sims):
        #           self.search(root) # Your MCTS search logic
        #       # Get move based on visit counts, temp=0 for greedy
        #       return self._select_move_from_root(root, temp)
        
        # mcts_instance = MCTS(game_rules, self.model, self.device, ...)
        # return mcts_instance.get_action_for_state(game_state, self.mcts_simulations)
        
        # --- Fallback: Direct policy (less accurate representation of AlphaZero strength) ---
        encoded_s = encode_state(game_state).unsqueeze(0).to(self.device) # Add batch dim
        with torch.no_grad():
            policy_logits, _ = self.model(encoded_s)
        
        valid_moves = game_rules.get_valid_moves(game_state)
        valid_move_indices = [m.row * GRID_COLS + m.col for m in valid_moves] # Assuming flattened policy

        # Mask invalid moves
        masked_policy = torch.full_like(policy_logits, -float('inf'))
        if valid_move_indices:
            masked_policy[0, valid_move_indices] = policy_logits[0, valid_move_indices]
        
        if not valid_moves: # Should not happen
             print(f"WARNING: AZ Agent {self.player_id} has no valid moves!")
             return Position(row=0, col=0)

        best_action_idx = torch.argmax(masked_policy).item()
        row, col = divmod(best_action_idx, GRID_COLS)
        return Position(row=row, col=col)


    def get_name(self) -> str:
        return f"AlphaZeroAgent_P{self.player_id}"


class Arena:
    def __init__(self, agent1: BaseAgent, agent2: BaseAgent, grid_size_config: GridSizeConfig, logger: Optional[Any] = None):
        self.agent1 = agent1
        self.agent2 = agent2
        self.grid_size_config = grid_size_config
        self.logger = logger # ExperimentLogger instance

    def play_game(self, start_player_id: PlayerId = 1) -> Tuple[Optional[PlayerId], int, GameState]:
        """Plays a single game."""
        players = [self.agent1.player_info, self.agent2.player_info]
        game_state = game_rules.get_initial_state(self.grid_size_config, players)
        game_state.status = "active"
        game_state.currentPlayerId = start_player_id
        
        turn_count = 0
        max_turns = self.grid_size_config.rows * self.grid_size_config.cols * 5 # Generous turn limit for draw

        while game_state.status == "active":
            current_agent = self.agent1 if game_state.currentPlayerId == self.agent1.player_id else self.agent2
            
            move = current_agent.get_move(game_state)
            
            # Validate move (basic check, game_rules.apply_move should be robust)
            cell = game_state.grid[move.row][move.col]
            if not (cell.player is None or cell.player == game_state.currentPlayerId):
                if self.logger: self.logger.log_error(f"Agent {current_agent.get_name()} made invalid move {move} on cell owned by {cell.player}")
                # Forfeit for invalid move
                return self.agent1.player_id if current_agent == self.agent2 else self.agent2.player_id, turn_count, game_state

            game_state = game_rules.apply_move(game_state, move) # This updates status, winnerId
            turn_count += 1

            if turn_count > max_turns:
                if self.logger: self.logger.log_info(f"Game ended in a draw due to max turns ({max_turns}).")
                game_state.status = "draw"
                game_state.winnerId = None
                break
        
        return game_state.winnerId, turn_count, game_state

    def run_matches(self, num_games: int) -> Dict[str, Any]:
        """Plays num_games, alternating starting player."""
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        total_turns = 0

        for i in range(num_games):
            start_player_id = self.agent1.player_id if i % 2 == 0 else self.agent2.player_id
            winner_id, turns, _ = self.play_game(start_player_id)
            total_turns += turns

            if winner_id == self.agent1.player_id:
                agent1_wins += 1
            elif winner_id == self.agent2.player_id:
                agent2_wins += 1
            else:
                draws += 1
            
            if self.logger:
                 self.logger.log_info(f"Match {i+1}/{num_games} ({self.agent1.get_name()} vs {self.agent2.get_name()}): Winner P{winner_id}, Turns: {turns}, Start P{start_player_id}")


        return {
            "agent1_name": self.agent1.get_name(),
            "agent2_name": self.agent2.get_name(),
            "agent1_wins": agent1_wins,
            "agent2_wins": agent2_wins,
            "draws": draws,
            "num_games": num_games,
            "avg_turns": total_turns / num_games if num_games > 0 else 0,
            "agent1_win_ratio": agent1_wins / num_games if num_games > 0 else 0
        }

def load_agent_from_config(agent_config: Dict[str, Any], player_id: PlayerId, player_info: PlayerInfo, mcts_sims_for_eval: int) -> BaseAgent:
    """Helper to create agents based on config."""
    agent_type = agent_config.get("type", "random")
    model_path = agent_config.get("path")

    if agent_type == "random":
        return RandomAgent(player_id, player_info)
    elif agent_type == "alphazero" or agent_type == "previous_best" or agent_type == "specific_checkpoint":
        if not model_path and agent_type == "previous_best":
            model_path = BEST_MODEL_PATH
        if not model_path:
            raise ValueError(f"Model path required for AlphaZero agent type '{agent_type}'")
        return AlphaZeroAgent(player_id, player_info, model_path, mcts_sims_for_eval)
    # Add HeuristicAgent if you implement one
    # elif agent_type == "heuristic":
    #     return HeuristicAgent(player_id, player_info, ...)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
```

**4. Enhancing `neural_net.py` and `self_play.py` for Logging**

*   **`neural_net.py` (during `train_step` or similar):**

    ```python
    # Inside your NN training loop/function
    # from .monitoring import get_experiment_logger
    # logger = get_experiment_logger()

    # ... after loss calculation ...
    # logger.log_scalar("Loss/Policy", policy_loss.item(), global_step)
    # logger.log_scalar("Loss/Value", value_loss.item(), global_step)
    # logger.log_scalar("Loss/Total", total_loss.item(), global_step)
    # logger.log_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)

    # Periodically log weights and gradients (e.g., every N steps)
    # if global_step % config.LOG_HISTOGRAM_EVERY_N_STEPS == 0:
    #     for name, param in self.model.named_parameters():
    #         if param.grad is not None:
    #             logger.log_histogram(f"Gradients/{name}", param.grad.cpu().numpy(), global_step)
    #         if param.data is not None:
    #             logger.log_histogram(f"Weights/{name}", param.data.cpu().numpy(), global_step)
    ```

*   **`self_play.py` (during or after `play_game`):**

    ```python
    # from .monitoring import get_experiment_logger
    # logger = get_experiment_logger()

    # ... after a self-play game finishes ...
    # game_length = len(game_history) # or game_state.turnNumber
    # winner = game_state.winnerId # +1, -1, or 0 from perspective of player 1
    
    # logger.log_scalar("SelfPlay/GameLength", game_length, global_iteration_num)
    # if winner == player1_id:
    #     logger.log_scalar("SelfPlay/Wins_P1", 1, global_iteration_num)
    # elif winner == player2_id:
    #     logger.log_scalar("SelfPlay/Wins_P2", 1, global_iteration_num) # or just "SelfPlay/Outcome" with a value
    # else:
    #     logger.log_scalar("SelfPlay/Draws", 1, global_iteration_num)

    # You might also want to average these over a batch of self-play games
    # and log the average.
    
    # MCTS stats (if MCTS exposes them, e.g. avg tree size, value prediction error at root)
    # avg_mcts_nodes = sum(s.mcts_nodes for s in game_history) / len(game_history)
    # logger.log_scalar("SelfPlay/Avg_MCTS_Nodes_Per_Turn", avg_mcts_nodes, global_iteration_num)
    ```

**5. Main Training Pipeline (`ai/alphazero/train_pipeline.py`)**

This script orchestrates the entire process.

```python
# ai/alphazero/train_pipeline.py
import os
import shutil
import torch # Assuming PyTorch

from . import config
from .monitoring import ExperimentLogger, get_experiment_logger
from .neural_net import YourNeuralNet # Your NN class and training function
from .self_play import run_self_play_games # Your self-play game generation function
from .evaluation import Arena, load_agent_from_config, DEFAULT_PLAYER_INFOS
from .game_rules import get_initial_state # For creating grid_size_config for Arena if needed
from ai.models import GridSizeConfig

def main_training_loop():
    # 1. Initialize Logger
    # Run name can be passed as CLI arg or generated
    logger = ExperimentLogger(run_name=f"az_train_{config.GRID_ROWS}x{config.GRID_COLS}")

    try:
        logger.log_info("Starting AlphaZero Training Pipeline...")
        logger.log_info(f"Configuration: {vars(config)}")

        # 2. Setup: Device, Model, Optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log_info(f"Using device: {device}")

        # TODO: Initialize your model and optimizer based on config
        # current_model = YourNeuralNet(input_shape=..., action_size=..., config=...).to(device)
        # optimizer = torch.optim.Adam(current_model.parameters(), lr=config.LEARNING_RATE)
        # Example:
        action_size = config.GRID_ROWS * config.GRID_COLS
        current_model = YourNeuralNet(input_channels=config.INPUT_CHANNELS, # define in config
                                     num_res_blocks=config.NUM_RES_BLOCKS, # define in config
                                     action_size=action_size).to(device) # Example
        optimizer = torch.optim.Adam(current_model.parameters(), lr=config.LEARNING_RATE)


        # Load latest checkpoint if exists
        latest_checkpoint_path = get_latest_checkpoint(config.CHECKPOINT_DIR)
        start_iteration = 0
        if latest_checkpoint_path:
            logger.log_info(f"Loading model from checkpoint: {latest_checkpoint_path}")
            # TODO: Implement load_checkpoint function
            # start_iteration, current_model, optimizer = load_checkpoint(latest_checkpoint_path, current_model, optimizer, device)
            pass # Placeholder for your load_checkpoint logic
        
        if os.path.exists(config.BEST_MODEL_PATH):
             logger.log_info(f"Best model already exists at: {config.BEST_MODEL_PATH}")
        else: # Initialize best_model.pth with the initial (random) weights if it doesn't exist
            torch.save(current_model.state_dict(), config.BEST_MODEL_PATH)
            logger.log_info(f"Initialized best model at {config.BEST_MODEL_PATH}")


        replay_buffer = [] # Your replay buffer implementation (e.g., collections.deque)

        # 3. Main Training Loop
        for iteration in range(start_iteration, config.TOTAL_TRAINING_ITERATIONS):
            logger.log_info(f"--- Starting Iteration {iteration + 1}/{config.TOTAL_TRAINING_ITERATIONS} ---")

            # 3a. Self-Play Phase
            # TODO: Implement run_self_play_games to return list of (state, policy_target, value_target)
            # It should use current_model for MCTS decisions and log game stats
            # new_examples = run_self_play_games(current_model, config.NUM_SELF_PLAY_GAMES_PER_ITERATION, iteration, config, logger, device)
            # replay_buffer.extend(new_examples)
            # replay_buffer = replay_buffer[-config.REPLAY_BUFFER_SIZE:] # Keep buffer size limited
            # logger.log_info(f"Self-play generated {len(new_examples)} examples. Buffer size: {len(replay_buffer)}")
            # Example placeholder:
            if iteration % 5 == 0 : # Simulate generating some data
                logger.log_scalar("SelfPlay/AvgGameLength", 20 + iteration % 5, iteration) # Dummy data
                logger.log_scalar("SelfPlay/WinRate_vs_Self", 0.5, iteration) # Dummy data

            # 3b. Training Phase
            if len(replay_buffer) >= config.MIN_BUFFER_FOR_TRAINING: # Define in config
                # TODO: Implement train_on_batch function in neural_net.py
                # It should sample from replay_buffer and train current_model, logging losses
                # train_losses = train_on_batch(current_model, optimizer, replay_buffer, config.TRAIN_BATCH_SIZE, config.NUM_TRAINING_EPOCHS_PER_ITERATION, iteration, logger, device)
                # logger.log_scalars("Loss", {"policy": train_losses['policy'], "value": train_losses['value']}, iteration)
                # Example placeholder:
                 logger.log_scalar("Loss/Policy", 0.5 - iteration*0.001, iteration)
                 logger.log_scalar("Loss/Value",  0.2 - iteration*0.0005, iteration)


            # 3c. Save Checkpoint
            if (iteration + 1) % config.SAVE_CHECKPOINT_EVERY_N_ITERATIONS == 0:
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_iter_{iteration+1:04d}.pth")
                # TODO: Implement save_checkpoint
                # save_checkpoint(iteration + 1, current_model, optimizer, checkpoint_path)
                # logger.log_info(f"Saved checkpoint: {checkpoint_path}")
                # Example placeholder:
                os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                torch.save(current_model.state_dict(), checkpoint_path)
                logger.log_info(f"Saved checkpoint: {checkpoint_path}")


            # 3d. Evaluation Phase
            if config.EVALUATION_ENABLED and (iteration + 1) % config.EVALUATE_EVERY_N_ITERATIONS == 0:
                logger.log_info(f"--- Starting Evaluation for Iteration {iteration + 1} ---")
                
                current_model_path = os.path.join(config.CHECKPOINT_DIR, f"model_iter_{iteration+1:04d}.pth") # Use last saved
                if not os.path.exists(current_model_path): # if not saved yet, save current one for eval
                    torch.save(current_model.state_dict(), current_model_path)

                player1_info = DEFAULT_PLAYER_INFOS[0]
                player2_info = DEFAULT_PLAYER_INFOS[1]
                
                grid_conf = GridSizeConfig(rows=config.GRID_ROWS, cols=config.GRID_COLS)

                # Agent for the current model being trained
                # Ensure AlphaZeroAgent uses the correct MCTS simulation count for evaluation
                eval_mcts_sims = config.MCTS_SIMULATIONS_EVAL # Define in config (can be same or different from training)
                current_az_agent = AlphaZeroAgent(player1_info.id, player1_info, current_model_path, eval_mcts_sims)

                for opponent_config in config.EVALUATION_OPPONENTS:
                    opponent_name = opponent_config["name"]
                    logger.log_info(f"Evaluating {current_az_agent.get_name()} vs {opponent_name}")

                    opponent_agent = load_agent_from_config(opponent_config, player2_info.id, player2_info, eval_mcts_sims)
                    
                    arena = Arena(current_az_agent, opponent_agent, grid_conf, logger)
                    results = arena.run_matches(config.EVALUATION_GAMES_PER_OPPONENT)
                    
                    logger.log_info(f"Evaluation vs {opponent_name}: "
                                    f"P1 Wins: {results['agent1_wins']}, P2 Wins: {results['agent2_wins']}, Draws: {results['draws']}")
                    
                    logger.log_scalar(f"Evaluation/{opponent_name}/WinRate_vs_Opponent", results['agent1_win_ratio'], iteration + 1)
                    logger.log_scalar(f"Evaluation/{opponent_name}/AvgTurns", results['avg_turns'], iteration + 1)

                    # Update best model if current model performs well against "PreviousBest"
                    if opponent_config["type"] == "previous_best":
                        if results['agent1_win_ratio'] > config.WIN_RATIO_THRESHOLD_FOR_BEST_MODEL:
                            logger.log_info(f"New best model found! Win ratio {results['agent1_win_ratio']:.2f} > {config.WIN_RATIO_THRESHOLD_FOR_BEST_MODEL}")
                            shutil.copy(current_model_path, config.BEST_MODEL_PATH)
                            logger.log_info(f"Copied {current_model_path} to {config.BEST_MODEL_PATH}")
                        else:
                            logger.log_info(f"Current model did not surpass best model. Win ratio {results['agent1_win_ratio']:.2f}")
            
            logger.log_info(f"--- Finished Iteration {iteration + 1} ---")

    except Exception as e:
        if 'logger' in locals() and logger: # check if logger was initialized
            logger.log_error(f"Critical error in training pipeline: {e}", exc_info=True)
        else:
            print(f"Critical error before logger initialization: {e}") # Fallback print
        raise
    finally:
        if 'logger' in locals() and logger:
            logger.close()

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    # Helper to find the checkpoint with the highest iteration number
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_iter_") and f.endswith(".pth")]
    if not checkpoints:
        return None
    # Sort by iteration number (e.g., model_iter_0010.pth)
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])


if __name__ == "__main__":
    # This allows running the training directly
    # Make sure your project structure allows imports like `from . import config`
    # You might need to adjust Python path or run as `python -m ai.alphazero.train_pipeline`
    # from .. import parent_directory_setup # if needed for path adjustments
    main_training_loop()
```

**How to Use This System:**

1.  **Install TensorBoard:** `pip install tensorboard`
2.  **Fill in TODOs:** Replace placeholder comments (like model initialization, self-play logic, training logic, checkpoint loading/saving) with your actual implementations.
3.  **Configure `config.py`:** Set paths, iteration counts, evaluation parameters, etc.
4.  **Run the Training Pipeline:**
    Navigate to the directory above `ai` (your project root) and run:
    `python -m ai.alphazero.train_pipeline`
    (The `-m` flag helps Python resolve relative imports correctly).
5.  **Monitor with TensorBoard:**
    In a separate terminal, navigate to your project root and run:
    `tensorboard --logdir ai/logs/tensorboard`
    Then open the URL provided (usually `http://localhost:6006`) in your browser. You'll see scalars (losses, win rates, game lengths), histograms (weights, gradients), and text logs.
6.  **Check `training.log`:** For detailed text-based logs.
7.  **Observe `trained_models/`:** See checkpoints being saved and `best_model.pth` being updated.

**Key Automation Aspects Covered:**

*   **Iterative Learning:** The `main_training_loop` handles self-play, training, saving, and evaluation repeatedly.
*   **Metric Logging:** `ExperimentLogger` automatically sends data to TensorBoard and a log file.
*   **Automated Evaluation:** The `Arena` class and its integration into the pipeline periodically test the current model's strength.
*   **Best Model Management:** The pipeline automatically updates `best_model.pth` based on evaluation performance.
*   **Checkpointing:** Models are saved regularly, allowing you to resume training or go back to previous versions.

This provides a robust framework for developing, training, and understanding your AlphaZero AI for Chain Reaction. Remember to implement the game-specific AI logic (MCTS, NN, self-play data generation) carefully within this framework.