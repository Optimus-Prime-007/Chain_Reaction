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