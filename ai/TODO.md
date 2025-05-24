# TODO List for AlphaZero AI Project

This file tracks pending tasks, potential improvements, and areas for future development for the AlphaZero AI project.

## High Priority / Core Functionality

*   **MCTS Module (`ai/alphazero/mcts.py`):**
    *   Verify and finalize the interface of the MCTS module being developed by the other agent/team. Key functions expected by `self_play.py` and `evaluation.py` are:
        *   `get_policy_for_state(nn, game_state, config)`: Returns an MCTS-improved policy vector.
        *   `select_move(policy_vector, turn_number, config)`: Selects a move based on policy and temperature.
    *   Consider if `AlphaZeroAgent.get_move` in `evaluation.py` should directly call a high-level MCTS "get best move" function if available, rather than re-implementing parts of the selection logic.

*   **Training Pipeline (`ai/alphazero/train_pipeline.py`):**
    *   **Implement `load_checkpoint`:** The logic for loading model and optimizer states to resume training is currently a placeholder. This needs to be fully implemented.
        *   Consider saving/loading iteration number and replay buffer state as well for full resumption.
    *   **Refactor Self-Play & Training Logic:** Consider refactoring the inline self-play game generation and batch training loops within `main_training_loop` into their own dedicated functions (e.g., `run_self_play_iteration(...)`, `train_on_replay_buffer_epochs(...)`) for better modularity and readability.

## Enhancements & Refinements

*   **Logging & Monitoring:**
    *   **Self-Play Stats:** Enhance `ai/alphazero/self_play.py` (the `play_game` function) to accept the `ExperimentLogger` instance and log detailed self-play game statistics:
        *   Average game length per iteration.
        *   Win/loss/draw outcomes from self-play games (e.g., if player 1 consistently wins, it might indicate an issue).
        *   If MCTS can expose them: average tree size, value prediction error at the root, etc., per move or per game.
    *   **Granular Training Logs:** Enhance the training step logic (either in `train_pipeline.py` or a refactored `neural_net.py` training function) to log:
        *   Losses per batch/epoch (not just per iteration average).
        *   Learning rate if a scheduler is used.
    *   **Histograms:** Add logging for weights and gradients histograms in `ExperimentLogger` (e.g., called periodically from the training loop). This requires `LOG_HISTOGRAM_EVERY_N_STEPS` in `config.py`.

*   **Evaluation Module (`ai/alphazero/evaluation.py`):**
    *   **MCTS in `AlphaZeroAgent.get_move`:** Update `AlphaZeroAgent.get_move` to use the full MCTS search (similar to how self-play determines moves, but with deterministic temperature) for a more accurate evaluation of model strength, instead of relying solely on the direct policy output.
    *   **More Opponent Types:** Implement other baseline agents (e.g., a simple heuristic-based agent).
    *   **Elo Rating System:** Consider implementing an Elo rating system for tracking model strength over time more formally.

*   **Configuration (`ai/alphazero/config.py`):**
    *   Add `LOG_HISTOGRAM_EVERY_N_STEPS` if histogram logging is prioritized.
    *   Review if `MCTS_SIMULATIONS` (for self-play) and `MCTS_SIMULATIONS_EVAL` (for evaluation) are sufficient, or if more granular MCTS settings are needed (e.g., separate C_PUCT for eval).

*   **Neural Network (`ai/alphazero/neural_net.py`):**
    *   **Architecture Experiments:** Allow for easier experimentation with different neural network architectures (e.g., varying number of residual blocks, convolutional layers, etc., possibly controlled by `config.py`).
    *   **Learning Rate Scheduler:** Implement an optional learning rate scheduler in the training pipeline.

## Testing & Robustness

*   **Unit Tests:** Develop unit tests for:
    *   `game_rules.py` functions.
    *   `utils.py` functions (`encode_state`, `convert_move_index_to_position`).
    *   `neural_net.py` model instantiation and forward pass.
    *   `monitoring.py` logger functionality (mocking file system and TensorBoard writer).
    *   `evaluation.py` agent logic and arena game play.
*   **Integration Tests:**
    *   Test the self-play loop with data generation.
    *   Test the training pipeline (self-play -> data collection -> training step).
    *   Test the evaluation pipeline.
*   **Error Handling:** Review and enhance error handling across all modules for greater robustness.

## Documentation

*   **Update `HOW_TO_RUN.md`:** As new features are added or refined, keep the guide updated.
*   **Code Comments & Docstrings:** Ensure all functions and complex code sections are well-commented and have clear docstrings.

```
