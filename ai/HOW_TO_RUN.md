# How to Run the AlphaZero AI Project

This guide provides instructions on how to configure, train, monitor, and evaluate the AlphaZero AI.

## 1. Prerequisites

*   **Python:** Python 3.8 or newer is recommended.
*   **Dependencies:** Install the required Python packages. A `requirements.txt` should ideally be provided. Key libraries include:
    *   `torch` (PyTorch, for neural networks)
    *   `tensorboard` (for monitoring training progress)
    *   `numpy` (for numerical operations)
    *   `fastapi` (for serving the AI via an API)
    *   `uvicorn` (for running the FastAPI server)
    *   (Other libraries as needed by the game logic or specific components)
    
    To install dependencies (assuming a `requirements.txt` file):
    ```bash
    pip install -r requirements.txt 
    # If no requirements.txt, install manually:
    # pip install torch tensorboard numpy fastapi uvicorn pydantic
    ```
*   **MCTS Module:** This implementation assumes an MCTS module (`ai/alphazero/mcts.py`) with a specific interface is available (as it's developed by another agent/team).

## 2. Configuration (`ai/alphazero/config.py`)

The main configuration for the AlphaZero agent, training pipeline, evaluation, and logging is managed in `ai/alphazero/config.py`. Before starting a training run, you might want to review and adjust these key parameters:

*   **Game Setup:**
    *   `GRID_SIZE`: Defines the game board dimensions (e.g., `GridSizeConfig(rows=6, cols=6)`).
    *   `GRID_ROWS`, `GRID_COLS`: Derived from `GRID_SIZE` for direct access.
*   **Training Loop Control:**
    *   `NUM_ITERATIONS`: Total number of training iterations (self-play -> train -> evaluate cycles).
    *   `NUM_SELF_PLAY_GAMES_PER_ITER`: Number of self-play games generated per iteration.
    *   `NUM_EPOCHS_PER_ITERATION`: Number of training epochs over the collected data per iteration.
*   **Neural Network & MCTS:**
    *   `INPUT_CHANNELS`: Number of channels for the NN input tensor (should match `encode_state` output).
    *   `LEARNING_RATE`: For the Adam optimizer.
    *   `MCTS_SIMULATIONS`: Number of MCTS simulations during self-play move selection.
    *   `MCTS_SIMULATIONS_EVAL`: Number of MCTS simulations during evaluation games.
*   **Data & Checkpointing:**
    *   `REPLAY_BUFFER_SIZE`, `MIN_BUFFER_FOR_TRAINING`, `BATCH_SIZE`.
    *   `LOG_DIR`, `TENSORBOARD_DIR`, `TRAINING_LOG_FILE`: Paths for logs.
    *   `CHECKPOINT_DIR`, `BEST_MODEL_PATH`, `CHECKPOINT_SAVE_PATH_FORMAT`: Paths for saving model weights.
    *   `SAVE_CHECKPOINT_EVERY_N_ITERATIONS`: Frequency of saving checkpoints.
    *   `LOAD_CHECKPOINT_PATH`: Set to a model path to resume training.
*   **Evaluation Settings (Covered in Part 2 of this guide)**

## 3. Running the Training Pipeline

The core training process is orchestrated by `ai/alphazero/train_pipeline.py`.

*   **To start training:**
    Navigate to the project's root directory (the one containing the `ai` folder) in your terminal and run:
    ```bash
    python -m ai.alphazero.train_pipeline
    ```
    Alternatively, if you are in the `ai/alphazero/` directory:
    ```bash
    python train_pipeline.py 
    ```
    (The first command is generally more robust for Python's module system).

*   **Expected Output:**
    *   **Console:** You will see logs printed to the console, showing the progress of iterations, self-play games, training epochs, and eventually evaluation phases. This output is managed by the `ExperimentLogger`.
    *   **Log Files:**
        *   A detailed text log will be saved to `ai/logs/training.log`.
        *   TensorBoard event files will be saved under `ai/logs/tensorboard/RUN_NAME/` (where `RUN_NAME` is timestamped, e.g., `AlphaZero_Train_YYYYMMDD_HHMMSS`).

## 4. Monitoring Training Progress

You can monitor the training process using TensorBoard and the text logs.

*   **TensorBoard:**
    1.  **Launch TensorBoard:** Open a new terminal, navigate to the project root, and run:
        ```bash
        tensorboard --logdir ai/logs/tensorboard
        ```
    2.  **Access in Browser:** Open your web browser and go to the URL provided by TensorBoard (usually `http://localhost:6006`).
    3.  **Key Metrics to Observe:**
        *   `training/avg_total_loss_iteration`
        *   `training/avg_policy_loss_iteration`
        *   `training/avg_value_loss_iteration`
        *   `self_play/num_new_examples_iteration`
        *   `self_play/replay_buffer_size`
        *   `evaluation/...` (various win rates and stats from evaluation - covered in Part 2)
        *   (Look for trends: losses should generally decrease, win rates against benchmarks should improve over iterations).

*   **Text Logs (`ai/logs/training.log`):**
    *   This file contains detailed, timestamped logs of the entire training pipeline.
    *   It's useful for debugging issues, seeing exact configuration values used for a run, and getting a chronological record of events.

```

## 5. Understanding Evaluation Output

The training pipeline includes an evaluation phase to assess the strength of newly trained models against various opponents, including previous versions of itself or baseline AIs like a random agent.

*   **Configuration:** Evaluation behavior is controlled by settings in `ai/alphazero/config.py`:
    *   `EVALUATION_ENABLED`: Set to `True` to enable the evaluation phase.
    *   `EVALUATE_EVERY_N_ITERATIONS`: Determines how often (in terms of training iterations) the evaluation is run.
    *   `EVALUATION_GAMES_PER_OPPONENT`: Number of games played against each specified opponent.
    *   `EVALUATION_OPPONENTS`: A list defining the opponents to play against (e.g., "random", "previous_best").
    *   `WIN_RATIO_THRESHOLD_FOR_BEST_MODEL`: The win rate a new model must achieve against the current "previous_best" to replace it.

*   **Triggering:** Evaluation is automatically triggered by the `train_pipeline.py` script based on these settings.

*   **Finding Results:**
    *   **Console/Text Logs (`ai/logs/training.log`):** During an evaluation phase, detailed logs are printed, including:
        *   Which model iteration is being evaluated.
        *   Matchups being played (e.g., "Iter_100 vs RandomBot").
        *   Individual game outcomes within a match.
        *   Summary statistics for each matchup (total wins, losses, draws, win ratio).
        *   Notifications if a new model becomes the "best_model.pth".
    *   **TensorBoard:**
        *   Scalar metrics for each evaluation matchup are logged under tags like:
            *   `evaluation/{opponent_name}/win_rate_vs_opponent`
            *   `evaluation/{opponent_name}/avg_turns_vs_opponent`
        *   These allow you to track how the win rate of your AlphaZero agent changes over training iterations against different benchmarks.

*   **The "Best Model" (`ai/trained_models/best_model.pth`):**
    *   This file stores the weights of the AlphaZero model that has performed best so far according to the evaluation criteria (specifically, by outperforming the previous "best_model.pth" by the `WIN_RATIO_THRESHOLD_FOR_BEST_MODEL`).
    *   It represents the current strongest version of your trained AI and is typically the model you would use for actual gameplay or deployment.

## 6. Running the Game Server (FastAPI)

The project includes a FastAPI server (`ai/server.py`) to serve moves from a trained AI, allowing it to be integrated with a game client or other applications.

*   **AI Model Used:**
    *   The server, through `ai/bot.py`, loads the model specified by `BEST_MODEL_PATH` in `ai/alphazero/config.py` when the application starts.
    *   Ensure `BEST_MODEL_PATH` points to the desired trained model (e.g., the automatically updated `best_model.pth` or a specific iteration checkpoint).

*   **To run the server:**
    Navigate to the project's root directory and execute:
    ```bash
    uvicorn ai.server:app --reload --port 8000
    ```
    *   `--reload` enables auto-reloading when code changes (useful for development).
    *   `--port 8000` specifies the port (you can change this if needed).

*   **Using the AlphaZero AI in a Game:**
    *   The `/get-ai-move` endpoint (POST request) expects a `GameState` object.
    *   The `ai/bot.py` module determines which AI logic to use for the `currentPlayerId` in the provided `GameState`.
    *   If you have configured `PlayerInfo` objects (e.g., in `ai/alphazero/config.py` under `PLAYERS_INFO`) with `isAI = True` and `ai_type = "alphazero"`, the server will use the loaded AlphaZero model for that player.
    *   If `ai_type` is "random" or not specified for an AI player, it will fall back to the random move generator.

This setup allows you to easily switch between different AI behaviors for players in your game environment when interacting with the API.
