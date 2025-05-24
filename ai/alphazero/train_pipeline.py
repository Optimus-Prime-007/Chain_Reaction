from typing import List, Tuple, Any, Optional
import torch
import torch.optim as optim
import torch.nn.functional as F
import collections
import random
import numpy as np
import importlib # Not strictly used by the new changes, but kept if used elsewhere
import json # For logging config as text
from datetime import datetime # For logger run name
import os # Added for os.makedirs

# Direct imports for AlphaZero components:
from .neural_net import AlphaZeroNet
from . import game_rules
from . import mcts # This module's actual implementation is by another agent
from . import utils
from . import self_play as self_play_module # Rename to avoid conflict
from .monitoring import ExperimentLogger # Added for logging
from .evaluation import run_evaluation # New import for evaluation

# config will be passed as an argument, but for type hinting:
# from . import config as config_module_type

# Existing functions prepare_batch and train_step are assumed to be below.


def prepare_batch(
    batch_data: List[Tuple[torch.Tensor, List[float], float]], 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Prepares a batch of training data for input into the neural network.

  The function unnpacks a list of training examples, converts them into 
  PyTorch tensors, and moves them to the specified device.

  Args:
    batch_data: A list of training examples. Each example is a tuple:
      - encoded_state (torch.Tensor): The game state encoded as a tensor.
      - mcts_policy (List[float]): The MCTS policy vector for that state.
      - game_outcome (float): The game outcome from the perspective of the 
                              player whose turn it was (-1.0 loss, 0.0 draw, 1.0 win).
    device: The PyTorch device (e.g., 'cpu' or 'cuda') to move the tensors to.

  Returns:
    A tuple containing three tensors:
      - states_tensor: A batch of encoded states.
                       Shape: (batch_size, channels, rows, cols)
      - policies_tensor: A batch of MCTS policy targets.
                         Shape: (batch_size, num_actions)
      - values_tensor: A batch of game outcome targets (value targets).
                       Shape: (batch_size,)
  """
  # 1. Unzip the batch_data
  encoded_states_list = [item[0] for item in batch_data]
  mcts_policies_list = [item[1] for item in batch_data]
  game_outcomes_list = [item[2] for item in batch_data]

  # 2. Convert lists into PyTorch tensors and move to device
  # Assumes encoded_states_list contains tensors from encode_state
  states_tensor = torch.stack(encoded_states_list).to(device)
  
  policies_tensor = torch.tensor(mcts_policies_list, dtype=torch.float32).to(device)
  
  # Ensure values_tensor is a column vector or a 1D tensor as appropriate for loss function
  # For CrossEntropyLoss with logits, target is usually (N).
  # For MSELoss, target could be (N) or (N,1). Here, (N) is fine.
  values_tensor = torch.tensor(game_outcomes_list, dtype=torch.float32).to(device)

  # 3. Return the tensors
  return states_tensor, policies_tensor, values_tensor

# Example Usage (for context, not part of the function itself):
# if __name__ == '__main__':
#     # Assuming GRID_SIZE and NUM_ACTIONS are defined (e.g. from config)
#     # For example:
#     # from ai.alphazero.config import GRID_SIZE, NUM_ACTIONS, MCTS_SIMULATIONS
#     # INPUT_CHANNELS = 5 # Example
# 
#     # sample_batch_data = [
#     # (torch.randn(INPUT_CHANNELS, GRID_SIZE.rows, GRID_SIZE.cols), [0.1]*NUM_ACTIONS, 1.0),
#     # (torch.randn(INPUT_CHANNELS, GRID_SIZE.rows, GRID_SIZE.cols), [0.2]*NUM_ACTIONS, -1.0)
#     # ]
#     # current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # print(f"Using device: {current_device}")
# 
#     # states, policies, values = prepare_batch(sample_batch_data, current_device)
# 
#     # print("States Tensor Shape:", states.shape)
#     # print("Policies Tensor Shape:", policies.shape)
#     # print("Values Tensor Shape:", values.shape)
#     # print("States Tensor Device:", states.device)
#     # print("Policies Tensor Device:", policies.device)
#     # print("Values Tensor Device:", values.device)

def train_step(
    nn: 'AlphaZeroNet',  # Using string literal for AlphaZeroNet type hint
    optimizer: torch.optim.Optimizer,
    states_tensor: torch.Tensor,
    policies_tensor: torch.Tensor,
    values_tensor: torch.Tensor,
    config: Any  # Assuming config object has VALUE_LOSS_WEIGHT
) -> Tuple[float, float, float]:
  """
  Performs a single training step on a batch of data.

  This includes a forward pass, loss calculation, backward pass, and
  optimizer step.

  Args:
    nn: The neural network (AlphaZeroNet instance) to be trained.
    optimizer: The PyTorch optimizer (e.g., Adam, SGD).
    states_tensor: Batch of encoded game states.
    policies_tensor: Batch of MCTS-derived policy targets.
    values_tensor: Batch of game outcome targets (value targets).
    config: A configuration object/module containing parameters like
            VALUE_LOSS_WEIGHT.

  Returns:
    A tuple containing the scalar values of:
      - total_loss: The combined policy and value loss.
      - policy_loss: The policy loss (e.g., cross-entropy).
      - value_loss: The value loss (e.g., mean squared error).
  """
  # 1. Set the neural network to training mode
  nn.train()

  # 2. Zero the optimizer's gradients
  optimizer.zero_grad()

  # 3. Perform a forward pass through the network
  policy_logits, value_preds = nn(states_tensor)

  # 4. Calculate the policy loss
  # policy_logits are raw scores, policies_tensor are probabilities from MCTS
  policy_loss = F.cross_entropy(policy_logits, policies_tensor)

  # 5. Calculate the value loss
  # value_preds might be [batch_size, 1], values_tensor is [batch_size]
  # .squeeze(-1) changes [batch_size, 1] to [batch_size]
  value_loss = F.mse_loss(value_preds.squeeze(-1), values_tensor)

  # 6. Calculate the total combined loss
  # config.VALUE_LOSS_WEIGHT should be defined in your config module/object
  total_loss = policy_loss + config.VALUE_LOSS_WEIGHT * value_loss

  # 7. Perform backpropagation
  total_loss.backward()

  # 8. Update the network's weights
  optimizer.step()

  # 9. Return the scalar loss values
  return total_loss.item(), policy_loss.item(), value_loss.item()

def main_training_loop(config: Any):
  """
  Main training loop for AlphaZero.

  Orchestrates self-play, training, and checkpointing.

  Args:
    config: A configuration module/object containing all necessary parameters
            for the training process (e.g., learning rate, batch size,
            number of iterations, MCTS settings, model paths, etc.).
  """
  # Initialize logger first
  logger = ExperimentLogger(run_name=f"AlphaZero_Train_{datetime.now().strftime('%Y%m%d_%H%M%S')}", config=config)
  logger.log_info("Main training loop started.")

  try:
    # Log key configurations
    config_log_dict = {
        "LEARNING_RATE": config.LEARNING_RATE,
        "BATCH_SIZE": config.BATCH_SIZE,
        "NUM_ITERATIONS": config.NUM_ITERATIONS,
        "MCTS_SIMULATIONS": config.MCTS_SIMULATIONS,
        "REPLAY_BUFFER_SIZE": config.REPLAY_BUFFER_SIZE,
        "MIN_BUFFER_FOR_TRAINING": config.MIN_BUFFER_FOR_TRAINING,
        "NUM_SELF_PLAY_GAMES_PER_ITER": config.NUM_SELF_PLAY_GAMES_PER_ITER,
        "NUM_EPOCHS_PER_ITERATION": config.NUM_EPOCHS_PER_ITERATION,
        "VALUE_LOSS_WEIGHT": config.VALUE_LOSS_WEIGHT,
        "GRID_ROWS": config.GRID_SIZE.rows,
        "GRID_COLS": config.GRID_SIZE.cols,
        # Add other important parameters if needed
    }
    logger.log_text("initial_config", json.dumps(config_log_dict, indent=2), 0)

    # 1. Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_info(f"Using device: {device}")

    # Create Checkpoint Directory
    if hasattr(config, 'CHECKPOINT_DIR'):
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        logger.log_info(f"Ensured checkpoint directory exists: {config.CHECKPOINT_DIR}")
    else:
        logger.log_warning("CHECKPOINT_DIR not found in config. Checkpoints may not be saved to the intended directory.")

    # Ensure all necessary config attributes are present
    required_attrs = [
      'INPUT_CHANNELS', 'GRID_SIZE', 'LEARNING_RATE', 'REPLAY_BUFFER_SIZE',
      'NUM_ITERATIONS', 'NUM_SELF_PLAY_GAMES_PER_ITER', 'MIN_BUFFER_FOR_TRAINING',
      'NUM_EPOCHS_PER_ITERATION', 'BATCH_SIZE', 'VALUE_LOSS_WEIGHT' 
      # CHECKPOINT_SAVE_PATH_FORMAT and LOAD_CHECKPOINT_PATH are optional
  ]
  for attr in required_attrs:
      if not hasattr(config, attr):
          raise AttributeError(f"Missing required configuration attribute: {attr}")
  if not hasattr(config.GRID_SIZE, 'rows') or not hasattr(config.GRID_SIZE, 'cols'):
      raise AttributeError("config.GRID_SIZE must have 'rows' and 'cols' attributes.")


  nn = AlphaZeroNet(config.INPUT_CHANNELS, config.GRID_SIZE.rows, config.GRID_SIZE.cols).to(device)
  
  if hasattr(config, 'LOAD_CHECKPOINT_PATH') and config.LOAD_CHECKPOINT_PATH:
    try:
      nn.load_model(config.LOAD_CHECKPOINT_PATH, device)
      logger.log_info(f"Loaded model from {config.LOAD_CHECKPOINT_PATH}")
    except FileNotFoundError:
      logger.log_warning(f"Checkpoint file not found at {config.LOAD_CHECKPOINT_PATH}, starting fresh.")
    except Exception as e:
      logger.log_error(f"Error loading model from {config.LOAD_CHECKPOINT_PATH}: {e}. Starting fresh.", exc_info=True)


  optimizer = optim.Adam(nn.parameters(), lr=config.LEARNING_RATE)
  replay_buffer = collections.deque(maxlen=config.REPLAY_BUFFER_SIZE)
  
  logger.log_info("Initialization complete.")

  # 2. Main Loop
  for iteration in range(1, config.NUM_ITERATIONS + 1):
    logger.log_info(f"--- Iteration {iteration}/{config.NUM_ITERATIONS} ---")

    # Self-Play Phase
    nn.eval() # Set network to evaluation mode for self-play
    logger.log_info("Starting self-play phase...")
    num_new_examples = 0
    for game_num in range(1, config.NUM_SELF_PLAY_GAMES_PER_ITER + 1):
      logger.log_info(f"  Playing game {game_num}/{config.NUM_SELF_PLAY_GAMES_PER_ITER}...")
      # self_play_module.play_game is expected to handle MCTS and NN interactions
      game_data = self_play_module.play_game(nn, config, game_rules, mcts, utils)
      replay_buffer.extend(game_data)
      num_new_examples += len(game_data)
    logger.log_info(f"Self-play phase complete. Added {num_new_examples} examples. Buffer size: {len(replay_buffer)}")
    logger.log_scalar("self_play/num_new_examples_iteration", num_new_examples, iteration)
    logger.log_scalar("self_play/replay_buffer_size", len(replay_buffer), iteration)

    # Training Phase
    logger.log_info("Starting training phase...")
    if len(replay_buffer) >= config.MIN_BUFFER_FOR_TRAINING:
      nn.train() # Set network to training mode
      total_iteration_loss = 0.0
      total_iteration_policy_loss = 0.0
      total_iteration_value_loss = 0.0
      epochs_run_this_iteration = 0 # Renamed from num_epochs_done for clarity with loop var

      for epoch in range(1, config.NUM_EPOCHS_PER_ITERATION + 1):
        # Ensure BATCH_SIZE isn't larger than current replay_buffer size
        current_batch_size = min(config.BATCH_SIZE, len(replay_buffer))
        if current_batch_size == 0:
            logger.log_info("  Replay buffer is empty, cannot sample for training.")
            break 
            
        sampled_data_points = random.sample(list(replay_buffer), current_batch_size)
        states_t, policies_t, values_t = prepare_batch(sampled_data_points, device)
        
        loss, pi_loss, v_loss = train_step(nn, optimizer, states_t, policies_t, values_t, config)
        
        total_iteration_loss += loss
        total_iteration_policy_loss += pi_loss
        total_iteration_value_loss += v_loss
        epochs_run_this_iteration += 1
        
        if epoch % 10 == 0: # Log every 10 epochs (or can be adjusted)
          # These are running averages within an iteration's epochs
          avg_epoch_loss = total_iteration_loss / epochs_run_this_iteration
          avg_epoch_pi_loss = total_iteration_policy_loss / epochs_run_this_iteration
          avg_epoch_v_loss = total_iteration_value_loss / epochs_run_this_iteration
          logger.log_info(f"  Epoch {epoch}: Running Avg Loss: {avg_epoch_loss:.4f} (Pi: {avg_epoch_pi_loss:.4f}, V: {avg_epoch_v_loss:.4f})")
      
      if epochs_run_this_iteration > 0:
        avg_iter_total_loss = total_iteration_loss / epochs_run_this_iteration
        avg_iter_policy_loss = total_iteration_policy_loss / epochs_run_this_iteration
        avg_iter_value_loss = total_iteration_value_loss / epochs_run_this_iteration
        logger.log_scalar("training/avg_total_loss_iteration", avg_iter_total_loss, iteration)
        logger.log_scalar("training/avg_policy_loss_iteration", avg_iter_policy_loss, iteration)
        logger.log_scalar("training/avg_value_loss_iteration", avg_iter_value_loss, iteration)
        logger.log_info(f"Training phase complete for iteration. Avg Iteration Loss: {avg_iter_total_loss:.4f} (Pi: {avg_iter_policy_loss:.4f}, V: {avg_iter_value_loss:.4f})")
      else:
        logger.log_info("No epochs run in training phase (e.g. buffer became too small or NUM_EPOCHS_PER_ITERATION is 0).")
    else:
      logger.log_info(f"Replay buffer size ({len(replay_buffer)}) is less than min ({config.MIN_BUFFER_FOR_TRAINING}). Skipping training.")

    # --- Save Checkpoint ---
    if hasattr(config, 'SAVE_CHECKPOINT_EVERY_N_ITERATIONS') and \
       hasattr(config, 'CHECKPOINT_SAVE_PATH_FORMAT') and \
       hasattr(config, 'CHECKPOINT_DIR'): # CHECKPOINT_DIR existence already handled

        if iteration % config.SAVE_CHECKPOINT_EVERY_N_ITERATIONS == 0:
            # CHECKPOINT_SAVE_PATH_FORMAT is expected to be a full path pattern like
            # "ai/trained_models/checkpoints/alphazero_iter_{iteration}.pth"
            save_path = config.CHECKPOINT_SAVE_PATH_FORMAT.format(iteration=iteration)
            
            try:
                # The directory config.CHECKPOINT_DIR was already created.
                # If CHECKPOINT_SAVE_PATH_FORMAT could define further subdirectories,
                # os.makedirs(os.path.dirname(save_path), exist_ok=True) would be needed here.
                # Given current config, CHECKPOINT_DIR covers the directory part of CHECKPOINT_SAVE_PATH_FORMAT.
                
                nn.save_model(save_path) # save_model is a method in AlphaZeroNet
                logger.log_info(f"Saved checkpoint to {save_path}")
            except Exception as e:
                logger.log_error(f"Error saving checkpoint to {save_path}: {e}", exc_info=True)
        else:
            logger.log_info(f"Skipping checkpoint save at iteration {iteration} (not a multiple of {config.SAVE_CHECKPOINT_EVERY_N_ITERATIONS}).")
    else:
        logger.log_warning("Checkpointing configuration attributes (SAVE_CHECKPOINT_EVERY_N_ITERATIONS, CHECKPOINT_DIR, CHECKPOINT_SAVE_PATH_FORMAT) missing or incomplete. Skipping checkpoint save.")


            # --- Evaluation Phase ---
            logger.log_info("Checking for evaluation trigger...")
            if hasattr(config, 'EVALUATION_ENABLED') and config.EVALUATION_ENABLED:
                if hasattr(config, 'EVALUATE_EVERY_N_ITERATIONS') and config.EVALUATE_EVERY_N_ITERATIONS > 0:
                    if iteration % config.EVALUATE_EVERY_N_ITERATIONS == 0:
                        logger.log_info(f"--- Triggering Evaluation for Iteration {iteration} ---")
                        try:
                            # Pass the current iteration number, config, and logger
                            # nn is not directly passed to run_evaluation; run_evaluation loads the model itself.
                            summary = run_evaluation(iteration, config, logger)
                            
                            # Log the summary dictionary as a JSON string for readability in text logs
                            # The run_evaluation function itself handles detailed scalar logging to TensorBoard.
                            logger.log_info(f"Evaluation summary for iteration {iteration}:")
                            logger.log_info(json.dumps(summary, indent=2)) # Log pretty-printed JSON

                            # Example of logging a specific overall metric if desired (optional)
                            # if summary and not summary.get("error") and "RandomBot" in summary: # Example check
                            #    if "agent1_win_ratio" in summary["RandomBot"]:
                            #        logger.log_scalar("evaluation/overall/win_rate_vs_RandomBot", summary["RandomBot"]["agent1_win_ratio"], iteration)

                        except Exception as e:
                            logger.log_error(f"Error during evaluation for iteration {iteration}: {e}", exc_info=True)
                        logger.log_info(f"--- Evaluation for Iteration {iteration} Finished ---")
                    else:
                        logger.log_info(f"Skipping evaluation at iteration {iteration} (not a multiple of {config.EVALUATE_EVERY_N_ITERATIONS}).")
                else:
                    logger.log_warning("EVALUATE_EVERY_N_ITERATIONS not configured properly (must be > 0 or attribute missing). Skipping evaluation.")
            else:
                logger.log_info("Evaluation is disabled in config or EVALUATION_ENABLED attribute missing. Skipping evaluation phase.")
            
    logger.log_info(f"--- Iteration {iteration} complete ---")
    logger.log_info("\n") # For spacing in logs

  logger.log_info("Main training loop finished successfully.")

  except Exception as e:
      if 'logger' in locals() and logger._initialized:
          logger.log_error(f"Exception in main training loop: {e}", exc_info=True)
      else: # Logger might not be initialized if error is early
          print(f"CRITICAL: Exception before/during logger init in main training loop: {e}")
      raise # Re-raise the exception after logging
  finally:
      if 'logger' in locals() and logger._initialized: # Check if logger was initialized
          logger.log_info("Closing logger.")
          logger.close()


if __name__ == '__main__':
    # Print statements are fine here as this is about script execution setup/failure
    print("Attempting to run main training loop from train_pipeline.py...")
    # This assumes config.py is in the same directory or accessible
    # and that MCTS module provides the expected interface.
    try:
        # Attempt to import the config module from the current package
        # This is a common way to structure AlphaZero projects
        from . import config as main_config 
        print(f"Loaded configuration module: {main_config}")
        
        # You might want to add specific checks here if main_config could be None
        # or if certain attributes are critical for __main__ execution
        
        main_training_loop(main_config)
        
    except ImportError as e:
        print(f"Failed to import config for __main__ execution: {e}. Ensure config.py is accessible and part of the package.")
        print("Make sure you are running this script as part of a package, e.g., using 'python -m ai.alphazero.train'")
    except AttributeError as e:
        print(f"Configuration error: {e}. Ensure all necessary attributes are in config.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
