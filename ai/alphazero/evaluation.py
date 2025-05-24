import random
import os
import torch # Assuming PyTorch
from typing import List, Dict, Any, Tuple, Optional
import shutil # For shutil.copy

# AlphaZero specific imports
# Adjust path if necessary, assuming ai.models is accessible
try:
    from ..models import GameState, Position, PlayerId, GridSizeConfig, PlayerInfo
except ImportError: # Fallback for direct execution or different structure
    from ai.models import GameState, Position, PlayerId, GridSizeConfig, PlayerInfo

from . import game_rules as game_rules_module # Import as module to avoid name clashes
from .neural_net import AlphaZeroNet
from . import utils as utils_module
from .monitoring import ExperimentLogger # For type hinting logger
# Config will be used for GRID_COLS, INPUT_CHANNELS etc.
# It's better to pass the config object or specific values to AlphaZeroAgent constructor

class BaseAgent:
    def __init__(self, player_id: PlayerId, player_info: PlayerInfo):
        self.player_id = player_id
        self.player_info = player_info

    def get_move(self, game_state: GameState) -> Position:
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_name(self) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")

class RandomAgent(BaseAgent):
    def get_move(self, game_state: GameState) -> Position:
        valid_moves = game_rules_module.get_valid_moves(game_state)
        if not valid_moves:
            # This should ideally not happen if game ends correctly
            # print(f"WARNING: RandomAgent {self.player_id} has no valid moves!") # Use logger in Arena
            # Fallback: pick first cell if grid not full (not a great fallback but avoids crash)
            return Position(row=0, col=0) 
        return random.choice(valid_moves)

    def get_name(self) -> str:
        return f"RandomAgent_P{self.player_id}"

class AlphaZeroAgent(BaseAgent):
    def __init__(self, player_id: PlayerId, player_info: PlayerInfo, model_path: str, 
                 mcts_simulations_eval: int, # Though not used in direct policy fallback yet
                 config: Any): # Pass the whole config object for parameters
        super().__init__(player_id, player_info)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config # Store config
        self.mcts_simulations_eval = mcts_simulations_eval # Store for potential MCTS use

        # Initialize model
        # Ensure config attributes exist, otherwise raise more informative error or use defaults
        if not all(hasattr(config, attr) for attr in ['INPUT_CHANNELS', 'GRID_ROWS', 'GRID_COLS']):
            raise AttributeError("AlphaZeroAgent: Config object is missing one or more required attributes (INPUT_CHANNELS, GRID_ROWS, GRID_COLS).")

        self.model = AlphaZeroNet(
            input_channels=self.config.INPUT_CHANNELS,
            grid_rows=self.config.GRID_ROWS,
            grid_cols=self.config.GRID_COLS
        ).to(self.device)

        if os.path.exists(model_path):
            try:
                # Ensure model is on CPU first before loading, then move to device
                # This avoids issues if model was saved on GPU and loading on CPU
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.to(self.device)
                # print(f"AlphaZeroAgent: Loaded model for P{self.player_id} from {model_path}") # Use logger
            except Exception as e:
                # print(f"AlphaZeroAgent WARNING: Error loading model {model_path} for P{self.player_id}: {e}. Using fresh model.") # Use logger
                # Initialize with random weights if load fails - model is already initialized above, so this re-initializes.
                # This ensures it's a fresh model if loading fails.
                self.model = AlphaZeroNet(
                    input_channels=self.config.INPUT_CHANNELS,
                    grid_rows=self.config.GRID_ROWS,
                    grid_cols=self.config.GRID_COLS
                ).to(self.device)
        else:
            # print(f"AlphaZeroAgent WARNING: Model path {model_path} not found for P{self.player_id}. Using fresh model.") # Use logger
            # Model is already initialized with random weights if path doesn't exist
            pass # Model already initialized fresh if path doesn't exist
            
        self.model.eval() # Set to evaluation mode

    def get_move(self, game_state: GameState) -> Position:
        # Fallback: Direct policy (less accurate representation of AlphaZero strength without MCTS)
        # Encode state from the perspective of the current player of the state
        # which should be self.player_id if it's this agent's turn.
        # Ensure current player is self.player_id, otherwise this encoding is from opponent's view
        if game_state.currentPlayerId != self.player_id:
            # This should not happen if the game loop calls agent.get_move() for the correct player.
            # print(f"Warning: AlphaZeroAgent {self.player_id} called to move, but current player is {game_state.currentPlayerId}")
            # Fallback or raise error, for now, proceeding but this indicates an issue.
            pass

        encoded_s = utils_module.encode_state(game_state, self.player_id).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(encoded_s)

        valid_moves = game_rules_module.get_valid_moves(game_state)
        if not valid_moves:
            # print(f"WARNING: AZ Agent {self.player_id} has no valid moves!") # Use logger
            return Position(row=0, col=0) # Fallback

        # Create a full policy mask
        # policy_logits is [1, num_actions]
        num_actions = self.config.GRID_ROWS * self.config.GRID_COLS
        masked_policy_logits = torch.full((1, num_actions), -float('inf'), device=self.device)

        valid_move_found_in_policy = False
        for move in valid_moves:
            move_idx = move.row * self.config.GRID_COLS + move.col
            if 0 <= move_idx < num_actions: # Should always be true for valid moves from valid_moves
                 masked_policy_logits[0, move_idx] = policy_logits[0, move_idx]
                 valid_move_found_in_policy = True
        
        if not valid_move_found_in_policy and valid_moves:
            # This means policy head is misaligned or all valid moves got -inf by chance from raw logits.
            # Fallback to random choice among valid moves if policy is all -inf for them.
            # print(f"Warning: AZ Agent {self.player_id} - no valid moves had finite logits. Choosing random valid move.") # Use logger
            chosen_move = random.choice(valid_moves)
            return Position(row=chosen_move.row, col=chosen_move.col)

        best_action_idx = torch.argmax(masked_policy_logits).item()
        
        # If all valid moves resulted in -inf (e.g. if valid_moves was empty and we didn't hit earlier check)
        # or if the argmax somehow picks an invalid action (shouldn't happen with masking)
        # This is a safeguard.
        final_row, final_col = divmod(best_action_idx, self.config.GRID_COLS)
        
        # Check if the chosen best_action_idx corresponds to a valid move.
        is_best_action_valid = any(vm.row == final_row and vm.col == final_col for vm in valid_moves)
        if not is_best_action_valid and valid_moves:
            # print(f"Warning: AZ Agent {self.player_id} - argmax chosen move is invalid. Choosing random valid move.") # Use logger
            chosen_move = random.choice(valid_moves)
            return Position(row=chosen_move.row, col=chosen_move.col)
            
        return Position(row=final_row, col=final_col)

    def get_name(self) -> str:
        # Model path can be long, so don't include it directly unless desired
        return f"AlphaZeroAgent_P{self.player_id}"

# Arena, load_agent_from_config, DEFAULT_PLAYER_INFOS will be added in the next step.

# Add BEST_MODEL_PATH from config for load_agent_from_config
try:
    # Import the main config module to pass to load_agent_from_config
    from . import config as main_config_module
except ImportError:
    # This is a critical error for load_agent_from_config if it needs main_config_module
    # print("CRITICAL Error: Could not import main_config_module in evaluation.py.") # Use logger if available
    main_config_module = None # Or handle more gracefully

# DEFAULT_PLAYER_INFOS should be defined before Arena or load_agent_from_config if used by them as defaults.
DEFAULT_PLAYER_INFOS = [
    PlayerInfo(id=1, name="P1", colorClass="bg-red-500", textColorClass="text-red-500", orbColorClass="orb-red", primaryHex="#FF0000", isAI=True, ai_type="alphazero"), # Example ai_type
    PlayerInfo(id=2, name="P2", colorClass="bg-blue-500", textColorClass="text-blue-500", orbColorClass="orb-blue", primaryHex="#0000FF", isAI=True, ai_type="random"), # Example ai_type
]

class Arena:
    def __init__(self, agent1: BaseAgent, agent2: BaseAgent, grid_size_config: GridSizeConfig, logger: Optional[Any] = None):
        self.agent1 = agent1
        self.agent2 = agent2
        self.grid_size_config = grid_size_config
        self.logger = logger # ExperimentLogger instance

    def play_game(self, start_player_id: PlayerId = 1) -> Tuple[Optional[PlayerId], int, GameState]:
        """Plays a single game between agent1 and agent2."""
        players = [self.agent1.player_info, self.agent2.player_info]
        
        game_state = game_rules_module.get_initial_state(self.grid_size_config, players)
        game_state.currentPlayerId = start_player_id
        game_state.status = "active" # Ensure game starts active

        turn_count = 0
        max_turns = self.grid_size_config.rows * self.grid_size_config.cols * 6 

        if self.logger:
            self.logger.log_debug(f"Arena: Starting game. P1: {self.agent1.get_name()} (ID: {self.agent1.player_id}), P2: {self.agent2.get_name()} (ID: {self.agent2.player_id}). Start: P{start_player_id}")

        while game_state.status == "active":
            current_player_for_move = game_state.currentPlayerId
            if current_player_for_move == self.agent1.player_id:
                current_agent = self.agent1
            elif current_player_for_move == self.agent2.player_id:
                current_agent = self.agent2
            else:
                if self.logger: self.logger.log_error(f"Arena: Unknown currentPlayerId {current_player_for_move} in game state. Agent1_ID: {self.agent1.player_id}, Agent2_ID: {self.agent2.player_id}")
                game_state.status = "draw" 
                game_state.winnerId = None
                break

            if self.logger: self.logger.log_debug(f"Arena: Turn {turn_count + 1}, Player {current_player_for_move} ({current_agent.get_name()}) to move.")
            
            move = current_agent.get_move(game_state)

            if not (0 <= move.row < self.grid_size_config.rows and 0 <= move.col < self.grid_size_config.cols):
                if self.logger: self.logger.log_error(f"Arena: Agent {current_agent.get_name()} made out-of-bounds move {move}.")
                game_state.status = "finished" 
                game_state.winnerId = self.agent1.player_id if current_agent == self.agent2 else self.agent2.player_id
                break
            
            cell = game_state.grid[move.row][move.col]
            if not (cell.player is None or cell.player == current_player_for_move):
                if self.logger: self.logger.log_error(f"Arena: Agent {current_agent.get_name()} made invalid move {move} on cell owned by P{cell.player} (current P{current_player_for_move}).")
                game_state.status = "finished" 
                game_state.winnerId = self.agent1.player_id if current_agent == self.agent2 else self.agent2.player_id
                break

            game_state = game_rules_module.apply_move(game_state, move, current_player_for_move) 
            turn_count += 1

            if turn_count >= max_turns and game_state.status == "active":
                if self.logger: self.logger.log_info(f"Arena: Game ended in a draw due to max turns ({max_turns}).")
                game_state.status = "draw"
                game_state.winnerId = None
                break
        
        if self.logger: self.logger.log_debug(f"Arena: Game finished. Winner: P{game_state.winnerId}, Status: {game_state.status}, Turns: {turn_count}")
        return game_state.winnerId, turn_count, game_state

    def run_matches(self, num_games: int) -> Dict[str, Any]:
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        total_turns = 0

        for i in range(num_games):
            start_player_id = self.agent1.player_id if i % 2 == 0 else self.agent2.player_id
            
            winner_id, turns, final_state = self.play_game(start_player_id)
            total_turns += turns

            if winner_id == self.agent1.player_id:
                agent1_wins += 1
            elif winner_id == self.agent2.player_id:
                agent2_wins += 1
            else: 
                draws += 1

            if self.logger:
                 self.logger.log_info(f"Match {i+1}/{num_games} ({self.agent1.get_name()} vs {self.agent2.get_name()}): Winner P{winner_id}, Turns: {turns}, Start P{start_player_id}")
        
        num_games_played = agent1_wins + agent2_wins + draws 
        if num_games_played == 0: num_games_played = 1 # Avoid division by zero

        return {
            "agent1_name": self.agent1.get_name(),
            "agent2_name": self.agent2.get_name(),
            "agent1_wins": agent1_wins,
            "agent2_wins": agent2_wins,
            "draws": draws,
            "num_games": num_games, 
            "avg_turns": total_turns / num_games_played,
            "agent1_win_ratio": agent1_wins / num_games_played
        }

def load_agent_from_config(agent_config_dict: Dict[str, Any], player_id: PlayerId, player_info: PlayerInfo, 
                           mcts_sims_for_eval: int, config_obj: Any) -> BaseAgent: # Renamed to config_obj
    """Helper to create agents based on config."""
    agent_type = agent_config_dict.get("type", "random")
    model_path_from_opponent_config = agent_config_dict.get("path")

    if agent_type == "random":
        return RandomAgent(player_id, player_info)
    elif agent_type == "alphazero" or agent_type == "previous_best" or agent_type == "specific_checkpoint":
        model_path_to_load = model_path_from_opponent_config
        if agent_type == "previous_best":
            if not hasattr(config_obj, 'BEST_MODEL_PATH'):
                raise ValueError("BEST_MODEL_PATH not found in config for 'previous_best' agent.")
            model_path_to_load = config_obj.BEST_MODEL_PATH
        
        if not model_path_to_load: # Should only happen if type is 'alphazero' or 'specific_checkpoint' and path is missing
            raise ValueError(f"Model path required for AlphaZero agent type '{agent_type}' but not found. Opponent config: {agent_config_dict}")

        # Pass config_obj to AlphaZeroAgent, which will use attributes like INPUT_CHANNELS, GRID_ROWS, GRID_COLS
        return AlphaZeroAgent(player_id, player_info, model_path_to_load, mcts_sims_for_eval, config_obj)
    else:
        raise ValueError(f"Unknown agent type in evaluation opponent config: {agent_type}")

# Example of how it might be used (for testing evaluation.py itself)

def run_evaluation(current_model_iteration: int, config: Any, logger: 'ExperimentLogger') -> Dict[str, Any]:
    """
    Orchestrates the evaluation of a given model iteration against configured opponents.
    """
    logger.log_info(f"--- Starting Evaluation for Model from Iteration {current_model_iteration} ---")

    # Construct path to the current model to be evaluated
    # CHECKPOINT_SAVE_PATH_FORMAT is expected to be like "alphazero_iter_{iteration}.pth" (filename pattern)
    # or a full path pattern like "ai/trained_models/checkpoints/alphazero_iter_{iteration}.pth"
    
    # Let's assume CHECKPOINT_SAVE_PATH_FORMAT might be just a filename pattern first
    filename_pattern = getattr(config, 'CHECKPOINT_SAVE_PATH_FORMAT', 'alphazero_iter_{iteration}.pth')
    current_model_filename = os.path.basename(filename_pattern.format(iteration=current_model_iteration))
    
    if hasattr(config, 'CHECKPOINT_DIR'):
        current_model_path = os.path.join(config.CHECKPOINT_DIR, current_model_filename)
    else: # Fallback if CHECKPOINT_DIR is not in config, assume format includes path
        logger.log_warning("CHECKPOINT_DIR not in config, assuming CHECKPOINT_SAVE_PATH_FORMAT is a full path.")
        current_model_path = filename_pattern.format(iteration=current_model_iteration)


    if not os.path.exists(current_model_path):
        logger.log_error(f"Evaluation failed: Model checkpoint for iteration {current_model_iteration} not found at {current_model_path}")
        return {"error": f"Model not found: {current_model_path}"}

    player1_info = DEFAULT_PLAYER_INFOS[0] 
    player2_info = DEFAULT_PLAYER_INFOS[1]

    current_az_agent_config = {
        "type": "alphazero", 
        "name": f"Iter_{current_model_iteration}", 
        "path": current_model_path
    }
    
    try:
        agent_current_model = load_agent_from_config(
            current_az_agent_config, 
            player1_info.id, 
            player1_info, 
            config.MCTS_SIMULATIONS_EVAL, 
            config 
        )
    except Exception as e:
        logger.log_error(f"Failed to load current model agent for iteration {current_model_iteration} from {current_model_path}: {e}", exc_info=True)
        return {"error": f"Failed to load current model agent: {e}"}

    evaluation_summary_results = {}
    logger.log_info(f"Evaluating model: {agent_current_model.get_name()} (Path: {current_model_path})")

    for opponent_conf in config.EVALUATION_OPPONENTS:
        opponent_name = opponent_conf.get("name", opponent_conf["type"]) 
        logger.log_info(f"--- Matchup: {agent_current_model.get_name()} vs {opponent_name} ---")

        try:
            opponent_agent = load_agent_from_config(
                opponent_conf, 
                player2_info.id, 
                player2_info, 
                config.MCTS_SIMULATIONS_EVAL, 
                config 
            )
        except Exception as e:
            logger.log_error(f"Failed to load opponent agent {opponent_name}: {e}", exc_info=True)
            evaluation_summary_results[opponent_name] = {"error": f"Failed to load opponent: {e}"}
            continue 

        arena = Arena(agent_current_model, opponent_agent, config.GRID_SIZE, logger)
        match_results = arena.run_matches(config.EVALUATION_GAMES_PER_OPPONENT)
        
        evaluation_summary_results[opponent_name] = match_results

        win_rate_key = f"evaluation/{opponent_name}/win_rate_vs_opponent" 
        avg_turns_key = f"evaluation/{opponent_name}/avg_turns_vs_opponent"
        
        logger.log_scalar(win_rate_key, match_results['agent1_win_ratio'], current_model_iteration)
        logger.log_scalar(avg_turns_key, match_results['avg_turns'], current_model_iteration)
        logger.log_info(f"Results vs {opponent_name}: {match_results}")

        if opponent_conf["type"] == "previous_best":
            # Ensure BEST_MODEL_PATH directory exists before attempting to copy
            best_model_dir = os.path.dirname(config.BEST_MODEL_PATH)
            if best_model_dir: # Ensure directory is not empty string if BEST_MODEL_PATH is just a filename
                 os.makedirs(best_model_dir, exist_ok=True)

            if not os.path.exists(config.BEST_MODEL_PATH):
                logger.log_info(f"No existing best model found at {config.BEST_MODEL_PATH}. Current model will become the best.")
                try:
                    shutil.copy(current_model_path, config.BEST_MODEL_PATH)
                    logger.log_info(f"Copied {current_model_path} to {config.BEST_MODEL_PATH} as initial best model.")
                except Exception as e:
                    logger.log_error(f"Error copying {current_model_path} to {config.BEST_MODEL_PATH}: {e}", exc_info=True)

            elif match_results['agent1_win_ratio'] > config.WIN_RATIO_THRESHOLD_FOR_BEST_MODEL:
                logger.log_info(f"New best model found! Iteration {current_model_iteration} ({match_results['agent1_win_ratio']:.2%}) vs Previous Best (Threshold > {config.WIN_RATIO_THRESHOLD_FOR_BEST_MODEL:.0%}).")
                try:
                    shutil.copy(current_model_path, config.BEST_MODEL_PATH)
                    logger.log_info(f"Copied {current_model_path} to {config.BEST_MODEL_PATH}")
                except Exception as e:
                    logger.log_error(f"Error copying {current_model_path} to {config.BEST_MODEL_PATH}: {e}", exc_info=True)
            else:
                logger.log_info(f"Current model (Iter {current_model_iteration}) did not surpass previous best model. Win rate: {match_results['agent1_win_ratio']:.2%}, Threshold: {config.WIN_RATIO_THRESHOLD_FOR_BEST_MODEL:.0%}")
    
    logger.log_info(f"--- Evaluation for Model from Iteration {current_model_iteration} Complete ---")
    return evaluation_summary_results

# if __name__ == '__main__':
#     if main_config_module: # Check if import was successful
#         from .monitoring import ExperimentLogger
#         logger = ExperimentLogger(run_name="evaluation_test", config=main_config_module)
#
#         p1_info = DEFAULT_PLAYER_INFOS[0]
#         p2_info = DEFAULT_PLAYER_INFOS[1]
#
#         # Create a dummy best_model.pth for testing AlphaZeroAgent loading
#         # Ensure AlphaZeroNet can be initialized with config values
#         # dummy_nn = AlphaZeroNet(main_config_module.INPUT_CHANNELS, main_config_module.GRID_ROWS, main_config_module.GRID_COLS)
#         # model_dir = os.path.dirname(main_config_module.BEST_MODEL_PATH)
#         # if model_dir: os.makedirs(model_dir, exist_ok=True)
#         # else: print(f"Warning: BEST_MODEL_PATH '{main_config_module.BEST_MODEL_PATH}' is unusual.")
#         # if hasattr(main_config_module, 'BEST_MODEL_PATH'):
#         #    torch.save(dummy_nn.state_dict(), main_config_module.BEST_MODEL_PATH)
#     
#         agent1_config = {"type": "previous_best", "name": "BestAZSoFar"} 
#         agent1 = load_agent_from_config(agent1_config, p1_info.id, p1_info, main_config_module.MCTS_SIMULATIONS_EVAL, main_config_module)
#         agent2 = RandomAgent(p2_info.id, p2_info)
#
#         arena = Arena(agent1, agent2, main_config_module.GRID_SIZE, logger)
#         results = arena.run_matches(num_games=4) 
#         logger.log_info(f"Arena Test Results: {results}")
#         logger.close()
#     else:
#         print("Cannot run __main__ test for evaluation.py: main_config_module not loaded.")
