import math
import numpy as np
import random
from __future__ import annotations # For type hint Node in parent: Optional[Node]
from typing import Optional, Dict, List, Tuple, Any # Added Tuple
from ai.models import GameState, Position, PlayerId, GridSizeConfig # Added GridSizeConfig
from ai.alphazero.neural_net import YourNeuralNet # Added YourNeuralNet

class Node:
    def __init__(self, state: GameState, player_id_at_node: PlayerId, parent: Optional[Node] = None, prior_probability: float = 0.0):
        self.state: GameState = state
        self.parent: Optional[Node] = parent
        self.children: Dict[Position, Node] = {}
        self.visit_count: int = 0
        self.total_action_value: float = 0.0
        self.prior_probability: float = prior_probability # This is the P(s,a) for the edge leading to *this* node
        self.player_id_at_node: PlayerId = player_id_at_node


def select_child(node: Node, c_puct: float) -> tuple[Position, Node]:
    """
    Selects the child node that maximizes the PUCT value.
    U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    Q(s,a) = W(s,a) / N(s,a) (from perspective of current player at node)
    PUCT = Q(s,a) + U(s,a)

    Assumes this is a two-player alternating game.
    total_action_value in a child Node is accumulated from the perspective of child_node.player_id_at_node.
    If child_node.player_id_at_node is the opponent of node.player_id_at_node,
    then the value child_node.total_action_value / child_node.visit_count is good for the opponent,
    meaning it's bad for the current player at `node`. Hence, it should be negated for Q(s,a).
    """
    if not node.children:
        raise ValueError("Cannot select child from a node with no children (e.g., unexpanded or terminal node).")

    best_score = -float('inf')
    best_action: Optional[Position] = None # Initialized to None, but must be set if node.children is not empty
    best_child_node: Optional[Node] = None # Initialized to None, but must be set

    parent_visit_count = node.visit_count # This is N(s)

    for action, child_node in node.children.items():
        if child_node.visit_count == 0:
            q_value = 0.0  # Q for unvisited children is 0
        else:
            # child_node.total_action_value is from perspective of player at child_node.
            # Q(s,a) must be from perspective of player at current node (parent).
            # If players alternate, player at child is opponent, so negate the value.
            q_value = - (child_node.total_action_value / child_node.visit_count)

        # U(s,a) - Exploration term
        # child_node.prior_probability is P(s,a)
        # parent_visit_count is N(s)
        # child_node.visit_count is N(s,a)
        u_value = c_puct * child_node.prior_probability * math.sqrt(parent_visit_count) / (1 + child_node.visit_count)
        
        score = q_value + u_value

        if score > best_score:
            best_score = score
            best_action = action
            best_child_node = child_node
    
    # If node.children was not empty, best_action and best_child_node should have been set.
    # (because the first child encountered would set them, unless all scores are -inf somehow, which is not expected here)
    if best_action is None or best_child_node is None:
        # This should ideally not be reached if node.children is non-empty.
        # Could happen if c_puct or priors are such that all scores are -inf, or if children list was empty
        # despite the initial check (e.g. concurrency issue, though not relevant here).
        # Or if all priors are 0 and parent_visit_count is 0, all u_values are 0. All q_values are 0. All scores are 0.
        # In this case, the first child would be chosen.
        # This path implies an unexpected state or logic error if node.children was guaranteed non-empty.
        raise Exception("Internal error: No child selected despite node having children. This should not happen.")

    return best_action, best_child_node


def run_simulations(
    root_state: GameState, 
    current_player_id: PlayerId, 
    nn: YourNeuralNet, # Changed Any to YourNeuralNet
    num_simulations: int, 
    game_rules_module: Any, 
    c_puct: float, 
    utils_module: Any, 
    config_module: Any # Passed for consistency, may not be used directly here
) -> Node:
    """
    Runs MCTS simulations to build a search tree and estimate action values.
    """
    root_node = Node(state=root_state, player_id_at_node=current_player_id, parent=None, prior_probability=0.0)

    for _ in range(num_simulations):
        current_node: Node = root_node
        path_taken: List[Node] = [current_node]

        # a. Selection Phase
        # Traverse the tree selecting the child with the highest PUCT value until a leaf node or terminal state is reached.
        # A node is a leaf if it has not been expanded yet (no children).
        # The selection loop condition `len(current_node.children) > 0` ensures we stop if it's an unexpanded leaf.
        while len(current_node.children) > 0 and not game_rules_module.is_terminal(current_node.state):
            _action, child_to_explore = select_child(current_node, c_puct) # _action is not used here
            current_node = child_to_explore # Update current_node to the selected child
            path_taken.append(current_node)

        # b. Expansion & Evaluation Phase
        value: float
        # This is the player for whom the NN's value prediction is made, or for whom the terminal reward is directly calculated.
        effective_value_perspective_player_id: PlayerId 
        player_for_nn_value = current_node.player_id_at_node # Player whose perspective NN value is from

        if game_rules_module.is_terminal(current_node.state):
            # Terminal node reached during selection or if a newly selected leaf is terminal.
            value = game_rules_module.get_reward(current_node.state, root_node.player_id_at_node)
            # Reward is from the perspective of root_node.player_id_at_node
            effective_value_perspective_player_id = root_node.player_id_at_node
        else:
            # Node is a new leaf (current_node.children is empty) or was a leaf and is now being expanded.
            # The selection loop ensures that if current_node.children was not empty, we would have selected a child.
            # So, if we are here and not terminal, current_node.children should be empty.

            encoded_state = utils_module.encode_state(current_node.state) # As per new prompt
            policy_logits, value_from_nn = nn.predict(encoded_state) # value_from_nn is for `player_for_nn_value`
            value = float(value_from_nn) # Ensure it's a Python float
            effective_value_perspective_player_id = player_for_nn_value

            valid_moves: List[Position] = game_rules_module.get_valid_moves(current_node.state)

            if not valid_moves:
                # This means it's a terminal state (e.g. stalemate not caught by is_terminal, or rules error)
                # Re-evaluate reward as if terminal. This might override NN's value if it was a stalemate.
                value = game_rules_module.get_reward(current_node.state, root_node.player_id_at_node)
                effective_value_perspective_player_id = root_node.player_id_at_node
                # No children to add, current_node remains a leaf for this path.
            else:
                # Expansion of current_node (which is a leaf)
                move_to_idx_map = utils_module.get_move_to_policy_idx_map(current_node.state.gridSize) # As per new prompt
                # idx_to_move_map = {v: k for k, v in move_to_idx_map.items()} # Not strictly needed by this logic

                masked_logits = np.full(len(policy_logits), -np.inf, dtype=float)
                valid_move_indices = [] # Indices in policy_logits for valid, mappable moves
                valid_moves_mapped_to_policy = [] # Corresponding Position objects for these moves

                for move in valid_moves:
                    idx = move_to_idx_map.get(move)
                    if idx is not None and 0 <= idx < len(policy_logits):
                        masked_logits[idx] = policy_logits[idx]
                        valid_move_indices.append(idx)
                        valid_moves_mapped_to_policy.append(move)
                
                if not valid_move_indices:
                    # No valid moves from game_rules could be mapped to the NN policy output.
                    # This could be a stalemate or an error. The value is already from NN.
                    # No children are added. It remains a leaf for this path.
                    pass # current_node remains a leaf for this simulation path.
                else:
                    # Apply softmax to the logits of valid moves that are in the policy map
                    relevant_logits = masked_logits[valid_move_indices] # This uses the filtered indices
                    
                    # Softmax calculation:
                    exp_logits = np.exp(relevant_logits - np.max(relevant_logits)) # Subtract max for numerical stability
                    sum_exp_logits = np.sum(exp_logits)
                    
                    if sum_exp_logits > 1e-8: # Check for non-zero sum
                        probs_for_valid_mapped_moves = exp_logits / sum_exp_logits
                    else:
                        # All valid, mappable moves had -inf logits or sum is numerically zero.
                        # Fallback to uniform probability for these valid, mapped moves.
                        probs_for_valid_mapped_moves = np.ones(len(valid_move_indices)) / len(valid_move_indices) if len(valid_move_indices) > 0 else np.array([])

                    if probs_for_valid_mapped_moves.size > 0:
                        next_player_id = game_rules_module.get_next_player_id(current_node.state, current_node.player_id_at_node)
                        for i, move in enumerate(valid_moves_mapped_to_policy):
                            prob = probs_for_valid_mapped_moves[i]
                            # Create new state by applying the move
                            new_state_after_move = game_rules_module.apply_move(current_node.state, move, current_node.player_id_at_node)
                            current_node.children[move] = Node(
                                state=new_state_after_move,
                                parent=current_node,
                                prior_probability=prob,
                                player_id_at_node=next_player_id
                            )
        
        # c. Backpropagation Phase
        # Update visit counts and total action values for all nodes in the path taken.
        for node_in_path in reversed(path_taken):
            node_in_path.visit_count += 1
            # Value is from the perspective of 'effective_value_perspective_player_id'
            # If the player at node_in_path is different from the one whose perspective 'value' holds, negate the value.
            if node_in_path.player_id_at_node != effective_value_perspective_player_id:
                node_in_path.total_action_value -= value 
            else:
                node_in_path.total_action_value += value
                
    return root_node


def get_best_move(root_node: Node, temperature: float = 1.0) -> Position:
    """
    Selects the best move from the root_node based on visit counts and temperature.
    Temperature = 0 means greedy selection (choose the most visited).
    Temperature > 0 means stochastic selection, weighted by visit_counts**(1/temperature).
    """
    if not root_node.children:
        raise ValueError("Root node has no children, cannot determine best move.")

    moves = list(root_node.children.keys())
    visit_counts = np.array([child.visit_count for child in root_node.children.values()], dtype=np.float32)

    if temperature == 0:
        best_move_idx = np.argmax(visit_counts)
        return moves[best_move_idx]
    else:
        # Stochastic selection based on temperature
        if np.any(visit_counts < 0):
            raise ValueError("Visit counts cannot be negative.")

        if len(moves) == 0: # Should be caught by the first check, but as a safeguard
             raise ValueError("No moves available in root node's children.")

        # If all visit_counts are 0, distribute probability uniformly.
        if np.all(visit_counts == 0):
            probabilities = np.ones(len(moves)) / len(moves)
        else:
            inv_temp = 1.0 / temperature
            
            # Handle temperature close to zero, which makes inv_temp infinity
            if np.isinf(inv_temp):
                # This would make probs for max visit count 1, others 0.
                # Equivalent to argmax.
                best_move_idx = np.argmax(visit_counts)
                return moves[best_move_idx]

            powered_counts = visit_counts ** inv_temp
            
            # Check for NaNs or Infs which can occur
            # np.isinf(powered_counts) can be an array, so use np.any
            if np.any(np.isnan(powered_counts)) or np.any(np.isinf(powered_counts)):
                # This can happen if inv_temp is very large and a count is 0,
                # or if inv_temp is negative (temperature < 0, not standard).
                # If inv_temp is positive: 0^inv_temp = 0. Non-zero^inv_temp can be huge (Inf).
                
                # Check if any powered_counts became Inf
                is_inf_mask = np.isinf(powered_counts)
                if np.any(is_inf_mask):
                    inf_indices = np.where(is_inf_mask)[0]
                    # If there are Infs, select randomly among them
                    chosen_idx_among_infs = random.choice(inf_indices)
                    return moves[chosen_idx_among_infs]
                else:
                    # No infs, but NaNs (e.g. 0 ** negative_inv_temp) or all zeros after power.
                    # Fallback to uniform probability.
                    probabilities = np.ones(len(moves)) / len(moves)
            else:
                # powered_counts are finite and non-NaN
                sum_powered_counts = np.sum(powered_counts)
                if sum_powered_counts > 1e-8:  # Avoid division by zero or near-zero
                    probabilities = powered_counts / sum_powered_counts
                else:
                    # All counts were effectively zero after powering, or sum is numerically zero.
                    # Fallback to uniform probability.
                    probabilities = np.ones(len(moves)) / len(moves)
        
        # Ensure probabilities sum to 1 (can have small deviations due to float precision)
        # probabilities = probabilities / np.sum(probabilities) # Re-normalize if concerned
        
        chosen_idx = np.random.choice(len(moves), p=probabilities)
        return moves[chosen_idx]
