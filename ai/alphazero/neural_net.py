import torch
import torch.nn as nn
import torch.nn.functional as F # For potential future use, like LogSoftmax if applied in model

class AlphaZeroNet(nn.Module):
  """
  Neural Network for AlphaZero.

  The network consists of a common convolutional trunk followed by separate
  policy and value heads.
  """
  def __init__(self, input_channels: int, grid_rows: int, grid_cols: int):
    """
    Initializes the AlphaZeroNet.

    Args:
      input_channels: Number of channels in the input state tensor (e.g., 5).
      grid_rows: Number of rows in the game grid.
      grid_cols: Number of columns in the game grid.
    """
    super().__init__()
    num_actions = grid_rows * grid_cols
    self.grid_rows = grid_rows
    self.grid_cols = grid_cols

    # Common Convolutional Trunk
    self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(128)
    self.relu1 = nn.ReLU()
    
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.relu2 = nn.ReLU()

    # Policy Head
    self.policy_conv = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0) # 1x1 conv
    self.policy_bn = nn.BatchNorm2d(32)
    self.policy_relu = nn.ReLU()
    self.flatten_policy = nn.Flatten()
    # In_features: 32 channels * grid_rows * grid_cols
    self.policy_fc = nn.Linear(32 * self.grid_rows * self.grid_cols, num_actions)

    # Value Head
    self.value_conv = nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0) # 1x1 conv
    self.value_bn = nn.BatchNorm2d(4)
    self.value_relu = nn.ReLU()
    self.flatten_value = nn.Flatten()
    # In_features: 4 channels * grid_rows * grid_cols
    self.value_fc1 = nn.Linear(4 * self.grid_rows * self.grid_cols, 64)
    self.value_fc1_relu = nn.ReLU()
    self.value_fc2 = nn.Linear(64, 1)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass of the network.

    Args:
      x: Input tensor representing the game state.
         Shape: (batch_size, input_channels, grid_rows, grid_cols)

    Returns:
      A tuple containing:
        - policy_logits: Raw logits for the policy (actions).
                         Shape: (batch_size, num_actions)
        - value: Estimated value of the state, scaled to [-1, 1] by tanh.
                 Shape: (batch_size, 1)
    """
    # Common Trunk
    x = self.relu1(self.bn1(self.conv1(x)))
    x = self.relu2(self.bn2(self.conv2(x)))

    # Policy Head
    policy_x = self.policy_relu(self.policy_bn(self.policy_conv(x)))
    policy_x = self.flatten_policy(policy_x)
    policy_logits = self.policy_fc(policy_x)

    # Value Head
    value_x = self.value_relu(self.value_bn(self.value_conv(x)))
    value_x = self.flatten_value(value_x)
    value_x = self.value_fc1_relu(self.value_fc1(value_x))
    value = self.value_fc2(value_x)
    
    return policy_logits, torch.tanh(value)

  def save_model(self, path: str):
    """
    Saves the model's state dictionary to the given path.

    Args:
      path: The file path where the model state will be saved.
    """
    torch.save(self.state_dict(), path)
    print(f"Model saved to {path}")

  def load_model(self, path: str, device: torch.device):
    """
    Loads the model's state dictionary from the given path.

    Args:
      path: The file path from which to load the model state.
      device: The torch device to map the loaded model to (e.g., 'cpu', 'cuda').
    """
    self.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path} to device {device}")

# Example Usage (not part of the class, just for illustration)
if __name__ == '__main__':
    # Configuration
    INPUT_CHANNELS = 5  # Example: (own_orbs, opp_orbs, own_cells, opp_cells, turn_plane)
    GRID_ROWS = 6
    GRID_COLS = 7
    
    # Create model instance
    model = AlphaZeroNet(input_channels=INPUT_CHANNELS, grid_rows=GRID_ROWS, grid_cols=GRID_COLS)
    
    # Create a dummy input tensor (batch_size=2 for example)
    batch_size = 2
    dummy_input = torch.randn(batch_size, INPUT_CHANNELS, GRID_ROWS, GRID_COLS)
    
    # Perform a forward pass
    model.eval() # Set to evaluation mode for inference
    with torch.no_grad(): # Disable gradient calculations for inference
        policy_logits_output, value_output = model(dummy_input)
    
    print("Policy Logits Output Shape:", policy_logits_output.shape)
    print("Value Output Shape:", value_output.shape)
    
    # Example of saving and loading
    # model_path = "alphazero_net_example.pth"
    # model.save_model(model_path)
    
    # new_model = AlphaZeroNet(input_channels=INPUT_CHANNELS, grid_rows=GRID_ROWS, grid_cols=GRID_COLS)
    # new_model.load_model(model_path, torch.device('cpu'))
    # print("New model loaded and ready.")
    
    # # Verify new model can perform forward pass
    # new_model.eval()
    # with torch.no_grad():
    #     policy_logits_new, value_new = new_model(dummy_input)
    # print("New Policy Logits Output Shape:", policy_logits_new.shape)
    # print("New Value Output Shape:", value_new.shape)
    # assert torch.allclose(policy_logits_output, policy_logits_new)
    # assert torch.allclose(value_output, value_new)
    # print("Outputs from original and loaded model match.")