from typing import List, Optional, Literal
from pydantic import BaseModel

PlayerId = int  # 1 for Player 1, 2 for Player 2, etc.

class CellState(BaseModel):
    player: Optional[PlayerId] = None # null for empty, or PlayerId
    orbs: int

Grid = List[List[CellState]]

class Position(BaseModel):
    row: int
    col: int

class PlayerInfo(BaseModel):
    id: PlayerId
    name: str
    colorClass: str # Tailwind background class e.g., 'bg-rose-600'
    textColorClass: str # Tailwind text class e.g., 'text-rose-500'
    orbColorClass: str # Tailwind class for individual orbs if visualized
    primaryHex: str # Hex color string for custom elements like cursors
    isAI: Optional[bool] = False # Added to identify AI players
    ai_type: Optional[str] = "random"  # Specifies AI behavior: "random", "alphazero", etc.

# Using Literal for GameStatus as it's a string enum in TypeScript
GameStatus = Literal["setup", "active", "finished", "draw"]

class GridSizeConfig(BaseModel):
    rows: int
    cols: int

class GameState(BaseModel):
    grid: Grid
    players: List[PlayerInfo]
    currentPlayerId: Optional[PlayerId] = None
    status: GameStatus
    winnerId: Optional[PlayerId] = None
    gridSize: GridSizeConfig
    turnNumber: int # Represents the current round number
