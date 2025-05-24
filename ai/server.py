from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Make sure models and bot are imported with relative paths
from ai.models import GameState, Position
from ai.bot import get_ai_player_move # Changed from get_random_ai_move

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173", # Default Vite port
    "http://localhost:3000", # Default Create React App port
    # Add any other origins if your frontend runs on a different port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

@app.post("/get-ai-move", response_model=Position)
async def get_ai_move_endpoint(game_state: GameState) -> Position:
    """
    Receives the current game state and returns the AI's next move.
    """
    if game_state.currentPlayerId is None:
        raise HTTPException(status_code=400, detail="currentPlayerId cannot be null")

    if game_state.status != "active":
        raise HTTPException(status_code=400, detail=f"Game status must be 'active' to get an AI move, got '{game_state.status}'")
        
    # Find the AI player. We assume the currentPlayerId is the AI.
    # More robust logic might involve checking player.isAI if multiple AIs or AI vs AI.
    ai_player = next((p for p in game_state.players if p.id == game_state.currentPlayerId), None)
    
    if not ai_player:
        raise HTTPException(status_code=404, detail=f"Current player with ID {game_state.currentPlayerId} not found in players list.")

    # It's good practice to ensure the current player is marked as AI,
    # though the current issue implies currentPlayerId will be the AI.
    # if not ai_player.isAI:
    #     raise HTTPException(status_code=400, detail="Current player is not an AI.")

    move = get_ai_player_move(game_state) # Changed from get_random_ai_move

    if move is None:
        # This might happen if no valid moves are available,
        # or if the AI logic itself returns None (e.g. game over).
        raise HTTPException(status_code=500, detail="AI was unable to determine a move.")
    
    return move

# To run this server, navigate to the 'ai' directory and use:
# uvicorn server:app --reload --port 8000
# Ensure you have installed fastapi, uvicorn, and pydantic:
# pip install fastapi "uvicorn[standard]" pydantic
