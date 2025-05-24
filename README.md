# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## Running with Python AI Backend

This project can optionally use a Python-based AI for game moves.

**Prerequisites for Python AI:**
*   Python 3.7+
*   `pip` (Python package installer)

**Setup and Running:**

1.  **Frontend (React App):**
    *   Follow the "Run Locally" instructions above (install Node.js dependencies and run `npm run dev` or `npm start`).
    *   The frontend will typically run on `http://localhost:5173` (Vite) or `http://localhost:3000` (Create React App).

2.  **Backend (Python FastAPI Server):**
    *   Navigate to the `ai` directory in your terminal:
        ```bash
        cd ai
        ```
    *   Create a virtual environment (recommended):
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    *   Install Python dependencies:
        ```bash
        pip install fastapi "uvicorn[standard]" pydantic
        ```
    *   Run the FastAPI server:
        ```bash
        uvicorn server:app --reload --port 8000
        ```
    *   The AI server will run on `http://localhost:8000`.

**How it Works:**

*   When it's an AI player's turn (configured in the game setup), the React frontend will make an API request to the Python backend at `http://localhost:8000/get-ai-move`.
*   The Python server will process the game state and return an AI-chosen move.
*   The frontend will then apply this move to the game.

**Important:** Both the React development server and the Python FastAPI server must be running concurrently for the AI functionality to work.
