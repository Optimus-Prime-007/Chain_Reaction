import React, { useState, useEffect, useCallback } from "react";
import Board from "./components/Board";
import GameInfoPanel from "./components/GameInfoPanel";
import HomePage from "./components/HomePage";
import RulesPage from "./components/RulesPage";
import GameSetupPage from "./components/GameSetupPage";
import ConfirmLeaveModal from "./components/ConfirmLeaveModal"; // New
import {
  GameState,
  PlayerId,
  GameStatus,
  Grid,
  Position,
  PlayerInfo,
  GameSetupOptions,
  GridSizeOption,
} from "./types";
import { AVAILABLE_PLAYER_COLORS, GRID_SIZE_OPTIONS } from "./constants";
import {
  getCriticalMass,
  createInitialGrid,
  deepCopyGrid,
  getNeighbors,
} from "./utils/gameLogic";

type View = "home" | "rules" | "setup" | "game";

const createGameStateFromOptions = (options: GameSetupOptions): GameState => {
  const selectedGridSize =
    GRID_SIZE_OPTIONS.find((gs) => gs.name === options.gridSizeName) ||
    GRID_SIZE_OPTIONS[0];

  const players: PlayerInfo[] = options.playerSettings.map((pSetup) => {
    const colorDetail =
      AVAILABLE_PLAYER_COLORS.find((c) => c.name === pSetup.colorName) ||
      AVAILABLE_PLAYER_COLORS[0];
    return {
      id: pSetup.id,
      name: pSetup.name,
      colorClass: colorDetail.bgClass,
      textColorClass: colorDetail.textClass,
      orbColorClass: colorDetail.orbClass,
      primaryHex: colorDetail.primaryHex || "#FFFFFF",
      isAI: pSetup.isAI || false,
    };
  });

  return {
    grid: createInitialGrid(selectedGridSize.rows, selectedGridSize.cols),
    players: players,
    currentPlayerId: players.length > 0 ? players[0].id : null,
    status: GameStatus.Active,
    winnerId: null,
    gridSize: { rows: selectedGridSize.rows, cols: selectedGridSize.cols },
    turnNumber: 1,
  };
};

const App: React.FC = () => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isProcessingMove, setIsProcessingMove] = useState(false);
  const [currentView, setCurrentView] = useState<View>("home");

  const [showLeaveConfirmationModal, setShowLeaveConfirmationModal] =
    useState(false);
  const [targetViewForNavigation, setTargetViewForNavigation] =
    useState<View | null>(null);

  const handleStartNewGameWithSetup = useCallback(
    (options: GameSetupOptions) => {
      const newGameState = createGameStateFromOptions(options);
      setGameState(newGameState);
      setIsProcessingMove(false);
      setCurrentView("game");
    },
    []
  );

  const requestNavigation = (target: View) => {
    if (currentView === "game" && gameState?.status === GameStatus.Active) {
      setTargetViewForNavigation(target);
      setShowLeaveConfirmationModal(true);
    } else {
      setCurrentView(target);
      if (target === "setup" && currentView === "game") {
        // For now, simply navigating. Game state will be overwritten by onStartGame.
      }
    }
  };

  const confirmAndNavigate = () => {
    if (targetViewForNavigation) {
      setCurrentView(targetViewForNavigation);
    }
    setShowLeaveConfirmationModal(false);
    setTargetViewForNavigation(null);
  };

  const cancelNavigation = () => {
    setShowLeaveConfirmationModal(false);
    setTargetViewForNavigation(null);
  };

  useEffect(() => {
    if (
      currentView === "game" &&
      gameState?.status === GameStatus.Active &&
      gameState.currentPlayerId
    ) {
      const currentPlayer = gameState.players.find(
        (p) => p.id === gameState.currentPlayerId
      );
      if (currentPlayer && !currentPlayer.isAI && currentPlayer.primaryHex) {
        const playerHexColor = currentPlayer.primaryHex;
        const strokeColor =
          playerHexColor.toLowerCase() === "#ffffff" ||
          playerHexColor.toLowerCase() === "#fffffe"
            ? "black"
            : "white";
        const cursorSvg = `<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9" fill="${playerHexColor}" stroke="${strokeColor}" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="${strokeColor}"/></svg>`;
        const encodedSvg = encodeURIComponent(cursorSvg);
        document.body.style.cursor = `url('data:image/svg+xml;charset=UTF-8,${encodedSvg}') 12 12, auto`;
      } else {
        document.body.style.cursor = "auto";
      }
    } else {
      document.body.style.cursor = "auto";
    }
    return () => {
      document.body.style.cursor = "auto";
    };
  }, [currentView, gameState]);

  // Effect for AI Turns using Python Backend
  useEffect(() => {
    if (
      !gameState ||
      gameState.status !== GameStatus.Active ||
      !gameState.currentPlayerId ||
      isProcessingMove
    ) {
      return;
    }

    const currentPlayer = gameState.players.find(
      (p) => p.id === gameState.currentPlayerId
    );

    if (currentPlayer?.isAI) {
      setIsProcessingMove(true); // Prevent human interaction and AI re-triggering
      console.log(
        `AI Player (${currentPlayer.name}, ID: ${currentPlayer.id}) is thinking... Attempting to fetch move from Python server.`
      );

      const thinkingDelay = 1000; // 1 second "thinking" delay on frontend

      setTimeout(async () => {
        try {
          const response = await fetch("http://localhost:8000/get-ai-move", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(gameState), // Send the entire current game state
          });

          if (!response.ok) {
            const errorData = await response
              .json()
              .catch(() => ({ detail: "Unknown server error format" }));
            console.error(
              `Error from AI server: ${response.status} - ${
                errorData.detail || response.statusText
              }`
            );
            // Potentially alert the user or try a fallback if desired
            setIsProcessingMove(false); // Allow game to continue or be fixed by human
            // Example: alert(`AI Error: ${errorData.detail || response.statusText}. AI cannot make a move.`);
            return;
          }

          const aiMove: Position = await response.json();
          console.log(
            `AI (${currentPlayer.name}) received move from Python server:`,
            aiMove
          );

          // Validate move structure (simple check)
          if (
            typeof aiMove.row !== "number" ||
            typeof aiMove.col !== "number"
          ) {
            console.error(
              "Invalid move format received from AI server:",
              aiMove
            );
            setIsProcessingMove(false);
            return;
          }

          handleCellClick(aiMove.row, aiMove.col);
          // handleCellClick will set isProcessingMove to true internally and then false when done with explosions.
          // The setIsProcessingMove(true) at the start of this effect covers the "thinking" and API call period.
        } catch (error) {
          console.error(
            "Network error or other issue fetching AI move:",
            error
          );
          // Potentially alert user that AI server is unavailable
          setIsProcessingMove(false); // Release processing lock
          alert(
            "Failed to connect to the AI server. Please ensure it's running. AI cannot make a move."
          );
        }
      }, thinkingDelay);
    }
  }, [gameState, isProcessingMove]); // Rerun if game state changes or processing flag changes

  const checkWinConditionAndDetermineNextPlayer = useCallback(
    (
      currentGrid: Grid,
      playerWhoMadeTheMove: PlayerId,
      currentRoundNumber: number,
      allPlayers: PlayerInfo[]
    ): Partial<GameState> => {
      if (allPlayers.length === 0) {
        return {
          status: GameStatus.Draw,
          winnerId: null,
          currentPlayerId: null,
        };
      }

      const playerOrbCounts: Map<PlayerId, number> = new Map();
      allPlayers.forEach((p) => playerOrbCounts.set(p.id, 0));
      let totalOrbsOnBoard = 0;

      currentGrid.forEach((row) =>
        row.forEach((cell) => {
          if (cell.player !== null) {
            playerOrbCounts.set(
              cell.player,
              (playerOrbCounts.get(cell.player) || 0) + cell.orbs
            );
          }
          totalOrbsOnBoard += cell.orbs;
        })
      );

      const playersStillWithOrbs = allPlayers.filter(
        (p) => (playerOrbCounts.get(p.id) || 0) > 0
      );
      const playerWhoMadeTheMoveIndex = allPlayers.findIndex(
        (p) => p.id === playerWhoMadeTheMove
      );

      const hasEveryoneHadATurnThisGame =
        currentRoundNumber > 1 ||
        (currentRoundNumber === 1 &&
          playerWhoMadeTheMoveIndex === allPlayers.length - 1);

      if (
        playersStillWithOrbs.length === 1 &&
        totalOrbsOnBoard > 0 &&
        hasEveryoneHadATurnThisGame
      ) {
        return {
          status: GameStatus.Finished,
          winnerId: playersStillWithOrbs[0].id,
          currentPlayerId: null,
        };
      } else if (
        playersStillWithOrbs.length === 0 &&
        totalOrbsOnBoard === 0 &&
        currentRoundNumber >= 1
      ) {
        return {
          status: GameStatus.Draw,
          winnerId: null,
          currentPlayerId: null,
        };
      }

      if (playerWhoMadeTheMoveIndex === -1) {
        console.error(
          "Error: playerWhoMadeTheMove (" +
            playerWhoMadeTheMove +
            ") not found. Defaulting."
        );
        const fallbackPlayerId =
          allPlayers.length > 0 ? allPlayers[0].id : null;
        if (!fallbackPlayerId)
          return {
            status: GameStatus.Draw,
            winnerId: null,
            currentPlayerId: null,
          };
        return { currentPlayerId: fallbackPlayerId, status: GameStatus.Active };
      }

      let nextPlayerIdToSet: PlayerId | null = null;

      if (hasEveryoneHadATurnThisGame) {
        let count = 0;
        let nextPotentialPlayerIndex =
          (playerWhoMadeTheMoveIndex + 1) % allPlayers.length;
        while (count < allPlayers.length) {
          const potentialNextPlayer = allPlayers[nextPotentialPlayerIndex];
          if (
            playersStillWithOrbs.some(
              (activePlayer) => activePlayer.id === potentialNextPlayer.id
            )
          ) {
            nextPlayerIdToSet = potentialNextPlayer.id;
            break;
          }
          nextPotentialPlayerIndex =
            (nextPotentialPlayerIndex + 1) % allPlayers.length;
          count++;
        }

        if (!nextPlayerIdToSet) {
          if (
            playersStillWithOrbs.length === 1 &&
            hasEveryoneHadATurnThisGame
          ) {
            return {
              status: GameStatus.Finished,
              winnerId: playersStillWithOrbs[0].id,
              currentPlayerId: null,
            };
          } else if (
            playersStillWithOrbs.length === 0 &&
            currentRoundNumber >= 1
          ) {
            return {
              status: GameStatus.Draw,
              winnerId: null,
              currentPlayerId: null,
            };
          } else {
            console.warn(
              `checkWinCondition: Could not find a next active player. Active: ${playersStillWithOrbs.map(
                (p) => p.id
              )}, All: ${allPlayers.map((p) => p.id)}`
            );
            if (playersStillWithOrbs.length > 0) {
              nextPlayerIdToSet = playersStillWithOrbs[0].id;
            } else {
              return {
                status: GameStatus.Draw,
                winnerId: null,
                currentPlayerId: null,
              };
            }
          }
        }
      } else {
        nextPlayerIdToSet =
          allPlayers[(playerWhoMadeTheMoveIndex + 1) % allPlayers.length].id;
      }

      if (!nextPlayerIdToSet) {
        console.error(
          "Critical: Next player ID is null when game should be active or explicitly ended."
        );
        if (playersStillWithOrbs.length === 1 && hasEveryoneHadATurnThisGame) {
          return {
            status: GameStatus.Finished,
            winnerId: playersStillWithOrbs[0].id,
            currentPlayerId: null,
          };
        }
        if (playersStillWithOrbs.length === 0 && currentRoundNumber >= 1) {
          return {
            status: GameStatus.Draw,
            winnerId: null,
            currentPlayerId: null,
          };
        }
        return {
          status: GameStatus.Draw,
          winnerId: null,
          currentPlayerId: null,
        };
      }

      return {
        currentPlayerId: nextPlayerIdToSet,
        status: GameStatus.Active,
        winnerId: null,
      };
    },
    []
  );

  const handleCellClick = useCallback(
    async (row: number, col: number) => {
      if (!gameState) return;

      const {
        status,
        currentPlayerId,
        grid: initialGrid,
        gridSize,
        players,
        turnNumber: initialTurnNumberForMove,
      } = gameState;

      const callingPlayerInfo = players.find((p) => p.id === currentPlayerId);
      // If it's an AI move, isProcessingMove might have been set by AI thinking delay.
      // For human moves, if isProcessingMove is true, block.
      if (
        status !== GameStatus.Active ||
        !currentPlayerId ||
        (isProcessingMove && !callingPlayerInfo?.isAI)
      ) {
        if (isProcessingMove && !callingPlayerInfo?.isAI) {
          console.log("Human click ignored: move processing or AI turn.");
        }
        return;
      }

      const currentCell = initialGrid[row][col];
      if (
        currentCell.player !== null &&
        currentCell.player !== currentPlayerId
      ) {
        console.warn("Cannot place orb in a cell owned by another player.");
        // If this was an AI move, this means AI might have chosen an invalid cell based on stale state or flawed logic.
        // We should release the processing lock if AI made a mistake.
        if (callingPlayerInfo?.isAI) {
          setIsProcessingMove(false);
        }
        return;
      }

      setIsProcessingMove(true);

      let currentLogicGrid = deepCopyGrid(initialGrid);
      const actingPlayerId = currentPlayerId;

      currentLogicGrid[row][col] = {
        player: actingPlayerId,
        orbs: currentLogicGrid[row][col].orbs + 1,
      };

      setGameState((prev) =>
        prev ? { ...prev, grid: deepCopyGrid(currentLogicGrid) } : null
      );
      await new Promise((resolve) => setTimeout(resolve, 30));

      let explosionsOccurredInWave: boolean;
      let waveCount = 0;
      const MAX_WAVES = gridSize.rows * gridSize.cols * 4;

      do {
        explosionsOccurredInWave = false;
        waveCount++;
        if (waveCount > MAX_WAVES) {
          console.error(
            "Max explosion waves reached, breaking loop. Declaring draw."
          );
          setGameState((prev) =>
            prev
              ? {
                  ...prev,
                  status: GameStatus.Draw,
                  winnerId: null,
                  currentPlayerId: null,
                }
              : null
          );
          setIsProcessingMove(false);
          return;
        }

        const cellsToExplodeThisWave: Position[] = [];
        for (let r = 0; r < gridSize.rows; r++) {
          for (let c = 0; c < gridSize.cols; c++) {
            const cell = currentLogicGrid[r][c];
            if (
              cell.player !== null &&
              cell.orbs >= getCriticalMass(r, c, gridSize.rows, gridSize.cols)
            ) {
              cellsToExplodeThisWave.push({ row: r, col: c });
            }
          }
        }

        if (cellsToExplodeThisWave.length > 0) {
          explosionsOccurredInWave = true;
          let nextWaveGrid = deepCopyGrid(currentLogicGrid);

          for (const pos of cellsToExplodeThisWave) {
            const explodingCellPlayer = nextWaveGrid[pos.row][pos.col].player;
            const criticalMassOfExplodingCell = getCriticalMass(
              pos.row,
              pos.col,
              gridSize.rows,
              gridSize.cols
            );

            nextWaveGrid[pos.row][pos.col].orbs -= criticalMassOfExplodingCell;
            if (nextWaveGrid[pos.row][pos.col].orbs <= 0) {
              nextWaveGrid[pos.row][pos.col].player = null;
              nextWaveGrid[pos.row][pos.col].orbs = 0;
            }

            const neighbors = getNeighbors(
              pos.row,
              pos.col,
              gridSize.rows,
              gridSize.cols
            );
            for (const neighborPos of neighbors) {
              nextWaveGrid[neighborPos.row][neighborPos.col].orbs += 1;
              nextWaveGrid[neighborPos.row][neighborPos.col].player =
                explodingCellPlayer;
            }
          }
          currentLogicGrid = nextWaveGrid;

          const playerOrbCountsMidExplosion: Map<PlayerId, number> = new Map();
          players.forEach((p) => playerOrbCountsMidExplosion.set(p.id, 0));
          let totalOrbsOnBoardMidExplosion = 0;

          currentLogicGrid.forEach((gridRow) =>
            gridRow.forEach((cell) => {
              if (cell.player !== null) {
                playerOrbCountsMidExplosion.set(
                  cell.player,
                  (playerOrbCountsMidExplosion.get(cell.player) || 0) +
                    cell.orbs
                );
              }
              totalOrbsOnBoardMidExplosion += cell.orbs;
            })
          );

          const playersStillWithOrbsMidExplosion = players.filter(
            (p) => (playerOrbCountsMidExplosion.get(p.id) || 0) > 0
          );

          const playerWhoInitiatedMoveIndex = players.findIndex(
            (p) => p.id === actingPlayerId
          );
          const hasEveryonePlayedThisGameInContextOfMove =
            initialTurnNumberForMove > 1 ||
            (initialTurnNumberForMove === 1 &&
              playerWhoInitiatedMoveIndex === players.length - 1);

          if (
            playersStillWithOrbsMidExplosion.length === 1 &&
            totalOrbsOnBoardMidExplosion > 0 &&
            hasEveryonePlayedThisGameInContextOfMove
          ) {
            setGameState((prev) =>
              prev
                ? {
                    ...prev,
                    grid: deepCopyGrid(currentLogicGrid),
                    status: GameStatus.Finished,
                    winnerId: playersStillWithOrbsMidExplosion[0].id,
                    currentPlayerId: null,
                  }
                : null
            );
            setIsProcessingMove(false);
            return;
          }

          const animationDelay = Math.max(
            50,
            150 -
              waveCount * 10 -
              currentLogicGrid.flat().filter((c) => c.orbs > 0).length * 2
          );
          setGameState((prev) =>
            prev ? { ...prev, grid: deepCopyGrid(currentLogicGrid) } : null
          );
          await new Promise((resolve) => setTimeout(resolve, animationDelay));
        }
      } while (explosionsOccurredInWave);

      const finalGrid = currentLogicGrid;
      const gameUpdate = checkWinConditionAndDetermineNextPlayer(
        finalGrid,
        actingPlayerId,
        initialTurnNumberForMove,
        players
      );

      let newTurnNumberResult = initialTurnNumberForMove;

      if (gameUpdate.status === GameStatus.Active) {
        const actingPlayerIndex = players.findIndex(
          (p) => p.id === actingPlayerId
        );
        if (players.length > 0 && actingPlayerIndex !== -1) {
          if (players.length > 1) {
            if (
              actingPlayerIndex === players.length - 1 &&
              gameUpdate.currentPlayerId === players[0].id
            ) {
              newTurnNumberResult = initialTurnNumberForMove + 1;
            }
          } else if (players.length === 1) {
            newTurnNumberResult = initialTurnNumberForMove + 1;
          }
        }
      }

      setGameState((prev) =>
        prev
          ? {
              ...prev,
              grid: finalGrid,
              status: gameUpdate.status ?? prev.status,
              winnerId:
                gameUpdate.winnerId !== undefined
                  ? gameUpdate.winnerId
                  : prev.winnerId,
              currentPlayerId:
                gameUpdate.currentPlayerId !== undefined
                  ? gameUpdate.currentPlayerId
                  : prev.currentPlayerId,
              turnNumber: newTurnNumberResult,
            }
          : null
      );

      setIsProcessingMove(false);
    },
    [gameState, checkWinConditionAndDetermineNextPlayer]
  );

  const NavButton: React.FC<{
    onClick: () => void;
    text: string;
    isActive: boolean;
  }> = ({ onClick, text, isActive }) => (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-md text-sm font-medium transition-colors
                  ${
                    isActive
                      ? "bg-sky-600 text-white shadow-md"
                      : "text-slate-300 hover:bg-slate-700 hover:text-white"
                  }`}
    >
      {text}
    </button>
  );

  return (
    <div className="min-h-screen flex flex-col items-center p-2 md:p-4 bg-gradient-to-br from-slate-900 to-slate-800 text-slate-100 select-none">
      <nav className="w-full max-w-5xl mx-auto mb-4 md:mb-6 p-3 bg-slate-800/70 backdrop-blur-md rounded-lg shadow-lg">
        <div className="flex justify-center md:justify-between items-center">
          <h1 className="text-2xl font-bold text-sky-400 hidden md:block">
            Chain Reaction
          </h1>
          <div className="flex space-x-2 md:space-x-3">
            <NavButton
              onClick={() => requestNavigation("home")}
              text="Home"
              isActive={currentView === "home"}
            />
            <NavButton
              onClick={() => requestNavigation("rules")}
              text="Rules"
              isActive={currentView === "rules"}
            />
            <NavButton
              onClick={() => requestNavigation("setup")}
              text={
                currentView === "game" &&
                gameState?.status === GameStatus.Active
                  ? "New Setup"
                  : "Play Game"
              }
              isActive={
                currentView === "setup" ||
                (currentView === "game" &&
                  gameState?.status !== GameStatus.Active)
              }
            />
          </div>
        </div>
      </nav>

      <main className="w-full max-w-5xl mx-auto">
        {currentView === "home" && (
          <HomePage
            onPlayGame={() => requestNavigation("setup")}
            onShowRules={() => requestNavigation("rules")}
          />
        )}
        {currentView === "rules" && (
          <RulesPage
            onPlayGame={() => requestNavigation("setup")}
            onShowHome={() => requestNavigation("home")}
          />
        )}
        {currentView === "setup" && (
          <GameSetupPage
            onStartGame={handleStartNewGameWithSetup}
            onCancel={() => requestNavigation("home")}
          />
        )}
        {currentView === "game" && gameState && (
          <div className="w-full flex flex-col md:flex-row gap-4 md:gap-6 items-start">
            <div className="w-full md:w-auto order-2 md:order-1 md:sticky md:top-20">
              <GameInfoPanel
                gameState={gameState}
                onNewGame={() => requestNavigation("setup")}
              />
            </div>
            <div className="w-full order-1 md:order-2 flex-grow">
              <Board gameState={gameState} onCellClick={handleCellClick} />
            </div>
          </div>
        )}
        {currentView === "game" && !gameState && (
          <div className="text-center p-10">
            <h2 className="text-2xl text-slate-400">Game not initialized.</h2>
            <button
              onClick={() => requestNavigation("setup")}
              className="mt-4 px-6 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-md"
            >
              Go to Setup
            </button>
          </div>
        )}
      </main>

      {currentView === "game" && gameState && (
        <footer className="mt-8 text-center text-slate-500 text-sm">
          <p>Chain Reaction Game - React & Tailwind CSS Edition</p>
          {gameState.status === GameStatus.Active &&
            gameState.currentPlayerId && (
              <p>
                Round: {gameState.turnNumber} | Current Player:{" "}
                {gameState.players.find(
                  (p) => p.id === gameState.currentPlayerId
                )?.name || "None"}
              </p>
            )}
          {gameState.status === GameStatus.Finished && gameState.winnerId && (
            <p>
              Game Over! Winner:{" "}
              {gameState.players.find((p) => p.id === gameState.winnerId)
                ?.name || "Unknown"}
            </p>
          )}
          {gameState.status === GameStatus.Draw && (
            <p>Game Over! It's a Draw.</p>
          )}
        </footer>
      )}
      {currentView !== "game" && (
        <footer className="mt-8 text-center text-slate-600 text-xs">
          <p>Chain Reaction Game by AI</p>
        </footer>
      )}
      <ConfirmLeaveModal
        isOpen={showLeaveConfirmationModal}
        onConfirm={confirmAndNavigate}
        onCancel={cancelNavigation}
        message="Are you sure you want to leave the current game? Progress will be lost."
      />
    </div>
  );
};

export default App;
