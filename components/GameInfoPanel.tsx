import React from "react";
import { GameState, PlayerInfo, GameStatus } from "../types";

interface GameInfoPanelProps {
  gameState: GameState;
  onNewGame: () => void;
}

const GameInfoPanel: React.FC<GameInfoPanelProps> = ({
  gameState,
  onNewGame,
}) => {
  const { status, currentPlayerId, winnerId, players, turnNumber, grid } =
    gameState;

  const currentPlayer = players.find((p) => p.id === currentPlayerId);
  const winner = players.find((p) => p.id === winnerId);

  let statusMessage = "";
  let statusColor = "text-slate-300";

  if (status === GameStatus.Active && currentPlayer) {
    statusMessage = `${currentPlayer.name}'s Turn`;
    statusColor = currentPlayer.textColorClass;
  } else if (status === GameStatus.Finished && winner) {
    statusMessage = `${winner.name} Wins!`;
    statusColor = winner.textColorClass;
  } else if (status === GameStatus.Draw) {
    statusMessage = "It's a Draw!";
    statusColor = "text-yellow-400";
  } else if (status === GameStatus.Setup) {
    statusMessage = "Game Setup In Progress";
  }

  const playerOrbCounts = React.useMemo(() => {
    const counts = new Map<PlayerInfo["id"], number>();
    players.forEach((p) => counts.set(p.id, 0));
    grid.flat().forEach((cell) => {
      if (cell.player !== null) {
        counts.set(cell.player, (counts.get(cell.player) || 0) + cell.orbs);
      }
    });
    return counts;
  }, [grid, players]);

  const activeDisplayPlayers: PlayerInfo[] = [];
  const eliminatedDisplayPlayers: PlayerInfo[] = [];

  players.forEach((pInfo) => {
    const orbsOnBoard = playerOrbCounts.get(pInfo.id) || 0;
    // Eliminated for display if: game active, 0 orbs, AND past the first round.
    const isEliminatedForDisplay =
      status === GameStatus.Active && orbsOnBoard === 0 && turnNumber > 1;

    if (isEliminatedForDisplay) {
      eliminatedDisplayPlayers.push(pInfo);
    } else {
      activeDisplayPlayers.push(pInfo);
    }
  });

  const PlayerListItem: React.FC<{
    pInfo: PlayerInfo;
    orbs: number;
    isActive: boolean;
    isEliminated?: boolean;
  }> = ({ pInfo, orbs, isActive, isEliminated }) => {
    let itemClasses = `flex justify-between items-center p-2 rounded transition-all duration-200 ${pInfo.colorClass} `;
    const itemStyles: React.CSSProperties = {};

    if (isEliminated) {
      itemClasses += "filter grayscale opacity-60";
    } else if (isActive) {
      itemClasses +=
        "ring-2 ring-offset-2 ring-offset-slate-800 ring-[var(--player-ring-color)] shadow-lg scale-105 brightness-110";
      itemStyles["--player-ring-color" as any] = pInfo.primaryHex;
    } else {
      itemClasses += "opacity-85 hover:opacity-100";
    }

    return (
      <li style={itemStyles} className={itemClasses}>
        <span
          className={`font-medium text-white truncate ${
            isEliminated ? "line-through" : ""
          }`}
          title={pInfo.name}
        >
          {pInfo.name}
        </span>
        <span
          className={`text-sm text-white bg-black/30 px-2 py-0.5 rounded whitespace-nowrap ${
            isEliminated ? "opacity-70" : ""
          }`}
        >
          Orbs: {orbs}
        </span>
      </li>
    );
  };

  return (
    <div className="p-4 md:p-6 bg-slate-800 rounded-lg shadow-xl space-y-4 md:min-w-[300px] max-h-[calc(100vh-120px)] flex flex-col">
      <h2 className="text-2xl font-bold text-center text-slate-100">
        Chain Reaction
      </h2>

      <div className="text-center">
        <p
          className={`text-xl font-semibold ${statusColor} transition-colors duration-300 min-h-[28px]`}
        >
          {statusMessage}
        </p>
        {status === GameStatus.Active && (
          <p className="text-sm text-slate-400">Round: {turnNumber}</p>
        )}
      </div>

      {(status === GameStatus.Finished || status === GameStatus.Draw) && (
        <div className="mt-4 text-center">
          <img
            src={`https://picsum.photos/seed/${
              winner?.name || "draw_game"
            }-${Date.now()}/200/150`}
            alt="Celebration"
            className="mx-auto rounded-md shadow-md mb-2"
            onError={(e) => (e.currentTarget.style.display = "none")} // Hide if image fails to load
          />
        </div>
      )}

      <div className="pt-2 flex-grow overflow-y-auto space-y-3">
        {activeDisplayPlayers.length > 0 && (
          <div>
            <h3 className="text-lg font-semibold mb-2 text-slate-200 sticky top-0 bg-slate-800 py-1 z-10">
              In Game:
            </h3>
            <ul className="space-y-1 pr-1">
              {activeDisplayPlayers.map((pInfo) => (
                <PlayerListItem
                  key={`active-${pInfo.id}`}
                  pInfo={pInfo}
                  orbs={playerOrbCounts.get(pInfo.id) || 0}
                  isActive={
                    pInfo.id === currentPlayerId && status === GameStatus.Active
                  }
                />
              ))}
            </ul>
          </div>
        )}

        {eliminatedDisplayPlayers.length > 0 && (
          <div className="pt-2">
            <h3 className="text-md font-semibold mb-2 text-slate-400 sticky top-0 bg-slate-800 py-1 z-10">
              Eliminated:
            </h3>
            <ul className="space-y-1 pr-1">
              {eliminatedDisplayPlayers.map((pInfo) => (
                <PlayerListItem
                  key={`eliminated-${pInfo.id}`}
                  pInfo={pInfo}
                  orbs={0} // They are eliminated, so 0 orbs
                  isActive={false}
                  isEliminated={true}
                />
              ))}
            </ul>
          </div>
        )}
      </div>

      <button
        onClick={onNewGame}
        className="w-full mt-auto px-4 py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-md shadow-md transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:ring-opacity-75"
      >
        New Game Setup
      </button>
    </div>
  );
};

export default GameInfoPanel;
