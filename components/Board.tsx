import React from "react";
// FIX: Import GameStatus
import {
  Grid as GridType,
  GameState,
  PlayerId,
  PlayerInfo,
  GameStatus,
} from "../types";
import CellComponent from "./Cell";

interface BoardProps {
  gameState: GameState;
  onCellClick: (row: number, col: number) => void;
}

const Board: React.FC<BoardProps> = ({ gameState, onCellClick }) => {
  const { grid, gridSize, currentPlayerId, status, players } = gameState;

  const playerColorMap = React.useMemo(() => {
    const map: Record<PlayerId, { bgClass: string; orbClass: string }> = {};
    players.forEach((p) => {
      map[p.id] = { bgClass: p.colorClass, orbClass: p.orbColorClass };
    });
    return map;
  }, [players]);

  const currentPlayerInfo = players.find((p) => p.id === currentPlayerId);
  const currentPlayerPrimaryHex = currentPlayerInfo?.primaryHex;

  return (
    <div
      className="grid gap-0.5 bg-slate-800 p-2 rounded-lg shadow-xl"
      style={{
        gridTemplateColumns: `repeat(${gridSize.cols}, minmax(0, 1fr))`,
      }}
      aria-label="Game Board"
    >
      {grid.map((row, rowIndex) =>
        row.map((cell, colIndex) => {
          const isCurrentPlayerOwner = cell.player === currentPlayerId;
          const isCellEmpty = cell.player === null;
          const isSelectable =
            status === GameStatus.Active &&
            (isCellEmpty || isCurrentPlayerOwner);

          return (
            <CellComponent
              key={`${rowIndex}-${colIndex}`}
              cell={cell}
              rowIndex={rowIndex}
              colIndex={colIndex}
              gridRows={gridSize.rows}
              gridCols={gridSize.cols}
              onClick={() => onCellClick(rowIndex, colIndex)}
              isCurrentPlayerCell={isCurrentPlayerOwner}
              isSelectable={isSelectable}
              playerColors={playerColorMap}
              currentPlayerCursorHex={
                isSelectable ? currentPlayerPrimaryHex : undefined
              }
              currentPlayerHighlightColorHex={
                isSelectable ? currentPlayerPrimaryHex : undefined
              }
            />
          );
        })
      )}
    </div>
  );
};

export default Board;
