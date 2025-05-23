import React from "react";
import { CellState, PlayerId } from "../types";
import { getCriticalMass } from "../utils/gameLogic";

interface CellProps {
  cell: CellState;
  rowIndex: number;
  colIndex: number;
  gridRows: number;
  gridCols: number;
  onClick: () => void;
  isCurrentPlayerCell: boolean;
  isSelectable: boolean;
  playerColors?: Record<PlayerId, { bgClass: string; orbClass: string }>;
  currentPlayerCursorHex?: string;
  currentPlayerHighlightColorHex?: string;
}

const CellComponent: React.FC<CellProps> = ({
  cell,
  rowIndex,
  colIndex,
  gridRows,
  gridCols,
  onClick,
  isCurrentPlayerCell,
  isSelectable,
  playerColors,
  currentPlayerCursorHex,
  currentPlayerHighlightColorHex,
}) => {
  const criticalMass = getCriticalMass(rowIndex, colIndex, gridRows, gridCols);

  const baseClasses =
    "aspect-square flex flex-col items-center justify-center border border-slate-600 transition-all duration-150 ease-in-out";

  const ownedByOtherPlayerAndNotEmpty =
    cell.player !== null && !isCurrentPlayerCell && cell.orbs > 0;

  let interactionClasses = "";
  const dynamicStyles: React.CSSProperties = {};

  if (!ownedByOtherPlayerAndNotEmpty && isSelectable) {
    interactionClasses = "transform hover:scale-105 hover:brightness-110 z-10";
    if (currentPlayerCursorHex) {
      const playerHexColor = currentPlayerCursorHex;
      const strokeColor =
        playerHexColor.toLowerCase() === "#ffffff" ||
        playerHexColor.toLowerCase() === "#fffffe"
          ? "black"
          : "white";
      const cursorSvg = `<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9" fill="${playerHexColor}" stroke="${strokeColor}" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="${strokeColor}"/></svg>`;
      const encodedSvg = encodeURIComponent(cursorSvg);
      dynamicStyles.cursor = `url('data:image/svg+xml;charset=UTF-8,${encodedSvg}') 12 12, auto`;
    } else {
      interactionClasses += " cursor-pointer"; // Fallback if hex not provided
    }

    if (currentPlayerHighlightColorHex) {
      dynamicStyles["--player-highlight-color" as any] =
        currentPlayerHighlightColorHex;
      interactionClasses +=
        " hover:shadow-[inset_0_0_0_3px_var(--player-highlight-color)]";
    }
  } else if (ownedByOtherPlayerAndNotEmpty) {
    interactionClasses = "cursor-not-allowed opacity-70";
  } else {
    interactionClasses = "cursor-default";
  }

  const resolvedCellPlayerColor = cell.player
    ? playerColors && playerColors[cell.player]
      ? playerColors[cell.player].bgClass
      : "bg-gray-500"
    : "bg-slate-700";

  const renderOrbs = () => {
    if (cell.orbs === 0) return null;

    return (
      <span
        className={`text-2xl font-bold ${
          cell.player ? "text-white" : "text-slate-400"
        }`}
      >
        {cell.orbs}
      </span>
    );
  };

  const canBeClicked = isSelectable && !ownedByOtherPlayerAndNotEmpty;

  return (
    <div
      className={`${baseClasses} ${resolvedCellPlayerColor} ${interactionClasses} relative`}
      style={dynamicStyles}
      onClick={canBeClicked ? onClick : undefined}
      role="button"
      aria-label={`Cell ${rowIndex}-${colIndex}, Orbs: ${cell.orbs}, Player: ${
        cell.player || "Empty"
      }, Critical Mass: ${criticalMass}`}
      tabIndex={canBeClicked ? 0 : -1}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && canBeClicked) {
          onClick();
        }
      }}
    >
      {renderOrbs()}
      <div className="absolute top-0 right-1 text-xs text-slate-400 opacity-70">
        {criticalMass}
      </div>
    </div>
  );
};

export default CellComponent;
