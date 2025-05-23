import { Grid, Position, PlayerId, CellState } from "../types";

export const getCriticalMass = (
  row: number,
  col: number,
  numRows: number,
  numCols: number
): number => {
  const isCorner =
    (row === 0 && col === 0) ||
    (row === 0 && col === numCols - 1) ||
    (row === numRows - 1 && col === 0) ||
    (row === numRows - 1 && col === numCols - 1);
  const isEdge =
    row === 0 || row === numRows - 1 || col === 0 || col === numCols - 1;

  if (isCorner) return 2;
  if (isEdge) return 3;
  return 4;
};

export const getNeighbors = (
  row: number,
  col: number,
  numRows: number,
  numCols: number
): Position[] => {
  const neighbors: Position[] = [];
  const deltas = [
    { dr: -1, dc: 0 }, // up
    { dr: 1, dc: 0 }, // down
    { dr: 0, dc: -1 }, // left
    { dr: 0, dc: 1 }, // right
  ];

  for (const delta of deltas) {
    const nr = row + delta.dr;
    const nc = col + delta.dc;
    if (nr >= 0 && nr < numRows && nc >= 0 && nc < numCols) {
      neighbors.push({ row: nr, col: nc });
    }
  }
  return neighbors;
};

export const createInitialGrid = (rows: number, cols: number): Grid => {
  return Array(rows)
    .fill(null)
    .map(() =>
      Array(cols)
        .fill(null)
        .map(() => ({ player: null, orbs: 0 }))
    );
};

// Deep copies a grid
export const deepCopyGrid = (grid: Grid): Grid => {
  return grid.map((row) => row.map((cell) => ({ ...cell })));
};
