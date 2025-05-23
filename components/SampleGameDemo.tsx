import React, { useState, useCallback, useMemo, useEffect } from "react";
import { Grid, CellState, PlayerId, Position, PlayerInfo } from "../types";
import CellComponent from "./Cell";
import {
  getCriticalMass as getActualCriticalMass,
  getNeighbors,
} from "../utils/gameLogic";
import { AVAILABLE_PLAYER_COLORS } from "../constants"; // Use available player colors for consistency

const DEMO_ROWS = 3;
const DEMO_COLS = 3;
const DEMO_PLAYER_1_ID: PlayerId = 1;
const DEMO_PLAYER_2_ID: PlayerId = 2;

// Create a fixed set of PlayerInfo for the demo using the first few available colors
const demoPlayerInfos: PlayerInfo[] = [
  {
    id: DEMO_PLAYER_1_ID,
    name: "Player 1 (Demo)",
    colorClass: AVAILABLE_PLAYER_COLORS[0].bgClass,
    textColorClass: AVAILABLE_PLAYER_COLORS[0].textClass,
    orbColorClass: AVAILABLE_PLAYER_COLORS[0].orbClass,
  },
  {
    id: DEMO_PLAYER_2_ID,
    name: "Player 2 (Demo)",
    colorClass: AVAILABLE_PLAYER_COLORS[1].bgClass,
    textColorClass: AVAILABLE_PLAYER_COLORS[1].textClass,
    orbColorClass: AVAILABLE_PLAYER_COLORS[1].orbClass,
  },
];

// Create a player color map for CellComponent, similar to Board.tsx
const demoPlayerColorMap = demoPlayerInfos.reduce((acc, player) => {
  acc[player.id] = {
    bgClass: player.colorClass,
    orbClass: player.orbColorClass,
  };
  return acc;
}, {} as Record<PlayerId, { bgClass: string; orbClass: string }>);

interface DemoStep {
  grid: Grid;
  message: string;
  action?: (currentGrid: Grid) => Grid;
  highlightCells?: Position[];
}

const createDemoGrid = (): Grid =>
  Array(DEMO_ROWS)
    .fill(null)
    .map(() =>
      Array(DEMO_COLS)
        .fill(null)
        .map(() => ({ player: null, orbs: 0 }))
    );

const SampleGameDemo: React.FC = () => {
  const initialDemoGrid = useMemo(() => createDemoGrid(), []);

  const demoSteps: DemoStep[] = useMemo(
    () => [
      {
        grid: initialDemoGrid,
        message:
          "Welcome! This is a 3x3 grid. Each cell shows its critical mass (top-right). Let's start with Player 1 (Rose).",
      },
      {
        grid: (() => {
          const g = createDemoGrid();
          g[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };
          return g;
        })(),
        action: (currentGrid: Grid) => {
          const newGrid = currentGrid.map((r) => r.map((c) => ({ ...c })));
          newGrid[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };
          return newGrid;
        },
        message:
          "Player 1 places an orb in the center cell (1,1). It now has 1 orb. Critical mass is 4.",
        highlightCells: [{ row: 1, col: 1 }],
      },
      {
        grid: (() => {
          const g = createDemoGrid();
          g[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };
          // g[0][0] is set by action in next step
          return g;
        })(),
        action: (currentGrid: Grid) => {
          const newGrid = currentGrid.map((r) => r.map((c) => ({ ...c })));
          // Ensure Player 1's orb from previous step is present if this action builds on it
          if (newGrid[1][1].player !== DEMO_PLAYER_1_ID)
            newGrid[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };

          newGrid[0][0] = { player: DEMO_PLAYER_2_ID, orbs: 1 };
          return newGrid;
        },
        message:
          "Player 2 (Sky Blue) places an orb in a corner cell (0,0). Critical mass for corners is 2.",
        highlightCells: [{ row: 0, col: 0 }],
      },
      {
        grid: (() => {
          const g = createDemoGrid();
          g[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };
          g[0][0] = { player: DEMO_PLAYER_2_ID, orbs: 1 };
          return g;
        })(),
        action: (currentGrid: Grid) => {
          const newGrid = currentGrid.map((r) => r.map((c) => ({ ...c })));
          // Ensure previous state
          if (newGrid[1][1].player !== DEMO_PLAYER_1_ID)
            newGrid[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };
          if (newGrid[0][0].player !== DEMO_PLAYER_2_ID)
            newGrid[0][0] = { player: DEMO_PLAYER_2_ID, orbs: 1 };

          newGrid[0][0].orbs += 1;
          return newGrid;
        },
        message:
          "Player 2 adds another orb to (0,0). It now has 2 orbs and reaches critical mass!",
        highlightCells: [{ row: 0, col: 0 }],
      },
      {
        grid: (() => {
          const g = createDemoGrid();
          g[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };
          g[0][0] = { player: DEMO_PLAYER_2_ID, orbs: 2 }; // State before explosion
          return g;
        })(),
        action: (currentGrid: Grid) => {
          const newGrid = currentGrid.map((r) => r.map((c) => ({ ...c })));
          // Ensure previous state for explosion base
          if (newGrid[1][1].player !== DEMO_PLAYER_1_ID)
            newGrid[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };
          if (
            newGrid[0][0].player !== DEMO_PLAYER_2_ID ||
            newGrid[0][0].orbs !== 2
          ) {
            newGrid[0][0] = { player: DEMO_PLAYER_2_ID, orbs: 2 };
          }

          const explodingPos = { row: 0, col: 0 };
          const explodingPlayer =
            newGrid[explodingPos.row][explodingPos.col].player;

          newGrid[explodingPos.row][explodingPos.col].orbs -=
            getActualCriticalMass(
              explodingPos.row,
              explodingPos.col,
              DEMO_ROWS,
              DEMO_COLS
            );
          if (newGrid[explodingPos.row][explodingPos.col].orbs <= 0) {
            newGrid[explodingPos.row][explodingPos.col].player = null;
            newGrid[explodingPos.row][explodingPos.col].orbs = 0;
          }

          const neighbors = getNeighbors(
            explodingPos.row,
            explodingPos.col,
            DEMO_ROWS,
            DEMO_COLS
          );
          neighbors.forEach((n) => {
            newGrid[n.row][n.col].orbs += 1;
            newGrid[n.row][n.col].player = explodingPlayer;
          });
          return newGrid;
        },
        message:
          "Cell (0,0) explodes! Orbs spread to (0,1) and (1,0), capturing them for Player 2.",
        highlightCells: [
          { row: 0, col: 0 },
          { row: 0, col: 1 },
          { row: 1, col: 0 },
        ],
      },
      {
        grid: (() => {
          const g = createDemoGrid();
          g[1][1] = { player: DEMO_PLAYER_1_ID, orbs: 1 };
          g[0][0] = { player: null, orbs: 0 };
          g[0][1] = { player: DEMO_PLAYER_2_ID, orbs: 1 };
          g[1][0] = { player: DEMO_PLAYER_2_ID, orbs: 1 };
          return g;
        })(),
        message:
          "The board after Player 2's explosion. Player 1's turn again. This concludes the basic demo!",
      },
    ],
    [initialDemoGrid]
  );

  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [currentDemoGrid, setCurrentDemoGrid] = useState<Grid>(
    () => demoSteps[0].grid
  );

  const updateGridForStep = useCallback(
    (stepIndex: number) => {
      const step = demoSteps[stepIndex];
      if (step.action) {
        // To build cumulative state for actions:
        // Find the grid result from the previous step's action, or the step's defined grid if no action.
        let baseGridForAction = step.grid; // Default to step's defined grid
        if (stepIndex > 0) {
          const prevStep = demoSteps[stepIndex - 1];
          // Simulate running previous actions to get the true previous state
          let tempGrid = initialDemoGrid;
          for (let i = 0; i < stepIndex; i++) {
            if (demoSteps[i].action) {
              tempGrid = demoSteps[i].action!(tempGrid);
            } else {
              tempGrid = demoSteps[i].grid;
            }
          }
          baseGridForAction = tempGrid;
        }
        setCurrentDemoGrid(step.action(baseGridForAction));
      } else {
        setCurrentDemoGrid(step.grid);
      }
    },
    [demoSteps, initialDemoGrid]
  );

  useEffect(() => {
    updateGridForStep(currentStepIndex);
  }, [currentStepIndex, updateGridForStep]);

  const handleNextStep = () => {
    if (currentStepIndex < demoSteps.length - 1) {
      setCurrentStepIndex((prev) => prev + 1);
    }
  };

  const handlePrevStep = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex((prev) => prev - 1); // Corrected to decrement
    }
  };

  const handleResetDemo = () => {
    setCurrentStepIndex(0);
    // Also reset grid to initial step's grid explicitly if updateGridForStep isn't sufficient
    setCurrentDemoGrid(demoSteps[0].grid);
  };

  const currentDemoStep = demoSteps[currentStepIndex];

  return (
    <div className="p-3 bg-slate-900/70 rounded-lg shadow-md">
      <div
        className="grid gap-0.5 bg-slate-800 p-1 rounded shadow-inner mx-auto mb-3"
        style={{
          gridTemplateColumns: `repeat(${DEMO_COLS}, minmax(0, 1fr))`,
          width: `${DEMO_COLS * 60}px`,
        }}
      >
        {currentDemoGrid.map((row, rowIndex) =>
          row.map((cell, colIndex) => {
            const isHighlighted = currentDemoStep.highlightCells?.some(
              (p) => p.row === rowIndex && p.col === colIndex
            );
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`${
                  isHighlighted ? "ring-2 ring-emerald-400 ring-inset" : ""
                }`}
              >
                <CellComponent
                  cell={cell}
                  rowIndex={rowIndex}
                  colIndex={colIndex}
                  gridRows={DEMO_ROWS}
                  gridCols={DEMO_COLS}
                  onClick={() => {}}
                  isCurrentPlayerCell={false}
                  isSelectable={false}
                  playerColors={demoPlayerColorMap} // Pass demo player colors
                />
              </div>
            );
          })
        )}
      </div>
      <p className="text-sm text-slate-300 mb-3 min-h-[40px] text-center">
        {currentDemoStep.message}
      </p>
      <div className="flex justify-center space-x-2">
        <button
          onClick={handlePrevStep}
          disabled={currentStepIndex === 0}
          className="px-4 py-2 text-xs bg-sky-700 hover:bg-sky-600 disabled:bg-slate-600 text-white rounded"
        >
          Previous
        </button>
        <button
          onClick={handleResetDemo}
          className="px-4 py-2 text-xs bg-slate-600 hover:bg-slate-500 text-white rounded"
        >
          Reset
        </button>
        <button
          onClick={handleNextStep}
          disabled={currentStepIndex === demoSteps.length - 1}
          className="px-4 py-2 text-xs bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-600 text-white rounded"
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default SampleGameDemo;
