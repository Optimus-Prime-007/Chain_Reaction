export type PlayerId = number; // 1 for Player 1, 2 for Player 2, etc.

export interface CellState {
  player: PlayerId | null; // null for empty, or PlayerId
  orbs: number;
}

export type Grid = CellState[][];

export interface Position {
  row: number;
  col: number;
}

export interface ColorOption {
  name: string;
  bgClass: string;
  textClass: string;
  orbClass: string;
  primaryHex?: string; // For color picker if needed
}

export interface PlayerInfo {
  id: PlayerId;
  name: string;
  colorClass: string; // Tailwind background class e.g., 'bg-rose-600'
  textColorClass: string; // Tailwind text class e.g., 'text-rose-500'
  orbColorClass: string; // Tailwind class for individual orbs if visualized
  primaryHex: string; // Hex color string for custom elements like cursors
  isAI?: boolean; // Added to identify AI players
}

export interface Move {
  playerId: PlayerId;
  position: Position;
}

export enum GameStatus {
  Setup = "setup",
  Active = "active",
  Finished = "finished",
  Draw = "draw",
}

export interface GridSizeConfig {
  rows: number;
  cols: number;
}

export interface GameState {
  grid: Grid;
  players: PlayerInfo[];
  currentPlayerId: PlayerId | null;
  status: GameStatus;
  winnerId: PlayerId | null;
  gridSize: GridSizeConfig;
  turnNumber: number; // Represents the current round number
}

export interface PlayerSetupInfo {
  id: PlayerId;
  name: string;
  colorName: string; // Name of the chosen ColorOption
  isAI?: boolean; // Added to identify AI players during setup
}

export interface GameSetupOptions {
  numberOfPlayers: number;
  gridSizeName: string; // Name of the chosen GridSizeOption
  playerSettings: PlayerSetupInfo[];
  playWithAI?: boolean; // To indicate if AI mode is selected
}

export interface GridSizeOption {
  name: string;
  rows: number;
  cols: number;
  label: string;
}
