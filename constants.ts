import { PlayerInfo, PlayerId, ColorOption, GridSizeOption } from "./types";

export const DEFAULT_ROWS = 6;
export const DEFAULT_COLS = 8;

export const MIN_PLAYERS = 2;
export const MAX_PLAYERS = 8;

export const AVAILABLE_PLAYER_COLORS: ColorOption[] = [
  {
    name: "Rose",
    bgClass: "bg-rose-600",
    textClass: "text-rose-400",
    orbClass: "bg-rose-500",
    primaryHex: "#E11D48",
  },
  {
    name: "Sky Blue",
    bgClass: "bg-sky-600",
    textClass: "text-sky-400",
    orbClass: "bg-sky-500",
    primaryHex: "#0284C7",
  },
  {
    name: "Emerald",
    bgClass: "bg-emerald-600",
    textClass: "text-emerald-400",
    orbClass: "bg-emerald-500",
    primaryHex: "#059669",
  },
  {
    name: "Amber",
    bgClass: "bg-amber-500",
    textClass: "text-amber-400",
    orbClass: "bg-amber-500",
    primaryHex: "#F59E0B",
  },
  {
    name: "Violet",
    bgClass: "bg-violet-600",
    textClass: "text-violet-400",
    orbClass: "bg-violet-500",
    primaryHex: "#7C3AED",
  },
  {
    name: "Pink",
    bgClass: "bg-pink-600",
    textClass: "text-pink-400",
    orbClass: "bg-pink-500",
    primaryHex: "#DB2777",
  },
  {
    name: "Teal",
    bgClass: "bg-teal-500",
    textClass: "text-teal-400",
    orbClass: "bg-teal-500",
    primaryHex: "#14B8A6",
  },
  {
    name: "Lime",
    bgClass: "bg-lime-500",
    textClass: "text-lime-400",
    orbClass: "bg-lime-500",
    primaryHex: "#84CC16",
  },
  {
    name: "Fuchsia",
    bgClass: "bg-fuchsia-600",
    textClass: "text-fuchsia-400",
    orbClass: "bg-fuchsia-500",
    primaryHex: "#C026D3",
  },
  {
    name: "Cyan",
    bgClass: "bg-cyan-500",
    textClass: "text-cyan-400",
    orbClass: "bg-cyan-500",
    primaryHex: "#06B6D4",
  },
  {
    name: "Orange",
    bgClass: "bg-orange-500",
    textClass: "text-orange-400",
    orbClass: "bg-orange-500",
    primaryHex: "#F97316",
  },
  {
    name: "Indigo",
    bgClass: "bg-indigo-500",
    textClass: "text-indigo-400",
    orbClass: "bg-indigo-500",
    primaryHex: "#6366F1",
  },
];

export const GRID_SIZE_OPTIONS: GridSizeOption[] = [
  { name: "Standard", rows: 6, cols: 8, label: "Standard (6x8)" },
  { name: "Medium", rows: 8, cols: 10, label: "Medium (8x10)" },
  { name: "Large", rows: 10, cols: 12, label: "Large (10x12)" },
  { name: "Extra Large", rows: 12, cols: 15, label: "XL (12x15)" },
];

// Max orbs for visualization in CellComponent, purely cosmetic
export const MAX_VISUAL_ORBS = 4;
