import React, { useState, useEffect } from "react";
import {
  GameSetupOptions,
  PlayerSetupInfo,
  ColorOption,
  GridSizeOption,
  PlayerId,
} from "../types";
import {
  AVAILABLE_PLAYER_COLORS,
  GRID_SIZE_OPTIONS,
  MIN_PLAYERS,
  MAX_PLAYERS,
} from "../constants";

interface GameSetupPageProps {
  onStartGame: (options: GameSetupOptions) => void;
  onCancel: () => void; // To go back to home or previous view
}

const GameSetupPage: React.FC<GameSetupPageProps> = ({
  onStartGame,
  onCancel,
}) => {
  const [numberOfPlayers, setNumberOfPlayers] = useState<number>(MIN_PLAYERS);
  const [gridSizeName, setGridSizeName] = useState<string>(
    GRID_SIZE_OPTIONS[0].name
  );
  const [playerSettings, setPlayerSettings] = useState<PlayerSetupInfo[]>([]);
  const [availableColors, setAvailableColors] = useState<ColorOption[]>(
    AVAILABLE_PLAYER_COLORS
  );
  const [playWithAI, setPlayWithAI] = useState<boolean>(false);

  useEffect(() => {
    const initialSettings: PlayerSetupInfo[] = [];
    const currentAvailableColorsList = [...AVAILABLE_PLAYER_COLORS]; // Full list to pick from
    const usedColorNamesThisUpdate: string[] = [];

    for (let i = 0; i < numberOfPlayers; i++) {
      const playerId = (i + 1) as PlayerId;
      const isAIPlayer =
        playWithAI && i === numberOfPlayers - 1 && numberOfPlayers > 0; // Last player is AI

      // Find a color that hasn't been used in this specific update cycle yet
      let colorToAssign = currentAvailableColorsList.find(
        (c) => !usedColorNamesThisUpdate.includes(c.name)
      );

      if (!colorToAssign) {
        // Fallback if we run out of unique colors from the preferred list (e.g. >12 players conceptually)
        colorToAssign =
          AVAILABLE_PLAYER_COLORS[i % AVAILABLE_PLAYER_COLORS.length];
      }
      if (!colorToAssign) {
        // Absolute fallback, should not happen with current constants
        colorToAssign = {
          name: "Default",
          bgClass: "bg-gray-500",
          textClass: "text-gray-700",
          orbClass: "bg-gray-400",
        };
      }

      usedColorNamesThisUpdate.push(colorToAssign.name);

      initialSettings.push({
        id: playerId,
        name: isAIPlayer ? "AI Bot" : `Player ${playerId}`,
        colorName: colorToAssign.name,
        isAI: isAIPlayer,
      });
    }
    setPlayerSettings(initialSettings);
    // Update the list of colors truly available for dropdowns (excluding those ALREADY selected)
    setAvailableColors(
      AVAILABLE_PLAYER_COLORS.filter(
        (c) => !initialSettings.some((s) => s.colorName === c.name)
      )
    );
  }, [numberOfPlayers, playWithAI]);

  const handlePlayerNameChange = (index: number, name: string) => {
    const newSettings = [...playerSettings];
    if (!newSettings[index].isAI) {
      // AI name should not be changed by user
      newSettings[index].name = name;
      setPlayerSettings(newSettings);
    }
  };

  const handlePlayerColorChange = (
    playerIndex: number,
    newColorName: string
  ) => {
    const newSettings = [...playerSettings];
    // oldColorName is not directly used here for logic, but good to know
    // const oldColorName = newSettings[playerIndex].colorName;

    newSettings[playerIndex].colorName = newColorName;
    setPlayerSettings(newSettings);

    // Update available colors based on all current selections
    const currentlySelectedColorNames = newSettings.map((p) => p.colorName);
    setAvailableColors(
      AVAILABLE_PLAYER_COLORS.filter(
        (c) => !currentlySelectedColorNames.includes(c.name)
      )
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStartGame({
      numberOfPlayers,
      gridSizeName,
      playerSettings,
      playWithAI, // Pass this along
    });
  };

  return (
    <div className="p-4 md:p-8 bg-slate-800 rounded-xl shadow-2xl text-slate-100 max-w-3xl mx-auto">
      <h2 className="text-3xl font-bold text-center text-emerald-400 mb-8">
        Game Setup
      </h2>
      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Number of Players & AI Toggle */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          <div>
            <label
              htmlFor="numberOfPlayers"
              className="block text-lg font-medium text-sky-400 mb-2"
            >
              Number of Players
            </label>
            <select
              id="numberOfPlayers"
              value={numberOfPlayers}
              onChange={(e) => setNumberOfPlayers(parseInt(e.target.value, 10))}
              className="w-full p-3 bg-slate-700 border border-slate-600 rounded-md shadow-sm focus:ring-emerald-500 focus:border-emerald-500"
            >
              {Array.from(
                { length: MAX_PLAYERS - MIN_PLAYERS + 1 },
                (_, i) => MIN_PLAYERS + i
              ).map((num) => (
                <option key={num} value={num}>
                  {num} Players
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-end pb-1">
            <div className="flex items-center h-full">
              <input
                id="playWithAI"
                type="checkbox"
                checked={playWithAI}
                onChange={(e) => setPlayWithAI(e.target.checked)}
                className="h-5 w-5 text-emerald-500 border-slate-500 rounded focus:ring-emerald-400 bg-slate-700"
                disabled={numberOfPlayers < 1} // Can't have AI if no players
              />
              <label
                htmlFor="playWithAI"
                className="ml-2 text-lg font-medium text-sky-400"
              >
                Play with AI?
              </label>
            </div>
          </div>
        </div>

        {/* Grid Size */}
        <div>
          <label
            htmlFor="gridSize"
            className="block text-lg font-medium text-sky-400 mb-2"
          >
            Grid Size
          </label>
          <select
            id="gridSize"
            value={gridSizeName}
            onChange={(e) => setGridSizeName(e.target.value)}
            className="w-full p-3 bg-slate-700 border border-slate-600 rounded-md shadow-sm focus:ring-emerald-500 focus:border-emerald-500"
          >
            {GRID_SIZE_OPTIONS.map((opt) => (
              <option key={opt.name} value={opt.name}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Player Settings */}
        <div className="space-y-6">
          {playerSettings.map((player, index) => (
            <div
              key={player.id}
              className={`p-4 border rounded-lg ${
                player.isAI
                  ? "border-teal-500 bg-teal-800/30"
                  : "border-slate-700 bg-slate-700/30"
              }`}
            >
              <h4
                className={`text-md font-semibold mb-3 ${
                  player.isAI ? "text-teal-400" : "text-emerald-400"
                }`}
              >
                Player {player.id} {player.isAI ? "(AI Bot)" : ""}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label
                    htmlFor={`playerName-${player.id}`}
                    className="block text-sm font-medium text-slate-300"
                  >
                    Name
                  </label>
                  <input
                    type="text"
                    id={`playerName-${player.id}`}
                    value={player.name}
                    onChange={(e) =>
                      handlePlayerNameChange(index, e.target.value)
                    }
                    className={`mt-1 w-full p-2 bg-slate-600 border border-slate-500 rounded-md shadow-sm focus:ring-emerald-500 focus:border-emerald-500 ${
                      player.isAI ? "text-slate-400 cursor-not-allowed" : ""
                    }`}
                    disabled={player.isAI}
                  />
                </div>
                <div>
                  <label
                    htmlFor={`playerColor-${player.id}`}
                    className="block text-sm font-medium text-slate-300"
                  >
                    Color
                  </label>
                  <select
                    id={`playerColor-${player.id}`}
                    value={player.colorName}
                    onChange={(e) =>
                      handlePlayerColorChange(index, e.target.value)
                    }
                    className="mt-1 w-full p-2 bg-slate-600 border border-slate-500 rounded-md shadow-sm focus:ring-emerald-500 focus:border-emerald-500"
                    // Consider disabling color for AI if you want a fixed AI color
                  >
                    {/* Current player's selected color */}
                    <option value={player.colorName}>
                      {AVAILABLE_PLAYER_COLORS.find(
                        (c) => c.name === player.colorName
                      )?.name || "Select Color"}
                    </option>
                    {/* Other available colors */}
                    {availableColors.map((colorOpt) => (
                      <option key={colorOpt.name} value={colorOpt.name}>
                        {colorOpt.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row justify-end space-y-3 sm:space-y-0 sm:space-x-4 pt-4">
          <button
            type="button"
            onClick={onCancel}
            className="px-6 py-3 text-md font-semibold text-slate-200 bg-slate-600 hover:bg-slate-500 rounded-lg shadow transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="px-8 py-3 text-md font-semibold text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg shadow transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-400"
            disabled={numberOfPlayers < 1} // Prevent starting game if somehow 0 players are selected
          >
            Start Game
          </button>
        </div>
      </form>
    </div>
  );
};

export default GameSetupPage;
