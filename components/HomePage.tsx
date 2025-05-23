import React from "react";

interface HomePageProps {
  onPlayGame: () => void; // Will call `requestNavigation('setup')` in App.tsx
  onShowRules: () => void; // Will call `requestNavigation('rules')` in App.tsx
}

const HomePage: React.FC<HomePageProps> = ({ onPlayGame, onShowRules }) => {
  return (
    <div className="flex flex-col items-center justify-center text-center p-6 md:p-12 bg-slate-800 rounded-xl shadow-2xl min-h-[calc(100vh-200px)] md:min-h-0">
      <header className="mb-10">
        <h1 className="text-5xl md:text-7xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-emerald-500 mb-4">
          Chain Reaction
        </h1>
        <p className="text-lg md:text-xl text-slate-300 max-w-2xl mx-auto">
          A thrilling turn-based strategy game of wits and explosive chain
          reactions. Outmaneuver your opponent to conquer the grid!
        </p>
      </header>

      <div className="space-y-5 md:space-y-0 md:space-x-6 flex flex-col md:flex-row">
        <button
          onClick={onPlayGame}
          className="px-10 py-4 text-lg font-semibold text-white bg-emerald-600 rounded-lg shadow-lg hover:bg-emerald-700 transition-transform transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-emerald-400 focus:ring-opacity-75"
          aria-label="Start a new game setup"
        >
          Play Game
        </button>
        <button
          onClick={onShowRules}
          className="px-10 py-4 text-lg font-semibold text-slate-200 bg-sky-700 rounded-lg shadow-lg hover:bg-sky-800 transition-transform transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-sky-500 focus:ring-opacity-75"
          aria-label="Learn how to play the game"
        >
          How to Play
        </button>
      </div>

      <section className="mt-12 pt-8 border-t border-slate-700 w-full max-w-3xl">
        <h2 className="text-2xl font-semibold text-slate-200 mb-4">Features</h2>
        <ul className="list-disc list-inside text-slate-400 space-y-2 text-left mx-auto max-w-md">
          <li>Engaging turn-based strategy</li>
          <li>Customizable games: 2-8 players, multiple grid sizes</li>
          <li>Player color selection</li>
          <li>Explosive chain reactions</li>
          <li>Sleek and responsive UI</li>
          <li>Interactive rules demonstration</li>
        </ul>
      </section>
    </div>
  );
};

export default HomePage;
