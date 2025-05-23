import React from "react";
import SampleGameDemo from "./SampleGameDemo";

interface RulesPageProps {
  onPlayGame: () => void; // Will call `requestNavigation('setup')` in App.tsx
  onShowHome: () => void; // Will call `requestNavigation('home')` in App.tsx
}

const RuleSection: React.FC<{ title: string; children: React.ReactNode }> = ({
  title,
  children,
}) => (
  <section className="mb-6 p-4 bg-slate-700/50 rounded-lg shadow">
    <h3 className="text-xl font-semibold text-sky-400 mb-2">{title}</h3>
    <div className="text-slate-300 space-y-1">{children}</div>
  </section>
);

const RulesPage: React.FC<RulesPageProps> = ({ onPlayGame, onShowHome }) => {
  return (
    <div className="p-4 md:p-6 bg-slate-800 rounded-xl shadow-2xl text-slate-100 max-w-4xl mx-auto">
      <header className="mb-6 text-center">
        <h2 className="text-4xl font-bold text-emerald-400">
          How to Play Chain Reaction
        </h2>
      </header>

      <RuleSection title="Game Objective">
        <p>
          The goal is to eliminate all opponent orbs from the grid. The last
          player with orbs remaining on the board wins!
        </p>
      </RuleSection>

      <RuleSection title="Gameplay">
        <p>
          Players take turns placing a single orb of their color into a cell on
          the grid.
        </p>
        <p>
          You can place an orb in an empty cell or a cell already occupied by
          your own orbs.
        </p>
        <p>You cannot place an orb in a cell occupied by an opponent's orbs.</p>
      </RuleSection>

      <RuleSection title="Critical Mass">
        <p>Each cell has a "critical mass" limit:</p>
        <ul className="list-disc list-inside ml-4">
          <li>
            <strong>Corner Cells:</strong> Critical mass of 2 orbs.
          </li>
          <li>
            <strong>Edge Cells (not corners):</strong> Critical mass of 3 orbs.
          </li>
          <li>
            <strong>Center Cells (not edges or corners):</strong> Critical mass
            of 4 orbs.
          </li>
        </ul>
        <p>
          The critical mass for each cell is typically displayed as a small
          number on the cell.
        </p>
      </RuleSection>

      <RuleSection title="Explosions & Chain Reactions">
        <p>
          When a cell reaches its critical mass by adding an orb, it explodes!
        </p>
        <p>During an explosion:</p>
        <ul className="list-disc list-inside ml-4">
          <li>
            The exploding cell loses orbs equal to its critical mass (e.g., a
            corner cell loses 2 orbs). If its orb count drops to 0, it becomes
            empty.
          </li>
          <li>
            One orb is sent to each adjacent (up, down, left, right) neighboring
            cell.
          </li>
          <li>
            Any neighboring cell that receives an orb from an explosion is
            captured by the player who triggered the explosion. Its orb count
            increases by one.
          </li>
        </ul>
        <p>
          If an explosion causes a neighboring cell to reach its critical mass,
          that cell also explodes, potentially leading to a chain reaction
          across the grid.
        </p>
      </RuleSection>

      <RuleSection title="Winning & Drawing">
        <p>
          <strong>Winning:</strong> You win if, after all chain reactions from
          your move resolve, you are the only player with orbs left on the grid.
          This is only checked after all players have had a chance to make at
          least one move in the game.
        </p>
        <p>
          <strong>Drawing:</strong> A game is a draw if all orbs are eliminated
          from the board simultaneously, or if a rare loop/stalemate condition
          is reached.
        </p>
      </RuleSection>

      <RuleSection title="Live Demo">
        <p className="mb-3">
          Interact with the demo below to see these rules in action on a small
          grid. Click "Next Step" to proceed through the demonstration.
        </p>
        <SampleGameDemo />
      </RuleSection>

      <div className="mt-8 flex flex-col sm:flex-row justify-center space-y-3 sm:space-y-0 sm:space-x-4">
        <button
          onClick={onPlayGame}
          className="px-8 py-3 text-md font-semibold text-white bg-emerald-600 rounded-lg shadow hover:bg-emerald-700 transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-400"
          aria-label="Setup a new game"
        >
          Play Game Now
        </button>
        <button
          onClick={onShowHome}
          className="px-8 py-3 text-md font-semibold text-slate-200 bg-sky-700 rounded-lg shadow hover:bg-sky-800 transition-colors focus:outline-none focus:ring-2 focus:ring-sky-500"
          aria-label="Back to Home page"
        >
          Back to Home
        </button>
      </div>
    </div>
  );
};

export default RulesPage;
