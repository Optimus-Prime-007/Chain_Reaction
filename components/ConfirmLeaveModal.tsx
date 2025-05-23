import React from "react";

interface ConfirmLeaveModalProps {
  isOpen: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  message: string;
}

const ConfirmLeaveModal: React.FC<ConfirmLeaveModalProps> = ({
  isOpen,
  onConfirm,
  onCancel,
  message,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-800 p-6 md:p-8 rounded-xl shadow-2xl max-w-md w-full text-slate-100">
        <h3 className="text-xl font-semibold mb-6 text-sky-400 text-center">
          {message}
        </h3>
        <div className="flex justify-around space-x-4">
          <button
            onClick={onCancel}
            className="px-6 py-3 w-1/2 text-md font-semibold text-slate-200 bg-slate-600 hover:bg-slate-500 rounded-lg shadow transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-6 py-3 w-1/2 text-md font-semibold text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg shadow transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-400"
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmLeaveModal;
