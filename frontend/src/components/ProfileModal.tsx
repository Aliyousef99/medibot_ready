import React from "react";

type ProfileModalProps = {
  open: boolean;
  onClose: () => void;
  children?: React.ReactNode;
  // add these only if you need them
  // user?: User;
  // onUpdate?: (profile: any) => void;
};

const ProfileModal: React.FC<ProfileModalProps> = ({ open, onClose, children }) => {
  if (!open) return null; // don’t render when closed

  return (
    <div className="fixed inset-0 z-50">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Modal card */}
      <div
        className="absolute inset-0 flex items-center justify-center p-4"
        role="dialog"
        aria-modal="true"
      >
        <div className="w-full max-w-md rounded-2xl bg-white dark:bg-zinc-950 shadow-xl border border-zinc-200 dark:border-zinc-800">
          <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
            <h3 className="text-sm font-semibold">Profile</h3>
            <button
              onClick={onClose}
              className="rounded-lg px-2 py-1 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-800"
            >
              Close
            </button>
          </div>

          <div className="p-4">
            {/* put your ProfileSettings form here if desired */}
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfileModal;
