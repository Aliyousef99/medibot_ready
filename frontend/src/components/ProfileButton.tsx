import { useState } from "react";
import ProfileModal from "./ProfileModal";

export function ProfileButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="w-full rounded-xl bg-neutral-800 hover:bg-neutral-700 text-white px-3 py-2 text-sm font-medium transition"
      >
        Profile
      </button>

      <ProfileModal open={open} onClose={() => setOpen(false)}>
        {/* optional: embed your ProfileSettings form here */}
      </ProfileModal>
    </>
  );
}
