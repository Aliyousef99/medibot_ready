// frontend/src/components/UserMenu.tsx
import { useMemo, useRef, useState, useEffect } from "react";
import { ChevronUp, ChevronDown, History } from "lucide-react";

export type UiUser = {
  email: string;
  profile?: { name?: string };
};

export default function UserMenu({
  user,
  onProfile,
  onLogout,
}: {
  user: UiUser;
  onProfile: () => void;
  onLogout: () => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (!ref.current) return;
      if (!ref.current.contains(e.target as Node)) setOpen(false);
    }
    if (open) document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const displayName = useMemo(() => {
    const n = user?.profile?.name?.trim();
    if (n) return n;
    const email = user?.email || "";
    return email.includes("@") ? email.split("@")[0] : email || "Account";
  }, [user]);

  const initials = useMemo(() => {
    const parts = displayName.split(/\s+/).filter(Boolean);
    if (!parts.length) return "U";
    const first = parts[0][0] ?? "";
    const second = parts.length > 1 ? parts[1][0] ?? "" : "";
    return (first + second).toUpperCase();
  }, [displayName]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-3 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 px-3 py-2 shadow-sm hover:bg-zinc-100 dark:hover:bg-zinc-900 transition"
      >
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-500 text-white text-sm font-semibold">
          {initials}
        </div>
        <div className="flex-1 text-left">
          <div className="text-sm font-medium leading-5">{displayName}</div>
        </div>
        {open ? (
          <ChevronDown className="w-4 h-4 text-zinc-400" />
        ) : (
          <ChevronUp className="w-4 h-4 text-zinc-400" />
        )}
      </button>

      {open && (
        <div className="absolute bottom-14 left-0 right-0 z-50 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 shadow-lg p-1">
          <a
            href="#/history"
            onClick={() => setOpen(false)}
            className="w-full text-left px-3 py-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-900 text-sm flex items-center gap-2"
          >
            <History size={16} />
            History
          </a>
          <button
            onClick={() => {
              setOpen(false);
              onProfile();
            }}
            className="w-full text-left px-3 py-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-900 text-sm"
          >
            Profile
          </button>
          <button
            onClick={() => {
              setOpen(false);
              onLogout();
            }}
            className="w-full text-left px-3 py-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-900 text-sm text-red-600"
          >
            Logout
          </button>
        </div>
      )}
    </div>
  );
}