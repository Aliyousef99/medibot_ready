import { create } from "zustand";
import type { User } from "../types";

type AuthState = {
  user: User | null;
  refreshing: boolean;
};

type AuthActions = {
  setUser: (u: User | null) => void;
  startRefresh: () => void;
  finishRefresh: () => void;
  logout: () => void;
};

export type AuthStore = AuthState & AuthActions;

function loadUser(): User | null {
  try {
    const saved = localStorage.getItem("auth");
    if (saved) return JSON.parse(saved);
  } catch {
    // ignore malformed localStorage
  }
  return null;
}

export const useAuthStore = create<AuthStore>((set) => {
  const persist = (u: User | null) => {
    try {
      if (u) localStorage.setItem("auth", JSON.stringify(u));
      else localStorage.removeItem("auth");
    } catch {
      // best-effort persistence
    }
  };

  return {
    user: loadUser(),
    refreshing: false,
    setUser: (u) => set(() => (persist(u), { user: u })),
    startRefresh: () => set({ refreshing: true }),
    finishRefresh: () => set({ refreshing: false }),
    logout: () => set(() => (persist(null), { user: null, refreshing: false })),
  };
});
