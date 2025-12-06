import { useCallback } from "react";
import { useAuthStore } from "../state/authStore";
import type { User } from "../types";

/**
 * Centralized auth hook backed by Zustand for persistence and refresh flags.
 */
export function useAuth() {
  const { user, setUser, logout, refreshing, startRefresh, finishRefresh } = useAuthStore();

  const setUserSafe = useCallback(
    (next: User | null) => {
      setUser(next);
    },
    [setUser]
  );

  return {
    user,
    setUser: setUserSafe,
    logout,
    refreshing,
    startRefresh,
    finishRefresh,
  };
}
