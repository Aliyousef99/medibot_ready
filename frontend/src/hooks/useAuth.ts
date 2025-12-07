import { useCallback } from "react";
import { useAuthStore } from "../state/authStore";
import type { User } from "../types";
import { chatScopeForUser, useChatStore } from "../state/chatStore";

/**
 * Centralized auth hook backed by Zustand for persistence and refresh flags.
 */
export function useAuth() {
  const { user, setUser: baseSetUser, logout: baseLogout, refreshing, startRefresh, finishRefresh } = useAuthStore();

  const resetChatForUser = useCallback((u: User | null) => {
    try {
      const scope = chatScopeForUser(u);
      useChatStore.getState().resetChat(scope);
    } catch {
      // best-effort reset; ignore if store unavailable
    }
  }, []);

  const setUserSafe = useCallback(
    (next: User | null) => {
      resetChatForUser(next);
      baseSetUser(next);
    },
    [baseSetUser, resetChatForUser]
  );

  const logoutSafe = useCallback(() => {
    resetChatForUser(null);
    baseLogout();
  }, [baseLogout, resetChatForUser]);

  return {
    user,
    setUser: setUserSafe,
    logout: logoutSafe,
    refreshing,
    startRefresh,
    finishRefresh,
  };
}
