import React, { useState } from "react";
import { Loader2, LogIn } from "lucide-react";
import { register as apiRegister, login as apiLogin, getProfile as apiGetProfile } from "../services/api";
import type { User } from "../types";
import { useAuth } from "../hooks/useAuth";

type AuthScreenProps = {
  onAuth: (user: User) => void;
};

export default function AuthScreen({ onAuth }: AuthScreenProps) {
  const { setUser } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const apiBase =
    (typeof window !== "undefined" && (window as any).__CHAT_API_BASE__) ||
    (import.meta as any)?.env?.VITE_API_BASE ||
    "http://localhost:8000";
  const apiBaseTrimmed = apiBase.replace(/\/+$/, "");

  async function submit() {
    setLoading(true);
    setError("");
    try {
      if (mode === "register") {
        await apiRegister(email, password);
      }
      const loginData = await apiLogin(email, password);
      const token = loginData.access_token;

      const userScaffold = { id: 0, email, token };
      // Set auth immediately so subsequent calls carry the bearer token
      setUser(userScaffold);

      const profile = await apiGetProfile().catch(() => null);

      const user = { ...userScaffold, profile };

      setUser(user);
      onAuth(user);
    } catch (e: any) {
      const detail = e?.response?.data?.detail;
      const status = e?.response?.status;
      if (status === 409 || detail === "User with this email already exists") {
        setError("An account with this email already exists. Try logging in instead.");
      } else if (status === 422 || status === 400) {
        setError("Please enter a valid email and a password of at least 6 characters.");
      } else {
        setError(detail || e?.message || "Unable to sign in right now. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-zinc-100 dark:bg-zinc-900">
      <div className="w-full max-w-sm rounded-xl bg-white dark:bg-zinc-950 p-6 shadow space-y-3">
        <h2 className="text-lg font-semibold mb-4">{mode === "login" ? "Login" : "Register"}</h2>
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          className="mb-2 w-full rounded-md border px-3 py-2"
          disabled={loading}
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          className="mb-4 w-full rounded-md border px-3 py-2"
          disabled={loading}
        />
        {error && (
          <div className="text-red-500 text-sm mb-2 bg-red-100 dark:bg-red-900/20 p-2 rounded-md">
            {error}
          </div>
        )}
        <button
          onClick={submit}
          disabled={loading}
          className="w-full rounded-md bg-emerald-600 text-white py-2 flex items-center justify-center gap-2 disabled:opacity-60"
        >
          {loading && <Loader2 className="w-4 h-4 animate-spin" />}
          {loading ? "Please wait..." : mode === "login" ? "Login" : "Register"}
        </button>
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <span className="flex-1 border-t border-zinc-200 dark:border-zinc-800" />
          <span>or</span>
          <span className="flex-1 border-t border-zinc-200 dark:border-zinc-800" />
        </div>
        <button
          type="button"
          className="w-full inline-flex items-center justify-center gap-2 rounded-md border border-zinc-200 dark:border-zinc-800 py-2 hover:bg-zinc-50 dark:hover:bg-zinc-900 disabled:opacity-60"
          onClick={() => {
            const redirect = encodeURIComponent(window.location.origin + window.location.hash);
            window.location.href = `${apiBaseTrimmed}/api/auth/google/start?redirect=${redirect}`;
          }}
          disabled={loading}
          aria-label="Sign in with Google"
        >
          <LogIn className="w-4 h-4" /> Sign in with Google
        </button>
        <button
          onClick={() => setMode(mode === "login" ? "register" : "login")}
          className="mt-3 text-sm text-blue-500 underline disabled:opacity-60"
          disabled={loading}
        >
          {mode === "login" ? "Create an account" : "Have an account? Login"}
        </button>
      </div>
    </div>
  );
}
