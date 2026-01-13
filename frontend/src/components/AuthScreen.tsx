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
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [loading, setLoading] = useState(false);

  const allowedDomains = [
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "icloud.com",
    "aol.com",
    "protonmail.com",
  ];

  const apiBase =
    (typeof window !== "undefined" && (window as any).__CHAT_API_BASE__) ||
    (import.meta as any)?.env?.VITE_API_BASE ||
    "http://localhost:8000";
  const apiBaseTrimmed = apiBase.replace(/\/+$/, "");

  async function submit() {
    setLoading(true);
    setError("");
    setInfo("");
    try {
      if (mode === "register") {
        const domain = (email.split("@")[1] || "").toLowerCase();
        if (!allowedDomains.includes(domain)) {
          setError("Please use a supported email domain (gmail.com, yahoo.com, outlook.com, hotmail.com, live.com, icloud.com, aol.com, protonmail.com).");
          return;
        }
        if (password.length < 6 || password.length > 24) {
          setError("Password must be between 6 and 24 characters.");
          return;
        }
        if (password !== confirmPassword) {
          setError("Passwords do not match.");
          return;
        }
        const reg = await apiRegister(email, password);
        if (reg?.status === "created") {
          setInfo("Account created. You can log in now.");
          setMode("login");
          setPassword("");
          setConfirmPassword("");
          return;
        }
      }
      const loginData = await apiLogin(email, password);
      const token = loginData.access_token;
      const refresh_token = (loginData as any)?.refresh_token;

      // Defer assigning id until we fetch profile (to avoid shared chat scope across users)
      const userScaffold = { id: undefined as any, email, token, refresh_token };
      // Set auth immediately so subsequent calls carry the bearer token
      setUser(userScaffold);

      const profile = await apiGetProfile().catch(() => null);
      const resolvedId = (profile && (profile as any).user_id) || email;

      const user = { ...userScaffold, id: resolvedId, profile };

      setUser(user);
      onAuth(user);
    } catch (e: any) {
      const detail = e?.response?.data?.detail;
      const status = e?.response?.status;
      if (status === 409 || detail === "User with this email already exists") {
        setError("An account with this email already exists. Try logging in instead.");
      } else if (status === 422 || status === 400) {
        setError("Please enter a valid email and a password between 6 and 24 characters.");
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
        <div className="relative mb-2">
          <input
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="w-full rounded-md border px-3 py-2 pr-16"
            maxLength={24}
            disabled={loading}
          />
          <button
            type="button"
            onClick={() => setShowPassword((v) => !v)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-zinc-500"
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? "Hide" : "Show"}
          </button>
        </div>
        {mode === "register" && (
          <div className="relative mb-2">
            <input
              type={showConfirmPassword ? "text" : "password"}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Confirm password"
              className="w-full rounded-md border px-3 py-2 pr-16"
              maxLength={24}
              disabled={loading}
            />
            <button
              type="button"
              onClick={() => setShowConfirmPassword((v) => !v)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-zinc-500"
              aria-label={showConfirmPassword ? "Hide password" : "Show password"}
            >
              {showConfirmPassword ? "Hide" : "Show"}
            </button>
          </div>
        )}
        {error && (
          <div className="text-red-500 text-sm mb-2 bg-red-100 dark:bg-red-900/20 p-2 rounded-md">
            {error}
          </div>
        )}
        {info && (
          <div className="text-emerald-600 text-sm mb-2 bg-emerald-100 dark:bg-emerald-900/20 p-2 rounded-md">
            {info}
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
