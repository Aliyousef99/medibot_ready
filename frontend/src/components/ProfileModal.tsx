import React, { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { updateProfile as apiUpdateProfile, deleteUserData, setConsent } from "../services/api";
import { useToastStore } from "../state/toastStore";
import type { User } from "../types";
import { useAuth } from "../hooks/useAuth";
import { chatScopeForUser, useChatStore } from "../state/chatStore";

type ProfileModalProps = {
  open: boolean;
  onClose: () => void;
  user: User;
  onUpdate: (profile: any) => void;
};

export default function ProfileModal({ open, onClose, user, onUpdate }: ProfileModalProps) {
  const [age, setAge] = useState<number | "">("");
  const [sex, setSex] = useState("");
  const [conditions, setConditions] = useState("");
  const [medications, setMedications] = useState("");
  const [consent, setConsentState] = useState<boolean>(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const { add: addToast } = useToastStore();
  const { logout, setUser } = useAuth();

  useEffect(() => {
    if (!open) return;
    const p = user.profile || {};
    setAge(p.age ?? "");
    setSex(p.sex ?? "");
    setConditions(Array.isArray(p.conditions) ? p.conditions.join(", ") : "");
    setMedications(Array.isArray(p.medications) ? p.medications.join(", ") : "");
    setConsentState(!!p.consent_given);
    setMsg(null);

    // focus first input when opening
    const firstInput = dialogRef.current?.querySelector("input, select, textarea") as HTMLElement | null;
    firstInput?.focus();
  }, [open, user.profile]);

   // Escape to close and basic focus trap
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
      if (e.key === "Tab") {
        const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusable || focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  async function save() {
    // basic client-side validation
    if (age !== "" && (Number.isNaN(Number(age)) || Number(age) <= 0 || Number(age) > 120)) {
      addToast({ type: "error", message: "Please enter a valid age between 1 and 120." });
      return;
    }
    const normConditions = conditions
      ? conditions.split(",").map((s) => s.trim()).filter(Boolean)
      : [];
    const normMedications = medications
      ? medications.split(",").map((s) => s.trim()).filter(Boolean)
      : [];

    setSaving(true);
    setMsg(null);
    try {
      const updated = await apiUpdateProfile({
        age: age === "" ? null : Number(age),
        sex: sex || null,
        conditions: normConditions,
        medications: normMedications,
        consent_given: consent,
      });
      await setConsent(consent);
      onUpdate(updated);
      setMsg("Profile saved successfully!");
      setTimeout(() => onClose(), 1000);
    } catch (e: any) {
      const message = e?.message || "Unknown error";
      setMsg(`Save failed: ${message}`);
      addToast({ type: "error", message: `Profile save failed: ${message}` });
    } finally {
      setSaving(false);
    }
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Profile settings"
      onMouseDown={(e) => {
        if (e.currentTarget === e.target) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className="w-full max-w-xl rounded-3xl bg-gradient-to-br from-emerald-50 via-white to-white dark:from-zinc-900/90 dark:via-zinc-950 dark:to-zinc-950 border border-emerald-100 dark:border-zinc-800 shadow-2xl p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-emerald-600 dark:text-emerald-300">Profile</p>
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50">Profile Settings</h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">Tune these fields so recommendations stay personalized.</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-full px-2 py-1 hover:bg-zinc-100 dark:hover:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-400"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <label className="text-sm text-zinc-600 dark:text-zinc-300 space-y-1">
              <span className="block font-medium">Age</span>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value ? parseInt(e.target.value, 10) : "")}
                placeholder="e.g., 42"
                className="w-full rounded-lg border px-3 py-2 bg-white dark:bg-zinc-950 focus:outline-none focus:ring-2 focus:ring-emerald-400"
                min={1}
                max={120}
              />
              <span className="text-[12px] text-zinc-400">Enter an age between 1 and 120.</span>
            </label>
            <label className="text-sm text-zinc-600 dark:text-zinc-300 space-y-1">
              <span className="block font-medium">Sex</span>
              <select
                value={sex}
                onChange={(e) => setSex(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 bg-white dark:bg-zinc-950 focus:outline-none focus:ring-2 focus:ring-emerald-400"
              >
                <option value="">Select</option>
                <option value="male">male</option>
                <option value="female">female</option>
                <option value="other">other</option>
              </select>
            </label>
          </div>

          <label className="text-sm text-zinc-600 dark:text-zinc-300 space-y-1">
            <span className="block font-medium">Conditions</span>
            <input
              value={conditions}
              onChange={(e) => setConditions(e.target.value)}
              placeholder="e.g., diabetes, hypertension"
              className="w-full rounded-lg border px-3 py-2 bg-white dark:bg-zinc-950 focus:outline-none focus:ring-2 focus:ring-emerald-400"
            />
            <span className="text-[12px] text-zinc-400">Comma-separated list.</span>
          </label>

          <label className="text-sm text-zinc-600 dark:text-zinc-300 space-y-1">
            <span className="block font-medium">Medications</span>
            <input
              value={medications}
              onChange={(e) => setMedications(e.target.value)}
              placeholder="e.g., metformin, lisinopril"
              className="w-full rounded-lg border px-3 py-2 bg-white dark:bg-zinc-950 focus:outline-none focus:ring-2 focus:ring-emerald-400"
            />
            <span className="text-[12px] text-zinc-400">Comma-separated list.</span>
          </label>

          <label className="flex items-start gap-3 rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-white dark:bg-zinc-900">
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsentState(e.target.checked)}
              className="mt-1"
            />
            <div className="text-sm text-zinc-700 dark:text-zinc-200">
              <div className="font-medium">Consent to data processing</div>
              <div className="text-[12px] text-zinc-500 dark:text-zinc-400">
                I agree to store and process my health data to provide recommendations. You can withdraw consent anytime and delete your data.
              </div>
            </div>
          </label>

          {msg && (
            <div className={`text-sm ${msg.startsWith("Save failed") ? "text-red-500" : "text-green-500"}`}>
              {msg}
            </div>
          )}

          <div className="pt-2 flex justify-between gap-2 items-center">
            <button
              type="button"
              onClick={async () => {
                if (deleting) return;
                const confirmed = window.confirm("Delete all of your stored data (labs, chats, recommendations, profile)? This cannot be undone.");
                if (!confirmed) return;
                try {
                  setDeleting(true);
                  await deleteUserData();
                  addToast({ type: "info", message: "Your data was deleted." });
                  // Keep user logged in but clear local profile/chat state
                  try {
                    const scope = chatScopeForUser(user);
                    useChatStore.getState().resetChat(scope);
                  } catch {
                    // best-effort reset
                  }
                  setUser({ ...user, profile: {} });
                  onClose();
                } catch (e: any) {
                  addToast({ type: "error", message: e?.message || "Failed to delete data." });
                } finally {
                  setDeleting(false);
                }
              }}
              className="text-xs px-3 py-2 rounded-xl border border-red-200 text-red-600 hover:bg-red-50 dark:border-red-800 dark:hover:bg-red-900/30 disabled:opacity-60"
              disabled={deleting}
            >
              {deleting ? "Deleting..." : "Delete all my data"}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl px-3 py-2 border hover:bg-zinc-100 dark:hover:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-400"
            >
              Cancel
            </button>
            <button
              onClick={save}
              disabled={saving}
              className="rounded-xl px-4 py-2 bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-60 flex items-center gap-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-400"
            >
              {saving && <Loader2 className="w-4 h-4 animate-spin" />}
              {saving ? "Saving..." : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
