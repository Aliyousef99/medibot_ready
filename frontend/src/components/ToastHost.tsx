import React, { useEffect } from "react";
import { X } from "lucide-react";
import { useToastStore } from "../state/toastStore";

const typeStyles: Record<string, string> = {
  info: "bg-zinc-900 text-white",
  success: "bg-emerald-600 text-white",
  error: "bg-red-600 text-white",
};

export default function ToastHost() {
  const { toasts, remove } = useToastStore();

  useEffect(() => {
    const timers = toasts.map((t) =>
      setTimeout(() => remove(t.id), 4000)
    );
    return () => timers.forEach(clearTimeout);
  }, [toasts, remove]);

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`flex items-start gap-3 rounded-lg px-4 py-3 shadow-lg ${typeStyles[t.type] || typeStyles.info}`}
        >
          <div className="flex-1 text-sm">{t.message}</div>
          <button
            aria-label="Close notification"
            className="p-1 hover:opacity-80"
            onClick={() => remove(t.id)}
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  );
}
