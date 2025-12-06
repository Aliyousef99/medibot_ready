import { create } from "zustand";

type ToastType = "info" | "success" | "error";

export type Toast = {
  id: string;
  type: ToastType;
  message: string;
};

type ToastState = {
  toasts: Toast[];
  add: (toast: Omit<Toast, "id"> | string) => string;
  remove: (id: string) => void;
  clear: () => void;
};

export const useToastStore = create<ToastState>((set, get) => ({
  toasts: [],
  add: (toast) => {
    const id = `t_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const payload =
      typeof toast === "string"
        ? { id, type: "info" as ToastType, message: toast }
        : { id, ...toast };
    set((s) => ({ toasts: [...s.toasts, payload] }));
    return id;
  },
  remove: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  clear: () => set({ toasts: [] }),
}));
