import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, beforeEach, describe, test, expect } from "vitest";
import AuthScreen from "../components/AuthScreen";
import { useAuthStore } from "../state/authStore";
import { handleApiError } from "../services/api";
import { useToastStore } from "../state/toastStore";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { ProtectedRoute } from "../components/ProtectedRoute";

vi.mock("../services/api", async () => {
  const actual = await vi.importActual<any>("../services/api");
  return {
    ...actual,
    register: vi.fn(),
    login: vi.fn().mockResolvedValue({ access_token: "token-123" }),
    getProfile: vi.fn().mockResolvedValue({ age: 30 }),
  };
});

describe("auth flows and guard", () => {
  beforeEach(() => {
    useAuthStore.getState().logout();
    localStorage.clear();
    useToastStore.getState().clear();
  });

  test("login saves user and calls onAuth", async () => {
    const onAuth = vi.fn();
    render(<AuthScreen onAuth={onAuth} />);
    fireEvent.change(screen.getByPlaceholderText("Email"), { target: { value: "u@example.com" } });
    fireEvent.change(screen.getByPlaceholderText("Password"), { target: { value: "pw" } });
    fireEvent.click(screen.getAllByText("Login")[1]);
    await waitFor(() => expect(onAuth).toHaveBeenCalled());
    expect(useAuthStore.getState().user?.email).toBe("u@example.com");
  });

  test("protected route redirects when unauthenticated", async () => {
    render(
      <MemoryRouter initialEntries={["/history"]}>
        <Routes>
          <Route path="/" element={<div>Home</div>} />
          <Route
            path="/history"
            element={
              <ProtectedRoute>
                <div>History</div>
              </ProtectedRoute>
            }
          />
        </Routes>
      </MemoryRouter>
    );
    expect(screen.getByText("Home")).toBeInTheDocument();
  });

  test("protected route renders content when authenticated", async () => {
    useAuthStore.getState().setUser({ email: "u@example.com", token: "t", id: 0 });
    render(
      <MemoryRouter initialEntries={["/history"]}>
        <Routes>
          <Route path="/" element={<div>Home</div>} />
          <Route
            path="/history"
            element={
              <ProtectedRoute>
                <div>History</div>
              </ProtectedRoute>
            }
          />
        </Routes>
      </MemoryRouter>
    );
    expect(screen.getByText("History")).toBeInTheDocument();
  });

  test("handleApiError surfaces toast", () => {
    handleApiError({ response: { status: 500 } });
    const toasts = useToastStore.getState().toasts;
    expect(toasts.find((t) => t.message.includes("Server error"))).toBeTruthy();
  });
});
