import { describe, test, expect, vi, beforeEach } from "vitest";
import { __testRefreshAccessToken, apiClient } from "../services/api";
import { useAuthStore } from "../state/authStore";

describe("auth refresh flow", () => {
  beforeEach(() => {
    useAuthStore.getState().logout();
    localStorage.clear();
    vi.restoreAllMocks();
  });

  test("refreshAccessToken stores new access token", async () => {
    localStorage.setItem("auth", JSON.stringify({ email: "u@example.com", refresh_token: "r1" }));
    useAuthStore.getState().setUser({ email: "u@example.com", refresh_token: "r1", token: "old", id: 1 });

    const postSpy = vi.spyOn(apiClient, "post").mockResolvedValue({ data: { access_token: "new-token" } } as any);

    const token = await __testRefreshAccessToken();
    expect(token).toBe("new-token");
    expect(useAuthStore.getState().user?.token).toBe("new-token");
    expect(postSpy).toHaveBeenCalledWith("/api/auth/refresh", { refresh_token: "r1" });
  });
});
