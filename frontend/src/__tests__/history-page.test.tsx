import { render, screen, fireEvent } from "@testing-library/react";
import { vi, beforeEach, describe, test, expect } from "vitest";
import HistoryPage from "../pages/History";
import * as Api from "../services/api";

vi.mock("../services/api", () => ({
  getHistory: vi.fn(),
  getHistoryDetail: vi.fn(),
  deleteHistory: vi.fn(),
  getRecommendationsForLab: vi.fn(),
}));

const api = Api as unknown as {
  getHistory: ReturnType<typeof vi.fn>;
  getHistoryDetail: ReturnType<typeof vi.fn>;
  deleteHistory: ReturnType<typeof vi.fn>;
  getRecommendationsForLab: ReturnType<typeof vi.fn>;
};

describe("History page states", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("shows empty guidance when no labs", async () => {
    api.getHistory.mockResolvedValueOnce([]);
    render(<HistoryPage />);
    expect(await screen.findByText(/Lab Report History/i)).toBeInTheDocument();
    expect(await screen.findByText(/You have no saved lab reports yet/i)).toBeInTheDocument();
    expect(screen.getByText(/Complete your profile/i)).toBeInTheDocument();
  });

  test("loads details and recommendations", async () => {
    api.getHistory.mockResolvedValueOnce([
      { id: "l1", summary: "sum", structured_json: {}, raw_text: "rt", created_at: new Date().toISOString(), updated_at: new Date().toISOString() },
    ]);
    api.getHistoryDetail.mockResolvedValue({
      id: "l1",
      summary: "sum",
      structured_json: {},
      raw_text: "rt",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });
    api.getRecommendationsForLab.mockResolvedValue({
      risk_tier: "low",
      actions: ["Keep monitoring"],
    });

    render(<HistoryPage />);
    const viewBtn = await screen.findByRole("button", { name: /View details/i });
    fireEvent.click(viewBtn);
    expect(await screen.findByText(/Lab Report Details/i)).toBeInTheDocument();
    const recBtn = await screen.findByRole("button", { name: /Get Recommendations/i });
    fireEvent.click(recBtn);
    expect(await screen.findByText(/Keep monitoring/i)).toBeInTheDocument();
  });
});
