import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, beforeEach, test, expect } from "vitest";
import { useAuthStore } from "../state/authStore";
import ClassicChatbotUI from "../components/ClassicChatbotUI";

vi.mock("../services/api", () => {
  return {
    register: vi.fn(),
    login: vi.fn().mockResolvedValue({ access_token: "token-123" }),
    getProfile: vi.fn().mockResolvedValue({}),
    updateProfile: vi.fn().mockResolvedValue({}),
    postChatMessage: vi.fn().mockResolvedValue({
      ai_explanation: "Assistant reply",
      summary: "",
      pipeline: null,
      request_id: "req-1",
      symptom_analysis: null,
      local_recommendations: null,
      disclaimer: "",
    }),
    extractText: vi.fn().mockResolvedValue({ text: "mock extracted text" }),
  };
});

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem("auth", JSON.stringify({ email: "u@example.com", token: "t" }));
  useAuthStore.getState().setUser({ email: "u@example.com", token: "t", id: 0 });
});

test("shows auth when no stored user", async () => {
  localStorage.clear();
  useAuthStore.setState({ user: null, refreshing: false });
  render(<ClassicChatbotUI />);
  const headings = await screen.findAllByText(/Login/i);
  expect(headings.length).toBeGreaterThan(0);
});

test("sends message and shows assistant reply", async () => {
  render(<ClassicChatbotUI />);
  const textarea = await screen.findByPlaceholderText(/Paste lab text or type your question/i);
  fireEvent.change(textarea, { target: { value: "What is my LDL?" } });
  const sendBtn = await screen.findByRole("button", { name: /Send/i });
  fireEvent.click(sendBtn);
  expect(await screen.findByText("Assistant reply")).toBeInTheDocument();
});

test("file upload extracts text into composer", async () => {
  render(<ClassicChatbotUI />);
  const uploadLabel = await screen.findByLabelText("Upload file");
  const uploadInput = uploadLabel.querySelector("input") as HTMLInputElement;
  const file = new File(["foo"], "lab.txt", { type: "text/plain" });
  fireEvent.change(uploadInput, { target: { files: [file] } });
  await waitFor(async () => {
    const textarea = await screen.findByPlaceholderText(/Paste lab text or type your question/i);
    expect((textarea as HTMLTextAreaElement).value).toBe("mock extracted text");
  });
});

test("can create a new conversation from the sidebar", async () => {
  render(<ClassicChatbotUI />);
  const newChatBtn = await screen.findByRole("button", { name: /New chat/i });
  fireEvent.click(newChatBtn);
  const listButtons = screen.getAllByRole("button", { name: /New chat|Welcome/i });
  expect(listButtons.length).toBeGreaterThanOrEqual(2);
});
