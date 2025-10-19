import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ClassicChatbotUI from '../components/ClassicChatbotUI';

vi.mock('../services/api', () => {
  const now = new Date().toISOString();
  return {
    startConversation: vi.fn().mockResolvedValue({ conversation_id: 'conv-1' }),
    getHistory: vi.fn().mockResolvedValue({
      conversation: { id: 'conv-1', title: 'Conversation 1', created_at: now },
      messages: [
        { id: 'u1', role: 'user', content: 'Hi', created_at: now },
        { id: 'a1', role: 'assistant', content: 'Hello', created_at: now },
      ],
    }),
    sendMessage: vi.fn().mockResolvedValue({
      message: { id: 'a2', role: 'assistant', content: 'Assistant reply', created_at: now },
    }),
    patchConversation: vi.fn().mockResolvedValue({ id: 'conv-1', active_lab_id: 'lab-1', created_at: now }),
    getLabReports: vi.fn().mockResolvedValue([{ id: 'lab-1', title: 'Lab A', created_at: now }]),
    getConversations: vi.fn().mockResolvedValue([{ id: 'conv-1', title: 'Conversation 1' }]),
  };
});

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem('auth', JSON.stringify({ email: 'u@example.com', token: 't' }));
});

test('persists conversationId in localStorage', async () => {
  render(<ClassicChatbotUI />);
  await waitFor(() => expect(localStorage.getItem('conversationId')).toBe('conv-1'));
});

test('renders history and loads older (re-fetch) when switching', async () => {
  render(<ClassicChatbotUI />);
  // history from initial fetch
  expect(await screen.findByText('Hello')).toBeInTheDocument();
  // Switch to same conversation to simulate reload (acts as "load older")
  const select = await screen.findByRole('combobox');
  fireEvent.change(select, { target: { value: 'conv-1' } });
  // still renders messages after re-fetch
  expect(await screen.findByText('Hello')).toBeInTheDocument();
});

test('PATCH active lab updates context chip', async () => {
  render(<ClassicChatbotUI />);
  // Open lab selection modal via "Change" link (appears once active_lab_id is set)
  // First, simulate user opening modal and selecting a lab
  const changeBtns = await screen.findAllByText(/Change/i);
  fireEvent.click(changeBtns[0]);
  const labBtn = await screen.findByRole('button', { name: /Lab A/i });
  fireEvent.click(labBtn);
  // After patch, a context banner should be visible
  expect(await screen.findByText(/Context: Lab Report/i)).toBeInTheDocument();
});

test('sends message and shows assistant reply', async () => {
  render(<ClassicChatbotUI />);
  const textarea = await screen.findByPlaceholderText(/Paste lab text or type your question/i);
  fireEvent.change(textarea, { target: { value: 'What is my LDL?' } });
  const sendBtn = await screen.findByRole('button', { name: /Send/i });
  fireEvent.click(sendBtn);
  expect(await screen.findByText('Assistant reply')).toBeInTheDocument();
});
