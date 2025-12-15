import React from 'react';
import { Navigate, Route, Routes, Link } from 'react-router-dom';
import ClassicChatbotUI from './components/ClassicChatbotUI';
import HistoryPage from './pages/History';
import ToastHost from './components/ToastHost';
import { ProtectedRoute } from './components/ProtectedRoute';

const App: React.FC = () => {
  return (
    <>
      <Routes>
        <Route path="/" element={<ClassicChatbotUI />} />
        <Route
          path="/history"
          element={
            <ProtectedRoute>
              <HistoryPageWithHeader />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ToastHost />
    </>
  );
};

// Wrapper for HistoryPage to include a header with a link back to the chat
const HistoryPageWithHeader: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-900 via-zinc-950 to-emerald-950 text-zinc-100">
      <header className="p-4 border-b border-zinc-800 bg-zinc-900/80 backdrop-blur">
        <Link to="/" className="inline-flex items-center gap-2 text-emerald-300 hover:text-emerald-200 hover:underline">
          <span aria-hidden="true">&larr;</span> Back to Chat
        </Link>
      </header>
      <HistoryPage />
    </div>
  );
};

export default App;
