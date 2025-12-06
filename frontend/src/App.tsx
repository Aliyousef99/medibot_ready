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
    <div>
      <header className="p-4 bg-gray-100 border-b">
        <Link to="/" className="text-blue-500 hover:underline">
          &larr; Back to Chat
        </Link>
      </header>
      <HistoryPage />
    </div>
  );
};

export default App;
