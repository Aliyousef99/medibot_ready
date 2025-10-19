import React, { useState, useEffect } from 'react';
import ClassicChatbotUI from './components/ClassicChatbotUI';
import HistoryPage from './pages/History';

const App: React.FC = () => {
  const [path, setPath] = useState(window.location.hash);

  useEffect(() => {
    const handleHashChange = () => {
      setPath(window.location.hash);
    };

    window.addEventListener('hashchange', handleHashChange);

    return () => {
      window.removeEventListener('hashchange', handleHashChange);
    };
  }, []);

  const renderPage = () => {
    switch (path) {
      case '#/history':
        return <HistoryPageWithHeader />;
      default:
        return <ClassicChatbotUI />;
    }
  };

  return renderPage();
};

// Wrapper for HistoryPage to include a header with a link back to the chat
const HistoryPageWithHeader: React.FC = () => {
  return (
    <div>
      <header className="p-4 bg-gray-100 border-b">
        <a href="#/" className="text-blue-500 hover:underline">
          &larr; Back to Chat
        </a>
      </header>
      <HistoryPage />
    </div>
  );
};

export default App;