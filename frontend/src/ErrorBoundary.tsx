import React from 'react';

type State = { hasError: boolean; info?: string };

export default class ErrorBoundary extends React.Component<React.PropsWithChildren<{}>, State> {
  constructor(props: {}) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: any): State {
    return { hasError: true, info: String(error?.message || error) };
  }

  componentDidCatch(error: any, errorInfo: any) {
    // eslint-disable-next-line no-console
    console.error('UI ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 16, color: '#e11d48', background: '#0b0b0b', minHeight: '100vh' }}>
          <h1 style={{ color: '#fff' }}>Something went wrong.</h1>
          <p style={{ color: '#ddd' }}>{this.state.info}</p>
          <div style={{ marginTop: 12 }}>
            <button
              onClick={() => {
                try { localStorage.removeItem('auth'); } catch {}
                location.reload();
              }}
              style={{ padding: '8px 12px', background: '#2563eb', color: '#fff', borderRadius: 8 }}
            >
              Clear login data and reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children as React.ReactElement;
  }
}

