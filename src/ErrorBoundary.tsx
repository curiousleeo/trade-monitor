import { Component, ReactNode } from 'react';

interface Props { children: ReactNode }
interface State { error: string | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(err: Error): State {
    return { error: err.message };
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: '100vh', background: '#0a0a0a', color: '#ef4444',
          fontFamily: 'monospace', fontSize: '13px', padding: '40px',
          flexDirection: 'column', gap: '12px'
        }}>
          <div>⚠ RENDER ERROR</div>
          <div style={{ color: '#6b7280', maxWidth: '600px', textAlign: 'center' }}>
            {this.state.error}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
