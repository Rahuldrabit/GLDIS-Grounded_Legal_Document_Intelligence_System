import React from 'react';
import './Layout.css';

interface LayoutProps {
  children: React.ReactNode;
  onNavigate: (page: string, params?: Record<string, string>) => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, onNavigate }) => {
  return (
    <div className="layout-container">
      <aside className="sidebar glass-panel">
        <div className="sidebar-header">
          <h2>GLDIS</h2>
          <p className="subtitle">Intelligence System</p>
        </div>
        <nav className="sidebar-nav">
          <button className="nav-item btn btn-secondary" onClick={() => onNavigate('dashboard')}>
            Dashboard
          </button>
        </nav>
      </aside>
      <main className="main-content">
        <header className="topbar glass-panel animate-fade-in">
          <div className="topbar-title">Workspace</div>
        </header>
        <div className="content-area animate-fade-in">
          {children}
        </div>
      </main>
    </div>
  );
};
