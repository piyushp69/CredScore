import { useEffect, useState } from "react";
import { api } from "./api.js";
import Underwriting from "./pages/Underwriting.jsx";
import Batch from "./pages/Batch.jsx";
import Portfolio from "./pages/Portfolio.jsx";
import Performance from "./pages/Performance.jsx";

const PAGES = [
  { id: "underwriting", label: "Underwriting", Component: Underwriting },
  { id: "batch", label: "Batch scoring", Component: Batch },
  { id: "portfolio", label: "Portfolio insights", Component: Portfolio },
  { id: "performance", label: "Model performance", Component: Performance },
];

function Icon({ id }) {
  const paths = {
    underwriting: "M10 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm-6 8a6 6 0 0 1 9.5-4.9M15.5 16.5a2.5 2.5 0 1 0 3.5 3.6 2.5 2.5 0 0 0-3.5-3.6Zm3.7 3.7L21.5 22",
    batch: "M3.5 5.5h17v13h-17zM3.5 10h17M9.5 10v8.5M15 10v8.5",
    portfolio: "M4 19.5h16M7 19V11m5 8V5.5M17 19v-5.5",
    performance: "M4.5 17a8 8 0 1 1 15 0M12 17l4-5.5",
  };
  return (
    <svg className="nav-icon" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={paths[id]} />
    </svg>
  );
}

function useHashRoute() {
  const read = () => window.location.hash.replace("#/", "") || PAGES[0].id;
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => setRoute(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return [route, (id) => (window.location.hash = `/${id}`)];
}

function ApiStatus() {
  const [health, setHealth] = useState(null);
  useEffect(() => {
    const check = () => api.health().then(setHealth).catch(() => setHealth({ status: "unreachable" }));
    check();
    const timer = setInterval(check, 30000);
    return () => clearInterval(timer);
  }, []);
  if (!health) return <div className="status">Checking API…</div>;
  if (health.model_loaded) {
    return (
      <div className="status ok">
        <span className="pulse" />
        <span>
          API online
          <br />
          <span className="tiny">model {health.model_version}</span>
        </span>
      </div>
    );
  }
  if (health.status === "degraded") {
    return <div className="status warn">API online, no model loaded. Run the training pipeline.</div>;
  }
  return <div className="status bad">API unreachable on :8000</div>;
}

export default function App() {
  const [route, go] = useHashRoute();
  const page = PAGES.find((p) => p.id === route) ?? PAGES[0];
  const { Component } = page;
  useEffect(() => {
    document.title = `CredScore · ${page.label}`;
  }, [page.label]);

  return (
    <div className="shell">
      <aside className="sidebar glass" style={{ borderRadius: 0, borderTop: 0, borderBottom: 0, borderLeft: 0 }}>
        <div className="brand">
          <div className="brand-mark">CS</div>
          <div>
            <div className="brand-name">CredScore</div>
            <div className="brand-sub">Credit risk scoring</div>
          </div>
        </div>
        <nav className="nav">
          {PAGES.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${item.id === route ? "active" : ""}`}
              onClick={() => go(item.id)}
              aria-current={item.id === route ? "page" : undefined}
            >
              <Icon id={item.id} />
              {item.label}
            </button>
          ))}
        </nav>
        <div className="stack" style={{ marginTop: "auto" }}>
          <ApiStatus />
          <a className="tiny muted" href="/docs" target="_blank" rel="noreferrer">
            API documentation ↗
          </a>
        </div>
      </aside>
      <main className="main">
        <Component key={page.id} />
      </main>
    </div>
  );
}
