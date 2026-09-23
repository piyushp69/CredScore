import { CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BAND_COLOR } from "./lib.js";

/* Chart palette: validated dark-mode steps.
   categorical BLUE/ORANGE/AQUA (identity) · diverging BLUE<->RED (polarity)
   · status colours only where the colour means good/bad, always with a label. */
export const C = {
  blue: "#3987e5",
  orange: "#d95926",
  aqua: "#199e70",
  riskUp: "#e66767",
  riskDown: "#3987e5",
  ink: "#ffffff",
  ink2: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  axis: "#383835",
  neutral: "#6d6c66",
};

export const axisProps = {
  stroke: C.axis,
  tick: { fill: C.muted, fontSize: 12 },
  tickLine: false,
  axisLine: { stroke: C.axis },
};
export const gridProps = { stroke: C.grid, vertical: false };

/* ---------- surfaces ---------- */
export function Card({ children, className = "", hover = true, ...rest }) {
  return (
    <div className={`glass card ${hover ? "glass-hover" : ""} ${className}`} {...rest}>
      {children}
    </div>
  );
}

export function Stat({ label, value, sub, delta, deltaUp, title }) {
  return (
    <div className="glass glass-hover stat" title={title}>
      <span className="label">{label}</span>
      <span className="stat-value mono-num">{value}</span>
      {delta && <span className={`stat-delta ${deltaUp ? "up" : "down"}`}>{delta}</span>}
      {sub && <span className="tiny muted">{sub}</span>}
    </div>
  );
}

export function Chip({ color, children }) {
  return (
    <span className="chip">
      <span className="dot" style={{ background: color }} />
      {children}
    </span>
  );
}

export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab}
          type="button"
          role="tab"
          aria-selected={tab === active}
          className={`tab ${tab === active ? "active" : ""}`}
          onClick={() => onChange(tab)}
        >
          {tab}
        </button>
      ))}
    </div>
  );
}

export function Disclosure({ label, children }) {
  return (
    <details className="disclosure">
      <summary>{label}</summary>
      <div>{children}</div>
    </details>
  );
}

/* ---------- feedback ---------- */
export function Loading({ what = "data" }) {
  return (
    <div className="center-note">
      <div className="spinner" />
      <span>Loading {what}…</span>
    </div>
  );
}

export function ErrorBox({ error, onRetry }) {
  return (
    <div className="glass card stack">
      <div className="status bad">
        <span>⚠</span>
        <span>{error?.message ?? String(error)}</span>
      </div>
      {onRetry && (
        <button type="button" className="btn btn-sm" onClick={onRetry} style={{ alignSelf: "flex-start" }}>
          Try again
        </button>
      )}
    </div>
  );
}

/* ---------- form controls ---------- */
export function NumberField({ label, value, onChange, hint, optional, ...rest }) {
  return (
    <label className="field">
      <span>
        {label}
        {hint && (
          <span className="muted" title={hint} style={{ cursor: "help" }}>
            ⓘ
          </span>
        )}
      </span>
      <input
        className="input mono-num"
        type="number"
        value={value ?? ""}
        placeholder={optional ? "Not available" : ""}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        {...rest}
      />
    </label>
  );
}

export function SelectField({ label, value, options, onChange, optional, hint }) {
  return (
    <label className="field">
      <span>
        {label}
        {hint && (
          <span className="muted" title={hint} style={{ cursor: "help" }}>
            ⓘ
          </span>
        )}
      </span>
      <select
        className="input"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
      >
        {optional && <option value="">Not stated</option>}
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ToggleField({ label, value, onChange }) {
  return (
    <div className="field">
      <span className="label">{label}</span>
      <label className="switch">
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
        <span className="track" />
        <span className="small">{value ? "Yes" : "No"}</span>
      </label>
    </div>
  );
}

/* ---------- table ---------- */
export function Table({ columns, rows, maxHeight }) {
  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined}>
      <table>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.key} className={col.align === "right" ? "num" : ""}>
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.id ?? i}>
              {columns.map((col) => (
                <td key={col.key} className={col.align === "right" ? "num" : ""}>
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- charts ---------- */
export function ChartCard({ title, note, height = 300, legend, children, className = "" }) {
  return (
    <div className={`glass card glass-hover stack ${className}`}>
      <div className="spread">
        <h3>{title}</h3>
        {legend && (
          <div className="legend">
            {legend.map((item) => (
              <span className="legend-item" key={item.label}>
                <span className="legend-swatch" style={{ background: item.color, height: item.dash ? 2 : 3 }} />
                {item.label}
              </span>
            ))}
          </div>
        )}
      </div>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
      {note && <p className="tiny muted">{note}</p>}
    </div>
  );
}

/** Glass tooltip; `rows` maps the hovered payload to label/value lines. */
export function GlassTooltip({ rows }) {
  const content = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const lines = rows ? rows(payload[0].payload, payload) : payload.map((p) => ({ label: p.name, value: p.value }));
    return (
      <div
        className="glass"
        style={{ padding: "10px 12px", borderRadius: 12, fontSize: "0.82rem", color: C.ink2, minWidth: 130 }}
      >
        {label != null && <div style={{ color: C.ink, fontWeight: 600, marginBottom: 4 }}>{label}</div>}
        {lines.map((line, i) => (
          <div key={i} className="spread" style={{ gap: 14 }}>
            <span>{line.label}</span>
            <span className="mono-num" style={{ color: line.color ?? C.ink }}>
              {line.value}
            </span>
          </div>
        ))}
      </div>
    );
  };
  return <Tooltip content={content} cursor={{ fill: "rgba(255,255,255,0.05)", stroke: C.axis }} />;
}

export const Grid = () => <CartesianGrid {...gridProps} />;
export const X = (props) => <XAxis {...axisProps} {...props} />;
export const Y = (props) => <YAxis {...axisProps} {...props} />;

/* ---------- score scale ---------- */
export function ScoreScale({ score, bands }) {
  const ordered = [...bands].sort((a, b) => a.min_score - b.min_score);
  const min = ordered[0].min_score;
  const max = ordered[ordered.length - 1].max_score;
  const position = ((score - min) / (max - min)) * 100;
  return (
    <div>
      <div className="scale-marker">
        <span style={{ left: `${Math.min(97, Math.max(3, position))}%` }} className="mono-num">
          {score} ▾
        </span>
      </div>
      <div className="scale">
        {ordered.map((band) => {
          const on = score >= band.min_score && score <= band.max_score;
          return (
            <div
              key={band.code}
              className={`scale-band ${on ? "on" : ""}`}
              style={on ? { background: BAND_COLOR[band.status] } : undefined}
              title={`${band.label}: ${band.min_score}–${band.max_score}`}
            >
              {band.code}
            </div>
          );
        })}
      </div>
      <div className="scale-ticks mono-num">
        <span>{min}</span>
        <span>higher is safer</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
