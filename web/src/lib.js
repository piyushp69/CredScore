/* Formatting, CSV helpers and example applicants. */

export const pct = (v, digits = 1) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
export const num = (v, digits = 0) =>
  v == null ? "—" : v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
export const dec = (v, digits = 3) => (v == null ? "—" : v.toFixed(digits));

export const DECISION = {
  APPROVE: { label: "Approve", icon: "✓", color: "var(--good)" },
  REVIEW: { label: "Manual review", icon: "!", color: "var(--warning)" },
  DECLINE: { label: "Decline", icon: "✕", color: "var(--critical)" },
};
export const BAND_COLOR = {
  good: "var(--good)",
  warning: "var(--warning)",
  serious: "var(--serious)",
  critical: "var(--critical)",
};

export function download(filename, content, mime = "text/plain") {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** Minimal RFC-4180 CSV parser: handles quoted fields, embedded commas and newlines. */
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  const push = () => {
    row.push(value);
    value = "";
  };
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        value += '"';
        i += 1;
      } else if (char === '"') quoted = false;
      else value += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") push();
    else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i += 1;
      push();
      rows.push(row);
      row = [];
    } else value += char;
  }
  if (value || row.length) {
    push();
    rows.push(row);
  }
  const [header = [], ...body] = rows.filter((r) => r.some((cell) => cell !== ""));
  return body.map((cells) =>
    Object.fromEntries(header.map((key, i) => [key.trim(), coerce(cells[i])])),
  );
}

function coerce(raw) {
  const value = (raw ?? "").trim();
  if (value === "") return null;
  if (/^(true|false)$/i.test(value)) return value.toLowerCase() === "true";
  const asNumber = Number(value);
  return Number.isNaN(asNumber) ? value : asNumber;
}

export function toCsv(rows, columns) {
  const keys = columns ?? [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const cell = (v) => {
    if (v == null) return "";
    const text = String(v);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [keys.join(","), ...rows.map((row) => keys.map((k) => cell(row[k])).join(","))].join("\n");
}

/* ---------- example applicants ---------- */
export const STRONG = {
  age_years: 45, family_status: "Married", children: 1, family_members: 3, education: "Higher education",
  housing_type: "House / apartment", owns_realty: true, owns_car: true, car_age_years: 5, region_rating: 1,
  income_type: "State servant", occupation: "Core staff", organization_type: null, annual_income: 270000,
  years_employed: 12, contract_type: "Cash loans", credit_amount: 450000, annuity_amount: 22000,
  goods_price: 450000, ext_source_1: 0.72, ext_source_2: 0.74, ext_source_3: 0.7, bureau_loans: 6,
  bureau_active_loans: 1, bureau_total_credit: 1200000, bureau_total_debt: 90000, bureau_overdue_amount: 0,
  bureau_days_overdue: 0, bureau_years_since_last_loan: 2, bureau_enquiries_last_year: 1, prev_applications: 4,
  prev_approved: 4, prev_refused: 0, installments_paid: 60, late_payment_share: 0, avg_days_past_due: 0,
  max_days_past_due: 0, payment_ratio: 1,
};

export const RISKY = {
  age_years: 24, family_status: "Single / not married", children: 0, family_members: 1,
  education: "Secondary / secondary special", housing_type: "With parents", owns_realty: false, owns_car: false,
  car_age_years: null, region_rating: 3, income_type: "Working", occupation: "Laborers", organization_type: null,
  annual_income: 90000, years_employed: 0.5, contract_type: "Cash loans", credit_amount: 900000,
  annuity_amount: 45000, goods_price: 810000, ext_source_1: null, ext_source_2: 0.15, ext_source_3: 0.2,
  bureau_loans: 6, bureau_active_loans: 5, bureau_total_credit: 600000, bureau_total_debt: 450000,
  bureau_overdue_amount: 25000, bureau_days_overdue: 30, bureau_years_since_last_loan: 0.2,
  bureau_enquiries_last_year: 6, prev_applications: 5, prev_approved: 2, prev_refused: 3, installments_paid: 20,
  late_payment_share: 0.3, avg_days_past_due: 6, max_days_past_due: 40, payment_ratio: 0.9,
};

/** Deterministic pseudo-random variations of the presets, for trying batch scoring. */
export function demoBatch(typical, n = 250, seed = 7) {
  let state = seed;
  const random = () => {
    state = (state * 1103515245 + 12345) % 2 ** 31;
    return state / 2 ** 31;
  };
  const jitter = (value, spread) => value * (1 + (random() - 0.5) * spread);
  const anchors = [typical, STRONG, RISKY];
  return Array.from({ length: n }, (_, i) => {
    const row = { ...anchors[random() < 0.6 ? 0 : random() < 0.62 ? 1 : 2] };
    const income = Math.max(30000, Math.round(jitter(row.annual_income, 0.9) / 1000) * 1000);
    const credit = Math.max(50000, Math.round(jitter(row.credit_amount, 1.1) / 1000) * 1000);
    return {
      ...row,
      applicant_id: `DEMO-${String(i + 1).padStart(4, "0")}`,
      age_years: Math.min(68, Math.max(21, Math.round(jitter(row.age_years, 0.45)))),
      annual_income: income,
      credit_amount: credit,
      goods_price: Math.round((credit * (0.85 + random() * 0.15)) / 1000) * 1000,
      annuity_amount: Math.round(credit * (0.03 + random() * 0.05)),
      ext_source_2: row.ext_source_2 == null ? null : clamp01(row.ext_source_2 + (random() - 0.5) * 0.3),
      ext_source_3: row.ext_source_3 == null ? null : clamp01(row.ext_source_3 + (random() - 0.5) * 0.3),
    };
  });
}

const clamp01 = (v) => Math.round(Math.min(0.95, Math.max(0.01, v)) * 1000) / 1000;
