"""Small synthetic Home Credit-shaped tables, so tests never need the real 1.5 GB dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credscore.features import DAYS_EMPLOYED_ANOMALY


def make_raw_tables(n: int = 3000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100_000, 100_000 + n)
    ext1 = np.where(rng.random(n) < 0.5, np.nan, rng.uniform(0.05, 0.95, n))
    ext2, ext3 = rng.uniform(0.02, 0.85, n), rng.uniform(0.02, 0.9, n)
    income = rng.lognormal(11.9, 0.45, n)
    credit = income * rng.uniform(1, 6, n)
    annuity = credit * rng.uniform(0.03, 0.09, n)
    pensioner = rng.random(n) < 0.15
    children = rng.poisson(0.4, n)
    owns_car = rng.random(n) < 0.35

    logit = -2.6 - 3.5 * (ext2 - 0.45) - 2.5 * (ext3 - 0.45) + 0.25 * (credit / income - 3.5)
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    application = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M", "XNA"], n, p=[0.65, 0.349, 0.001]),
        "FLAG_OWN_CAR": np.where(owns_car, "Y", "N"),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.7, 0.3]),
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income.round(-2),
        "AMT_CREDIT": credit.round(-2),
        "AMT_ANNUITY": annuity.round(1),
        "AMT_GOODS_PRICE": (credit * 0.9).round(-2),
        "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Family", None], n, p=[0.8, 0.18, 0.02]),
        "NAME_INCOME_TYPE": np.where(pensioner, "Pensioner",
                                     rng.choice(["Working", "Commercial associate", "State servant"], n)),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"], n),
        "NAME_FAMILY_STATUS": rng.choice(
            ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], n),
        "NAME_HOUSING_TYPE": rng.choice(
            ["House / apartment", "With parents", "Rented apartment", "Municipal apartment"], n),
        "DAYS_BIRTH": -(rng.uniform(21, 68, n) * 365.25).round(),
        "DAYS_EMPLOYED": np.where(pensioner, DAYS_EMPLOYED_ANOMALY, -(rng.uniform(0.1, 20, n) * 365.25).round()),
        "DAYS_REGISTRATION": -rng.uniform(0, 12000, n).round(),
        "OWN_CAR_AGE": np.where(owns_car, rng.integers(0, 25, n), np.nan),
        "FLAG_MOBIL": 1,
        "FLAG_PHONE": rng.integers(0, 2, n),
        "OCCUPATION_TYPE": np.where(pensioner, None,
                                    rng.choice(["Laborers", "Sales staff", "Core staff", "Managers", "Drivers"], n)),
        "CNT_FAM_MEMBERS": (children + rng.integers(1, 3, n)).astype(float),
        "REGION_RATING_CLIENT": (region := rng.integers(1, 4, n)),
        "REGION_RATING_CLIENT_W_CITY": region,
        "WEEKDAY_APPR_PROCESS_START": rng.choice(["MONDAY", "TUESDAY", "FRIDAY"], n),
        "HOUR_APPR_PROCESS_START": rng.integers(8, 20, n),
        "ORGANIZATION_TYPE": np.where(pensioner, "XNA",
                                      rng.choice(["Business Entity Type 3", "Self-employed", "Government"], n)),
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "APARTMENTS_AVG": np.where(rng.random(n) < 0.5, np.nan, rng.uniform(0, 0.3, n)),
        "APARTMENTS_MODE": rng.uniform(0, 0.3, n),
        "APARTMENTS_MEDI": rng.uniform(0, 0.3, n),
        "TOTALAREA_MODE": np.where(rng.random(n) < 0.5, np.nan, rng.uniform(0, 0.3, n)),
        "HOUSETYPE_MODE": rng.choice(["block of flats", None], n),
        "FLAG_DOCUMENT_2": 0,
        "FLAG_DOCUMENT_3": rng.integers(0, 2, n),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(n) < 0.1, np.nan, rng.poisson(1.5, n)),
    })

    risky = target == 1
    bureau_rows, previous_rows, installment_rows = [], [], []
    bureau_id = prev_id = 0
    for applicant, is_risky in zip(ids, risky):
        if rng.random() < 0.85:
            for _ in range(rng.integers(1, 9)):
                bureau_id += 1
                amount = rng.uniform(20e3, 800e3)
                active = rng.random() < (0.55 if is_risky else 0.35)
                bureau_rows.append({
                    "SK_ID_CURR": applicant, "SK_ID_BUREAU": bureau_id,
                    "CREDIT_ACTIVE": "Active" if active else "Closed",
                    "DAYS_CREDIT": -rng.integers(0, 2900), "CREDIT_DAY_OVERDUE": int(rng.random() < 0.03) * 30,
                    "DAYS_CREDIT_ENDDATE": rng.integers(-2000, 2000),
                    "AMT_CREDIT_MAX_OVERDUE": np.nan if rng.random() < 0.5 else rng.choice([0, 5000]),
                    "CNT_CREDIT_PROLONG": 0, "AMT_CREDIT_SUM": amount,
                    "AMT_CREDIT_SUM_DEBT": amount * rng.uniform(0.2, 0.9) if active else 0.0,
                    "AMT_CREDIT_SUM_OVERDUE": 0.0,
                })
        if rng.random() < 0.9:
            for _ in range(rng.integers(1, 6)):
                prev_id += 1
                requested = rng.uniform(20e3, 500e3)
                status = rng.choice(["Approved", "Refused", "Canceled"], p=[0.5, 0.4, 0.1] if is_risky else [0.75, 0.15, 0.1])
                previous_rows.append({
                    "SK_ID_PREV": prev_id, "SK_ID_CURR": applicant, "NAME_CONTRACT_STATUS": status,
                    "AMT_APPLICATION": requested, "AMT_CREDIT": requested * rng.uniform(0.9, 1.1),
                    "AMT_ANNUITY": requested / 12, "DAYS_DECISION": -rng.integers(1, 2900),
                    "CNT_PAYMENT": float(rng.choice([6, 12, 24])),
                })
                if status == "Approved":
                    late = rng.random() < (0.25 if is_risky else 0.05)
                    for k in range(rng.integers(3, 12)):
                        due = -1000 + 30 * k
                        installment_rows.append({
                            "SK_ID_PREV": prev_id, "SK_ID_CURR": applicant, "NUM_INSTALMENT_VERSION": 1,
                            "NUM_INSTALMENT_NUMBER": k + 1, "DAYS_INSTALMENT": due,
                            "DAYS_ENTRY_PAYMENT": due + (rng.integers(1, 30) if late else -rng.integers(0, 10)),
                            "AMT_INSTALMENT": requested / 12, "AMT_PAYMENT": requested / 12,
                        })
    return {
        "application": application,
        "bureau": pd.DataFrame(bureau_rows),
        "previous": pd.DataFrame(previous_rows),
        "installments": pd.DataFrame(installment_rows),
    }
