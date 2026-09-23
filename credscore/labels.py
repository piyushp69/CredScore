"""Human-readable names and value formatting for model features."""

from __future__ import annotations

import math

# feature -> (label, value kind). Kinds drive `format_value`.
FEATURE_INFO: dict[str, tuple[str, str]] = {
    # External scores
    "EXT_SOURCE_1": ("External score 1", "score"),
    "EXT_SOURCE_2": ("External score 2", "score"),
    "EXT_SOURCE_3": ("External score 3", "score"),
    "EXT_SOURCE_MEAN": ("External scores (average)", "score"),
    "EXT_SOURCE_MIN": ("External scores (lowest)", "score"),
    "EXT_SOURCE_MAX": ("External scores (highest)", "score"),
    "EXT_SOURCE_STD": ("External scores (spread)", "score"),
    # Applicant
    "DAYS_BIRTH": ("Age", "age"),
    "CNT_CHILDREN": ("Children", "count"),
    "CNT_FAM_MEMBERS": ("Family members", "count"),
    "NAME_FAMILY_STATUS": ("Family status", "category"),
    "NAME_EDUCATION_TYPE": ("Education", "category"),
    "NAME_HOUSING_TYPE": ("Housing situation", "category"),
    "NAME_TYPE_SUITE": ("Accompanied by", "category"),
    "FLAG_OWN_CAR": ("Owns a car", "flag"),
    "FLAG_OWN_REALTY": ("Owns real estate", "flag"),
    "OWN_CAR_AGE": ("Car age (years)", "count"),
    "REGION_RATING_CLIENT": ("Region rating", "count"),
    "REGION_RATING_CLIENT_W_CITY": ("Region rating incl. city", "count"),
    "REGION_POPULATION_RELATIVE": ("Region population density", "number"),
    "DAYS_REGISTRATION": ("Time since registration change", "duration"),
    "DAYS_ID_PUBLISH": ("Time since ID document issued", "duration"),
    "DAYS_LAST_PHONE_CHANGE": ("Time since phone number change", "duration"),
    "FLAG_DOCUMENT_3": ("Provided document 3", "flag"),
    "FLAG_EMP_PHONE": ("Provided employer phone", "flag"),
    "FLAG_WORK_PHONE": ("Provided work phone", "flag"),
    "FLAG_CONT_MOBILE": ("Mobile phone reachable", "flag"),
    "FLAG_PHONE": ("Provided home phone", "flag"),
    "FLAG_EMAIL": ("Provided email", "flag"),
    "REG_REGION_NOT_LIVE_REGION": ("Registered and home region differ", "flag"),
    "REG_REGION_NOT_WORK_REGION": ("Registered and work region differ", "flag"),
    "LIVE_REGION_NOT_WORK_REGION": ("Home and work region differ", "flag"),
    "REG_CITY_NOT_LIVE_CITY": ("Registered and home city differ", "flag"),
    "REG_CITY_NOT_WORK_CITY": ("Registered and work city differ", "flag"),
    "LIVE_CITY_NOT_WORK_CITY": ("Home and work city differ", "flag"),
    "OBS_30_CNT_SOCIAL_CIRCLE": ("Social circle: observed 30 DPD", "count"),
    "DEF_30_CNT_SOCIAL_CIRCLE": ("Social circle: defaulted 30 DPD", "count"),
    "OBS_60_CNT_SOCIAL_CIRCLE": ("Social circle: observed 60 DPD", "count"),
    "DEF_60_CNT_SOCIAL_CIRCLE": ("Social circle: defaulted 60 DPD", "count"),
    "TOTALAREA_MODE": ("Building total area (normalized)", "number"),
    # Employment & income
    "NAME_INCOME_TYPE": ("Income type", "category"),
    "OCCUPATION_TYPE": ("Occupation", "category"),
    "ORGANIZATION_TYPE": ("Employer type", "category"),
    "DAYS_EMPLOYED": ("Time in current job", "duration"),
    "EMPLOYED_AGE_RATIO": ("Share of life in current job", "percent"),
    "AMT_INCOME_TOTAL": ("Annual income", "amount"),
    "INCOME_PER_PERSON": ("Income per family member", "amount"),
    # Loan request
    "NAME_CONTRACT_TYPE": ("Contract type", "category"),
    "AMT_CREDIT": ("Credit amount", "amount"),
    "AMT_ANNUITY": ("Loan annuity", "amount"),
    "AMT_GOODS_PRICE": ("Goods price", "amount"),
    "CREDIT_INCOME_RATIO": ("Credit-to-income ratio", "ratio"),
    "ANNUITY_INCOME_RATIO": ("Annuity-to-income ratio", "percent"),
    "PAYMENT_RATE": ("Payment rate (annuity / credit)", "percent"),
    "CREDIT_GOODS_RATIO": ("Credit-to-goods-price ratio", "ratio"),
    # Credit bureau
    "AMT_REQ_CREDIT_BUREAU_HOUR": ("Bureau enquiries (last hour)", "count"),
    "AMT_REQ_CREDIT_BUREAU_DAY": ("Bureau enquiries (last day)", "count"),
    "AMT_REQ_CREDIT_BUREAU_WEEK": ("Bureau enquiries (last week)", "count"),
    "AMT_REQ_CREDIT_BUREAU_MON": ("Bureau enquiries (last month)", "count"),
    "AMT_REQ_CREDIT_BUREAU_QRT": ("Bureau enquiries (last quarter)", "count"),
    "AMT_REQ_CREDIT_BUREAU_YEAR": ("Bureau enquiries (last year)", "count"),
    "BUREAU_LOAN_COUNT": ("Credit bureau loans", "count"),
    "BUREAU_ACTIVE_COUNT": ("Active bureau loans", "count"),
    "BUREAU_ACTIVE_RATIO": ("Share of bureau loans still active", "percent"),
    "BUREAU_DAYS_CREDIT_MEAN": ("Average age of bureau loans", "duration"),
    "BUREAU_DAYS_CREDIT_MAX": ("Time since latest bureau loan", "duration"),
    "BUREAU_DAYS_CREDIT_ENDDATE_MEAN": ("Average remaining bureau term (days)", "days"),
    "BUREAU_CREDIT_DAY_OVERDUE_MAX": ("Days overdue on bureau loans", "days"),
    "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX": ("Largest overdue amount (bureau)", "amount"),
    "BUREAU_AMT_CREDIT_SUM": ("Total bureau credit", "amount"),
    "BUREAU_AMT_CREDIT_SUM_DEBT": ("Outstanding bureau debt", "amount"),
    "BUREAU_AMT_CREDIT_SUM_OVERDUE": ("Currently overdue (bureau)", "amount"),
    "BUREAU_CNT_CREDIT_PROLONG": ("Bureau loan prolongations", "count"),
    "BUREAU_DEBT_CREDIT_RATIO": ("Bureau debt-to-credit ratio", "percent"),
    "BUREAU_DEBT_INCOME_RATIO": ("Bureau debt-to-income ratio", "ratio"),
    # Previous applications
    "PREV_APP_COUNT": ("Previous applications", "count"),
    "PREV_APPROVED_COUNT": ("Previous applications approved", "count"),
    "PREV_REFUSED_COUNT": ("Previous applications refused", "count"),
    "PREV_REFUSED_RATIO": ("Share of previous applications refused", "percent"),
    "PREV_APPROVED_RATIO": ("Share of previous applications approved", "percent"),
    "PREV_AMT_APPLICATION_MEAN": ("Average amount requested before", "amount"),
    "PREV_AMT_CREDIT_MEAN": ("Average amount granted before", "amount"),
    "PREV_AMT_ANNUITY_MEAN": ("Average previous annuity", "amount"),
    "PREV_APP_CREDIT_RATIO_MEAN": ("Requested-to-granted ratio (previous)", "ratio"),
    "PREV_DAYS_DECISION_MEAN": ("Average time since previous decisions", "duration"),
    "PREV_DAYS_DECISION_MAX": ("Time since last previous decision", "duration"),
    "PREV_CNT_PAYMENT_MEAN": ("Average previous term (payments)", "count"),
    # Repayment history
    "INSTAL_COUNT": ("Installments on record", "count"),
    "INSTAL_DPD_MEAN": ("Average days past due", "days"),
    "INSTAL_DPD_MAX": ("Worst days past due", "days"),
    "INSTAL_DBD_MEAN": ("Average days paid early", "days"),
    "INSTAL_LATE_SHARE": ("Share of installments paid late", "percent"),
    "INSTAL_PAYMENT_RATIO_MEAN": ("Average share of installment paid", "percent"),
    "INSTAL_PAYMENT_SHORTFALL_MEAN": ("Average installment shortfall", "amount"),
    "INSTAL_AMT_PAYMENT_SUM": ("Total repaid on previous loans", "amount"),
}


def feature_label(feature: str) -> str:
    if feature in FEATURE_INFO:
        return FEATURE_INFO[feature][0]
    if feature.endswith("_AVG"):
        return "Building: " + feature[: -len("_AVG")].replace("_", " ").lower()
    return feature.replace("_", " ").capitalize()


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def format_value(feature: str, value) -> str:
    if _is_missing(value):
        return "not available"
    kind = FEATURE_INFO.get(feature, ("", "number"))[1]
    if kind == "category" or isinstance(value, str):
        return str(value)
    value = float(value)
    if kind == "age":
        return f"{-value / 365.25:.0f} years"
    if kind == "duration":
        years = -value / 365.25
        return f"{-value:.0f} days" if abs(years) < 1 else f"{years:.1f} years"
    if kind == "days":
        return f"{value:.0f} days"
    if kind == "amount":
        return f"{value:,.0f}"
    if kind == "ratio":
        return f"{value:.2f}x"
    if kind == "percent":
        return f"{value:.1%}"
    if kind == "score":
        return f"{value:.3f}"
    if kind == "count":
        return f"{value:.0f}" if value == round(value) else f"{value:.1f}"
    if kind == "flag":
        return "yes" if value >= 0.5 else "no"
    return f"{value:.4g}"
