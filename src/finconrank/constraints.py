from __future__ import annotations

import re
from typing import Any


METRICS: dict[str, dict[str, Any]] = {
    "revenue": {"statement": "income_statement", "aliases": ["revenue", "revenues", "sales", "turnover", "net sales", "营业收入", "营业总收入", "收入"]},
    "operating_income": {"statement": "income_statement", "aliases": ["operating income", "operating profit", "income from operations", "营业利润", "经营利润"]},
    "net_income": {"statement": "income_statement", "aliases": ["net income", "net profit", "net earnings", "profit attributable", "归母净利润", "净利润", "归属于母公司股东的净利润"]},
    "gross_profit": {"statement": "income_statement", "aliases": ["gross profit", "gross income", "毛利", "毛利润"]},
    "eps": {"statement": "income_statement", "aliases": ["earnings per share", "diluted eps", "basic eps", "eps", "每股收益", "基本每股收益", "稀释每股收益"]},
    "total_assets": {"statement": "balance_sheet", "aliases": ["total assets", "assets total", "总资产", "资产总计"]},
    "liabilities": {"statement": "balance_sheet", "aliases": ["total liabilities", "liabilities", "负债合计", "总负债", "负债总计"]},
    "equity": {"statement": "balance_sheet", "aliases": ["shareholders' equity", "stockholders' equity", "total equity", "equity", "股东权益", "所有者权益", "权益合计"]},
    "cash": {"statement": "balance_sheet", "aliases": ["cash and cash equivalents", "cash", "cash balance", "货币资金", "现金及现金等价物"]},
    "inventory": {"statement": "balance_sheet", "aliases": ["inventory", "inventories", "存货"]},
    "receivables": {"statement": "balance_sheet", "aliases": ["accounts receivable", "trade receivables", "receivables", "应收账款", "应收款项"]},
    "operating_cash_flow": {"statement": "cash_flow", "aliases": ["operating cash flow", "cash from operating activities", "net cash provided by operating activities", "经营活动现金流", "经营活动产生的现金流量净额"]},
    "investing_cash_flow": {"statement": "cash_flow", "aliases": ["investing cash flow", "cash from investing activities", "投资活动现金流", "投资活动产生的现金流量净额"]},
    "financing_cash_flow": {"statement": "cash_flow", "aliases": ["financing cash flow", "cash from financing activities", "筹资活动现金流", "筹资活动产生的现金流量净额"]},
    "free_cash_flow": {"statement": "cash_flow", "aliases": ["free cash flow", "fcf", "自由现金流"]},
    "gross_margin": {"statement": "income_statement", "aliases": ["gross margin", "gross profit margin", "毛利率"]},
    "net_margin": {"statement": "income_statement", "aliases": ["net margin", "net profit margin", "净利率", "销售净利率"]},
    "roe": {"statement": "ratio", "aliases": ["return on equity", "roe", "净资产收益率"]},
    "roa": {"statement": "ratio", "aliases": ["return on assets", "roa", "总资产收益率", "资产收益率"]},
    "debt_ratio": {"statement": "ratio", "aliases": ["debt ratio", "debt-to-assets", "debt to asset ratio", "资产负债率"]},
}

STATEMENTS = {
    "income_statement": ["income statement", "statement of operations", "profit and loss", "利润表", "损益表", "合并利润表"],
    "balance_sheet": ["balance sheet", "statement of financial position", "资产负债表", "合并资产负债表"],
    "cash_flow": ["cash flow statement", "statement of cash flows", "现金流量表", "合并现金流量表"],
    "notes": ["notes to", "financial statement notes", "附注", "财务报表附注"],
}

UNITS = ["million", "millions", "billion", "billions", "thousand", "rmb", "cny", "usd", "eur", "%", "percent", "百万元", "千元", "万元", "亿元", "人民币", "美元", "欧元", "百分比"]
OPERATIONS = ["increase", "decrease", "growth", "change", "difference", "compare", "higher", "lower", "ratio", "trend", "calculate", "增长", "增加", "下降", "减少", "变化", "差异", "比较", "高于", "低于", "占比", "比率", "趋势", "计算"]
PERIODS = ["q1", "q2", "q3", "q4", "quarter", "quarterly", "fiscal year", "year ended", "six months", "nine months", "本期", "上期", "期末", "期初", "上年同期", "第一季度", "第二季度", "第三季度", "第四季度", "年度", "半年度", "报告期"]
SCOPES = ["consolidated", "parent company", "attributable to parent", "adjusted", "non-gaap", "gaap", "合并", "母公司", "归属于母公司", "调整后", "非公认会计准则", "非经常性"]
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _contains(text: str, term: str) -> bool:
    if re.fullmatch(r"[a-z0-9% .'-]+", term):
        return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None
    return term in text


def find_terms(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return sorted({term for term in terms if _contains(lowered, term.lower())}, key=lambda item: (-len(item), item))


def extract_entity(query: str) -> tuple[str | None, float]:
    patterns = [
        r"according to\s+(.{3,100}?)(?:'s|’s|\s+annual|\s+financial|,|\?)",
        r"in\s+(.{3,100}?)'s\s+(?:annual|financial)",
        r"根据(.{2,60}?)(?:的|20\d{2}年|年度报告|财务报表|，|,)",
        r"在(.{2,60}?)(?:的|首次公开发行|年度报告|财务报表)中",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" ,，。'")
            if len(value) >= 2:
                return value, 0.85
    return None, 0.0


def extract_constraints(query: str) -> dict[str, Any]:
    text = (query or "").lower()
    canonical = []
    aliases = []
    for metric, meta in METRICS.items():
        hits = find_terms(text, meta["aliases"])
        if hits:
            canonical.append(metric)
            aliases.extend(hits)
    statement = "unknown"
    for name, terms in STATEMENTS.items():
        if find_terms(text, terms):
            statement = name
            break
    if statement == "unknown" and canonical:
        inferred = {METRICS[name]["statement"] for name in canonical}
        if len(inferred) == 1:
            statement = next(iter(inferred))
    entity, entity_confidence = extract_entity(query or "")
    result = {
        "entity": entity,
        "entity_confidence": entity_confidence,
        "canonical_metrics": sorted(set(canonical)),
        "metric_terms": sorted(set(aliases), key=lambda item: (-len(item), item)),
        "year_terms": sorted(set(YEAR_RE.findall(text))),
        "period_terms": find_terms(text, PERIODS),
        "unit_terms": find_terms(text, UNITS),
        "statement_type": statement,
        "operation_terms": find_terms(text, OPERATIONS),
        "accounting_scope": find_terms(text, SCOPES),
    }
    observed = sum(bool(result[key]) for key in ("entity", "canonical_metrics", "year_terms", "period_terms", "unit_terms", "operation_terms", "accounting_scope"))
    if statement != "unknown":
        observed += 1
    result["confidence"] = min(1.0, observed / 4.0)
    result["unknown"] = observed == 0
    return result


def classify_query(query: str, constraints: dict[str, Any]) -> str:
    text = (query or "").lower()
    if constraints["statement_type"] == "cash_flow" or any("cash_flow" in name for name in constraints["canonical_metrics"]):
        return "cash_flow"
    if constraints["statement_type"] == "balance_sheet" or any(name in {"total_assets", "liabilities", "equity", "cash", "inventory", "receivables"} for name in constraints["canonical_metrics"]):
        return "balance_sheet"
    if constraints["statement_type"] == "income_statement" and not constraints["operation_terms"]:
        return "income_statement"
    if find_terms(text, ["growth", "increase", "decrease", "change", "ratio", "margin", "增长", "增加", "下降", "变化率", "比率", "占比", "毛利率", "净利率"]):
        return "ratio_growth"
    if find_terms(text, ["compare", "higher", "lower", "highest", "lowest", "difference", "比较", "高于", "低于", "最高", "最低", "差异"]):
        return "table_comparison"
    if constraints["year_terms"] or constraints["canonical_metrics"] or re.search(r"\d|多少|how much|calculate", text):
        return "numeric_lookup"
    return "other_financial"

