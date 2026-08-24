
from __future__ import annotations

import argparse
import json
import math
import time
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXCEL_FILE = Path(r"C:\Users\USER\OneDrive\Research\SOX RAW.xlsx")
OUT_FILE = ROOT / "apex_predator_dashboard.html"
PAGES_FILE = ROOT / "index.html"
DATA_SOURCE = "yahoo"
FALLBACK_TO_EXCEL = True
YAHOO_TICKERS = {
    "SOX": "^SOX",
    "TWD": "TWD=X",
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "IXIC": "^IXIC",
}

# Change the start date when needed, then rerun this file to refresh every table and chart.
# End date defaults to "auto", which uses the latest trading day available from the data source.
# You can also override them from PowerShell:
# python generate_apex_predator_dashboard.py --start-date 2011-01-03 --end-date auto
BACKTEST_START_DATE = "2011-01-03"
BACKTEST_END_DATE = "auto"

CONFIG = {
    "START_DATE": BACKTEST_START_DATE,
    "END_DATE": BACKTEST_END_DATE,
    "CHART_START_DATE": "1993-01-01",
    "BASE_AMOUNT": 5000,
    "PRICE_COL": "Close",
    "VIX_COL": "Close_VIX",
    "RMDD_WINDOW": 252,
    "VIX_PANIC_THRESHOLD": 32,
    "REGULAR_IDX_DCA_PURE": 9,
    "LEVEL_CONFIG": [
        {"threshold": -0.10, "units": 1.0, "desc": "Lv1"},
        {"threshold": -0.175, "units": 1.25, "desc": "Lv2"},
        {"threshold": -0.375, "units": 1.5, "desc": "Lv3"},
        {"threshold": -0.425, "units": 1.75, "desc": "Lv4"},
        {"threshold": -0.575, "units": 2.0, "desc": "Lv5"},
    ],
    "SNIPER_CONFIG": [
        {"threshold": -0.175, "units": 0.75, "desc": "Sniper Shot 1"},
        {"threshold": -0.375, "units": 1.0, "desc": "Sniper Shot 2"},
        {"threshold": -0.425, "units": 1.5, "desc": "Sniper Shot 3"},
        {"threshold": -0.575, "units": 0.5, "desc": "Sniper Shot 4"},
    ],
    # 5/6/7 月少買已移除；只保留 10 月月底保底買入 1.5x。
    "SEASONAL_FACTORS": {10: 1.5},
}


def load_from_yahoo_once() -> tuple[pd.DataFrame, list[str]]:
    import yfinance as yf

    symbols = list(YAHOO_TICKERS.values())
    raw = yf.download(
        symbols,
        start="1993-01-01",
        end=None,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("Yahoo Finance response has no Close prices.")
    else:
        close = raw[["Close"]].copy()
        close.columns = [symbols[0]]

    close = close.rename(columns={symbol: name for name, symbol in YAHOO_TICKERS.items()})
    required = ["SOX", "VIX", "SPX", "IXIC"]
    missing = [name for name in required if name not in close.columns]
    if missing:
        raise ValueError(f"Yahoo Finance is missing required tickers: {missing}")

    merged = close.rename(
        columns={
            "SOX": "Close",
            "TWD": "Close_TWD",
            "SPX": "Close_SPX",
            "VIX": "Close_VIX",
            "IXIC": "Close_IXIC",
        }
    )
    merged.index = pd.to_datetime(merged.index).tz_localize(None)
    merged = merged.sort_index()
    for col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["Close", "Close_VIX", "Close_SPX", "Close_IXIC"])
    if merged.empty:
        raise ValueError("Yahoo Finance data is empty after alignment.")
    return merged, symbols


def load_from_yahoo() -> tuple[pd.DataFrame, list[str]]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return load_from_yahoo_once()
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                wait_seconds = 5 * (attempt + 1)
                print(f"Yahoo Finance attempt {attempt + 1} failed: {exc}")
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    assert last_error is not None
    raise last_error


def load_from_excel(file_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df_raw = pd.read_excel(file_path, header=1)
    market_data: dict[str, pd.DataFrame] = {}

    first_ticker = df_raw.columns[0]
    other_tickers: list[str] = []
    for val in df_raw.iloc[:, 0]:
        if isinstance(val, str) and val:
            other_tickers.append(val)
        else:
            break

    all_tickers = [first_ticker] + other_tickers
    for i, ticker in enumerate(all_tickers):
        col_time_idx = 1 + i * 2
        col_close_idx = 2 + i * 2
        if col_close_idx >= len(df_raw.columns):
            break

        sub = df_raw.iloc[:, [col_time_idx, col_close_idx]].copy()
        clean = ticker.replace(".", "").replace("=", "")
        sub.columns = ["Date", "Close"] if i == 0 else ["Date", f"Close_{clean}"]
        sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
        sub = sub.dropna(subset=["Date"]).set_index("Date")
        sub[sub.columns[0]] = pd.to_numeric(sub[sub.columns[0]], errors="coerce")
        market_data[clean] = sub

    merged = market_data["SOX"].copy()
    for name, df in market_data.items():
        if name != "SOX":
            merged = pd.merge(merged, df, left_index=True, right_index=True, how="inner")
    return merged.sort_index(), all_tickers


def load_market_data() -> tuple[pd.DataFrame, list[str], str]:
    if DATA_SOURCE.lower() == "yahoo":
        try:
            merged, tickers = load_from_yahoo()
            return merged, tickers, "Yahoo Finance"
        except Exception as exc:
            if not FALLBACK_TO_EXCEL:
                raise
            if not EXCEL_FILE.exists():
                raise RuntimeError(
                    "Yahoo Finance failed and the Excel fallback file is not available in this environment."
                ) from exc
            print(f"Yahoo Finance failed: {exc}")
            print("Falling back to Excel.")

    merged, tickers = load_from_excel(EXCEL_FILE)
    return merged, tickers, str(EXCEL_FILE)


def calculate_rmdd(df: pd.DataFrame, price_col: str, window: int) -> pd.DataFrame:
    out = df.copy().sort_index()
    out["Roll_Max"] = out[price_col].rolling(window, min_periods=1).max()
    out["RMDD"] = (out[price_col] - out["Roll_Max"]) / out["Roll_Max"]
    return out


def mark_fallback_days(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_index()
    out["Fallback_Day"] = False
    if out.empty:
        return out

    periods = pd.Series(out.index.to_period("M"), index=out.index)
    next_periods = periods.shift(-1)
    fallback_days = next_periods.notna() & (next_periods != periods)

    last_date = out.index[-1]
    next_business_day = last_date + pd.offsets.BDay(1)
    fallback_days.loc[last_date] = next_business_day.to_period("M") != last_date.to_period("M")
    out["Fallback_Day"] = fallback_days.astype(bool)
    return out


def xirr(cashflows: list[float], dates: list[pd.Timestamp]) -> float:
    if len(cashflows) < 2:
        return 0.0

    def npv(rate: float) -> float:
        start = dates[0]
        return sum(
            cf / ((1 + rate) ** ((date - start).days / 365.0))
            for cf, date in zip(cashflows, dates)
        )

    lo, hi = -0.99, 10.0
    flo, fhi = npv(lo), npv(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)) or flo * fhi > 0:
        return 0.0

    for _ in range(200):
        mid = (lo + hi) / 2
        fmid = npv(mid)
        if abs(fmid) < 1e-7:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return (lo + hi) / 2


def run_strategy(df_raw: pd.DataFrame, config: dict) -> dict:
    price_col = config["PRICE_COL"]
    vix_col = config["VIX_COL"]
    df = mark_fallback_days(calculate_rmdd(df_raw, price_col, config["RMDD_WINDOW"]))
    df = df.loc[config["START_DATE"] : config["END_DATE"]].copy()
    df["Prev_RMDD"] = df["RMDD"].shift(1)
    df["Cumulative_Return"] = (df[price_col] / df[price_col].iloc[0] - 1) * 100

    base_amount = config["BASE_AMOUNT"]
    shares_strat = 0.0
    shares_dca = 0.0
    cost_strat = 0.0
    cost_dca = 0.0
    max_monthly_cost = 0.0
    cf_strat: list[tuple[pd.Timestamp, float]] = []
    cf_dca: list[tuple[pd.Timestamp, float]] = []
    buys: list[dict] = []
    curve: list[dict] = []

    for period, group in df.groupby(df.index.to_period("M")):
        std_triggered = [False] * len(config["LEVEL_CONFIG"])
        sniper_idx = 0
        month_has_buy = False
        month_cost = 0.0
        seasonal_factor = config["SEASONAL_FACTORS"].get(period.month, 1.0)
        dca_idx = config["REGULAR_IDX_DCA_PURE"] if len(group) > config["REGULAR_IDX_DCA_PURE"] else len(group) - 1
        dca_date = group.index[dca_idx]

        for i, (date, row) in enumerate(group.iterrows()):
            price = row[price_col]
            vix = row[vix_col]
            rmdd = row["RMDD"]
            if pd.isna(price) or pd.isna(vix):
                continue

            if date == dca_date:
                shares_dca += base_amount / price
                cost_dca += base_amount
                cf_dca.append((date, -base_amount))

            amount = 0.0
            notes: list[str] = []
            is_sniper = False

            active_std_indices = [
                idx
                for idx, setting in enumerate(config["LEVEL_CONFIG"])
                if rmdd <= setting["threshold"] and not std_triggered[idx]
            ]
            newly_crossed_std_indices = [
                idx
                for idx in active_std_indices
                if pd.isna(row["Prev_RMDD"]) or row["Prev_RMDD"] > config["LEVEL_CONFIG"][idx]["threshold"]
            ]

            if newly_crossed_std_indices:
                trigger_std_indices = newly_crossed_std_indices
            elif i == 0 and active_std_indices:
                trigger_std_indices = [max(active_std_indices)]
            else:
                trigger_std_indices = []

            if trigger_std_indices:
                deepest_triggered_idx = max(trigger_std_indices)
                for idx in trigger_std_indices:
                    setting = config["LEVEL_CONFIG"][idx]
                    add = base_amount * setting["units"]
                    amount += add
                    notes.append(f"{setting['desc']} ({setting['units']}u)")
                for handled_idx in range(deepest_triggered_idx + 1):
                    std_triggered[handled_idx] = True
                month_has_buy = True

            if vix > config["VIX_PANIC_THRESHOLD"] and sniper_idx < len(config["SNIPER_CONFIG"]):
                setting = config["SNIPER_CONFIG"][sniper_idx]
                if rmdd <= setting["threshold"]:
                    add = base_amount * setting["units"]
                    amount += add
                    sniper_idx += 1
                    month_has_buy = True
                    is_sniper = True
                    notes.append(f"SNIPER ({setting['units']}u)")

                    while sniper_idx < len(config["SNIPER_CONFIG"]):
                        next_setting = config["SNIPER_CONFIG"][sniper_idx]
                        if rmdd <= next_setting["threshold"]:
                            add = base_amount * next_setting["units"]
                            amount += add
                            sniper_idx += 1
                            notes.append(f"SNIPER GAP ({next_setting['units']}u)")
                        else:
                            break

            if bool(row["Fallback_Day"]) and not month_has_buy:
                amount += base_amount * seasonal_factor
                month_has_buy = True
                note = f"Fallback Buy ({seasonal_factor}u)"
                if seasonal_factor != 1.0:
                    note += " [Oct Boost]"
                notes.append(note)

            if amount > 0:
                shares_strat += amount / price
                cost_strat += amount
                cf_strat.append((date, -amount))
                month_cost += amount
                buy_type = "Fallback Buy" if any(n.startswith("Fallback") for n in notes) else ("Sniper Shot" if is_sniper else "STD Level Buy")
                buys.append(
                    {
                        "Date": date,
                        "Type": buy_type,
                        "Notes": " + ".join(notes),
                        "Amount": amount,
                        "Price": price,
                        "RMDD": rmdd,
                        "VIX": vix,
                        "Strat_Val": shares_strat * price,
                    }
                )

            curve.append(
                {
                    "Date": date,
                    "Strat_Val": shares_strat * price,
                    "DCA_Val": shares_dca * price,
                    "SOX": price,
                    "VIX": vix,
                    "RMDD": rmdd,
                    "Cum_Ret": row["Cumulative_Return"],
                }
            )

        max_monthly_cost = max(max_monthly_cost, month_cost)

    final_date = df.index[-1]
    final_price = df[price_col].iloc[-1]
    final_val_strat = shares_strat * final_price
    final_val_dca = shares_dca * final_price
    xirr_strat = xirr([x[1] for x in cf_strat] + [final_val_strat], [x[0] for x in cf_strat] + [final_date]) * 100
    xirr_dca = xirr([x[1] for x in cf_dca] + [final_val_dca], [x[0] for x in cf_dca] + [final_date]) * 100

    return {
        "data": df,
        "buys": pd.DataFrame(buys),
        "curve": pd.DataFrame(curve).set_index("Date"),
        "metrics": {
            "final_date": final_date,
            "final_price": final_price,
            "cost_strat": cost_strat,
            "cost_dca": cost_dca,
            "final_val_strat": final_val_strat,
            "final_val_dca": final_val_dca,
            "xirr_strat": xirr_strat,
            "xirr_dca": xirr_dca,
            "xirr_diff": xirr_strat - xirr_dca,
            "max_monthly_cost": max_monthly_cost,
        },
    }


def js_list(values) -> list:
    out = []
    for value in values:
        if isinstance(value, (pd.Timestamp,)):
            out.append(value.strftime("%Y-%m-%d"))
        elif pd.isna(value):
            out.append(None)
        elif isinstance(value, (np.floating, float)):
            out.append(float(value))
        elif isinstance(value, (np.integer, int)):
            out.append(int(value))
        else:
            out.append(value)
    return out


def build_market_rows(df_raw: pd.DataFrame, config: dict) -> list[dict]:
    df = mark_fallback_days(calculate_rmdd(df_raw, config["PRICE_COL"], config["RMDD_WINDOW"]))
    df = df.loc[pd.Timestamp(config["CHART_START_DATE"]) :].copy()
    df["Prev_RMDD"] = df["RMDD"].shift(1)
    rows: list[dict] = []
    for date, row in df.iterrows():
        price = row[config["PRICE_COL"]]
        vix = row[config["VIX_COL"]]
        rmdd = row["RMDD"]
        if pd.isna(price) or pd.isna(vix) or pd.isna(rmdd):
            continue
        rows.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Close": float(price),
                "VIX": float(vix),
                "RMDD": float(rmdd),
                "PrevRMDD": None if pd.isna(row["Prev_RMDD"]) else float(row["Prev_RMDD"]),
                "FallbackDay": bool(row["Fallback_Day"]),
            }
        )
    return rows


def build_client_config(config: dict) -> dict:
    return {
        "startDate": config["START_DATE"],
        "endDate": config["END_DATE"],
        "baseAmount": config["BASE_AMOUNT"],
        "vixPanicThreshold": config["VIX_PANIC_THRESHOLD"],
        "regularIdxDcaPure": config["REGULAR_IDX_DCA_PURE"],
        "levelConfig": config["LEVEL_CONFIG"],
        "sniperConfig": config["SNIPER_CONFIG"],
        "seasonalFactors": config["SEASONAL_FACTORS"],
    }


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the SOX Apex Predator dashboard.")
    parser.add_argument(
        "--start-date",
        default=BACKTEST_START_DATE,
        help="Backtest start date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=BACKTEST_END_DATE,
        help='Backtest end date, YYYY-MM-DD, or "auto" for the latest available trading day.',
    )
    return parser.parse_args()


def fmt_units(value: float) -> str:
    return f"{value:g} 份"


def build_rule_table(config: dict) -> str:
    base = config["BASE_AMOUNT"]
    body_rows: list[str] = []

    def add_group(
        rmdd: str,
        normal: list[tuple[str, str]],
        panic: list[tuple[str, str]],
        note: str,
        note_rowspan: int | None = None,
        suppress_note: bool = False,
    ) -> None:
        row_count = max(len(normal), len(panic))
        normal_spans_group = len(normal) == 1 and row_count > 1
        panic_spans_group = len(panic) == 1 and row_count > 1

        for index in range(row_count):
            cells = ["<tr>"]
            if index == 0:
                cells.append(f'<th rowspan="{row_count}">{escape(rmdd)}</th>')
            if normal_spans_group:
                if index == 0:
                    cells.append(
                        f'<td class="rule-cell grouped-cell" rowspan="{row_count}">'
                        f'{escape(normal[0][0])}</td>'
                    )
                    cells.append(
                        f'<td class="amount-cell grouped-cell" rowspan="{row_count}">'
                        f'{escape(normal[0][1])}</td>'
                    )
            else:
                normal_item = normal[index] if index < len(normal) else ("", "")
                cells.append(f'<td class="rule-cell">{escape(normal_item[0])}</td>')
                cells.append(f'<td class="amount-cell">{escape(normal_item[1])}</td>')

            if panic_spans_group:
                if index == 0:
                    cells.append(
                        f'<td class="rule-cell grouped-cell" rowspan="{row_count}">'
                        f'{escape(panic[0][0])}</td>'
                    )
                    cells.append(
                        f'<td class="amount-cell grouped-cell" rowspan="{row_count}">'
                        f'{escape(panic[0][1])}</td>'
                    )
            else:
                panic_item = panic[index] if index < len(panic) else ("", "")
                cells.append(f'<td class="rule-cell">{escape(panic_item[0])}</td>')
                cells.append(f'<td class="amount-cell">{escape(panic_item[1])}</td>')
            if index == 0 and not suppress_note:
                span = note_rowspan or row_count
                cells.append(f'<td class="note-cell" rowspan="{span}">{escape(note)}</td>')
            cells.append("</tr>")
            body_rows.append("".join(cells))

    add_group(
        "月底仍未觸發任何加碼",
        [(f"fallback buy {fmt_units(1.0)}", money(base))],
        [(f"fallback buy {fmt_units(1.0)}", money(base))],
        "每月最多一次",
    )
    add_group(
        "10 月月底仍未觸發任何加碼",
        [(f"fallback buy {fmt_units(1.5)}", money(base * 1.5))],
        [(f"fallback buy {fmt_units(1.5)}", money(base * 1.5))],
        "10 月保底為 1.5 份。",
    )

    thresholds = sorted(
        {item["threshold"] for item in config["LEVEL_CONFIG"]}
        | {item["threshold"] for item in config["SNIPER_CONFIG"]},
        reverse=True,
    )
    std_by_threshold = {item["threshold"]: item for item in config["LEVEL_CONFIG"]}
    sniper_by_threshold = {item["threshold"]: item for item in config["SNIPER_CONFIG"]}
    threshold_groups = []

    for threshold in thresholds:
        std = std_by_threshold.get(threshold)
        sniper = sniper_by_threshold.get(threshold)
        normal: list[tuple[str, str]] = []
        panic: list[tuple[str, str]] = []
        note_parts: list[str] = []

        if std:
            std_text = f"{std['desc']} {fmt_units(std['units'])}"
            std_money = money(base * std["units"])
            normal.append((std_text, std_money))
            panic.append((std_text, std_money))
            note_parts.append("新跌破可連續觸發；跨月延續只觸發當下最深層。另外，由於季節效應，若訊號觸發月份為 10 月，則 Lv1 加碼份數為 1.5 份。")

        if sniper:
            sniper_text = f"{sniper['desc']} {fmt_units(sniper['units'])}"
            sniper_money = money(base * sniper["units"])
            panic.append((sniper_text, sniper_money))
            note_parts.append("VIX > 32 才啟動；跳空大跌可連續觸發 gap sniper")

        normal_total = std["units"] if std else 0.0
        panic_total = normal_total + (sniper["units"] if sniper else 0.0)
        if std and sniper:
            panic.append((f"合計 {fmt_units(panic_total)}", money(base * panic_total)))

        group = {
            "rmdd": f"RMDD <= {threshold * 100:g}%",
            "normal": normal or [("無", "—")],
            "panic": panic or [("無", "—")],
            "note": "；".join(dict.fromkeys(note_parts)),
            "merge_sniper_note": bool(std and sniper),
        }
        group["row_count"] = max(len(group["normal"]), len(group["panic"]))
        threshold_groups.append(group)

    sniper_note = "新跌破可連續觸發；跨月延續只觸發當下最深層。另外，由於季節效應，若訊號觸發月份為 10 月，則 Lv1 加碼份數為 1.5 份。另外，VIX > 32 才啟動；跳空大跌可連續觸發 gap sniper"
    sniper_note_rowspan = sum(
        group["row_count"] for group in threshold_groups if group["merge_sniper_note"]
    )
    sniper_note_rendered = False

    for group in threshold_groups:
        if group["merge_sniper_note"]:
            add_group(
                group["rmdd"],
                group["normal"],
                group["panic"],
                sniper_note,
                note_rowspan=sniper_note_rowspan if not sniper_note_rendered else None,
                suppress_note=sniper_note_rendered,
            )
            sniper_note_rendered = True
        else:
            add_group(group["rmdd"], group["normal"], group["panic"], group["note"])

    body = "\n".join(body_rows)
    return f"""
  <section class="rules">
    <h2>加碼規則矩陣</h2>
    <p id="ruleUnitHelp" class="rule-help">RMDD 是縱軸，VIX 是橫軸；表格列的是「該條件首次觸發時新增投入的份數」。1 份 = {money(base)}。</p>
    <div class="rule-scroll">
      <table class="rule-table">
        <colgroup>
          <col class="rmdd-col">
          <col class="rule-col">
          <col class="amount-col">
          <col class="rule-col">
          <col class="amount-col">
          <col class="note-col">
        </colgroup>
        <thead>
          <tr class="rule-group-head">
            <th rowspan="2">RMDD 條件</th>
            <th colspan="2">VIX &lt;= {config['VIX_PANIC_THRESHOLD']}<br><span>正常 / 非恐慌</span></th>
            <th colspan="2">VIX &gt; {config['VIX_PANIC_THRESHOLD']}<br><span>恐慌區</span></th>
            <th rowspan="2">備註</th>
          </tr>
          <tr class="rule-subhead">
            <th>規則</th>
            <th>金額</th>
            <th>規則</th>
            <th>金額</th>
          </tr>
        </thead>
        <tbody id="ruleTableBody">{body}</tbody>
      </table>
    </div>
  </section>
"""


def make_marker_trace(buys: pd.DataFrame, buy_type: str, color: str, symbol: str, size: int) -> dict:
    sub = buys[buys["Type"] == buy_type] if not buys.empty else pd.DataFrame()
    return {
        "type": "scatter",
        "mode": "markers",
        "name": buy_type,
        "x": js_list(sub["Date"]) if not sub.empty else [],
        "y": js_list(sub["Strat_Val"]) if not sub.empty else [],
        "customdata": js_list(sub["Notes"]) if not sub.empty else [],
        "marker": {"color": color, "symbol": symbol, "size": size, "line": {"width": 1, "color": "white"}},
        "hovertemplate": "%{x}<br>%{customdata}<br>Portfolio: $%{y:,.0f}<extra></extra>",
        "xaxis": "x",
        "yaxis": "y",
    }


def build_interactive_script(market_rows: list[dict], client_config: dict, tickers: list[str], source_label: str) -> str:
    script = r"""
  <script>
    const marketRows = __MARKET_ROWS__;
    const strategyConfig = __CLIENT_CONFIG__;
    const defaultBaseAmount = strategyConfig.baseAmount;
    const sourceLabel = __SOURCE_LABEL__;
    const sourceTickers = __TICKERS__;

    const moneyFmt = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
    const intFmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
    const signedIntFmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0, signDisplay: 'always' });

    function money(value) {
      return moneyFmt.format(value || 0);
    }

    function pct(value) {
      return `${Number(value || 0).toFixed(2)}%`;
    }

    function signedPct(value) {
      const numeric = Number(value || 0);
      return `${numeric > 0 ? '+' : ''}${numeric.toFixed(2)}%`;
    }

    function htmlEscape(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    function formatUnits(value) {
      return `${Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 })} 份`;
    }

    function dateObj(dateText) {
      return new Date(`${dateText}T00:00:00`);
    }

    function yearsBetween(start, end) {
      return (dateObj(end) - dateObj(start)) / (365.25 * 24 * 60 * 60 * 1000);
    }

    function xnpv(rate, cashflows) {
      const start = dateObj(cashflows[0].date);
      return cashflows.reduce((sum, cf) => {
        const days = (dateObj(cf.date) - start) / (24 * 60 * 60 * 1000);
        return sum + cf.value / Math.pow(1 + rate, days / 365.25);
      }, 0);
    }

    function xirr(cashflows) {
      if (!cashflows.length || !cashflows.some(cf => cf.value < 0) || !cashflows.some(cf => cf.value > 0)) return NaN;
      let lo = -0.9999;
      let hi = 1.0;
      let flo = xnpv(lo, cashflows);
      let fhi = xnpv(hi, cashflows);
      for (let i = 0; i < 80 && flo * fhi > 0; i += 1) {
        hi *= 2;
        fhi = xnpv(hi, cashflows);
        if (hi > 10000) break;
      }
      if (!Number.isFinite(flo) || !Number.isFinite(fhi) || flo * fhi > 0) return NaN;
      for (let i = 0; i < 200; i += 1) {
        const mid = (lo + hi) / 2;
        const fmid = xnpv(mid, cashflows);
        if (Math.abs(fmid) < 1e-6) return mid * 100;
        if (flo * fmid <= 0) {
          hi = mid;
          fhi = fmid;
        } else {
          lo = mid;
          flo = fmid;
        }
      }
      return ((lo + hi) / 2) * 100;
    }

    function runStrategy(startDate, endDate) {
      const rows = marketRows
        .filter(row => row.Date >= startDate && row.Date <= endDate)
        .map((row, index, arr) => ({
          ...row,
          CumRet: ((row.Close / arr[0].Close) - 1) * 100,
        }));

      if (!rows.length) throw new Error('指定期間內沒有資料。');

      const groups = new Map();
      rows.forEach(row => {
        const month = row.Date.slice(0, 7);
        if (!groups.has(month)) groups.set(month, []);
        groups.get(month).push(row);
      });

      let sharesStrat = 0;
      let sharesDca = 0;
      let costStrat = 0;
      let costDca = 0;
      let maxMonthlyCost = 0;
      const cfStrat = [];
      const cfDca = [];
      const buys = [];
      const curve = [];

      for (const [month, group] of groups) {
        const stdTriggered = strategyConfig.levelConfig.map(() => false);
        let sniperIdx = 0;
        let monthHasBuy = false;
        let monthCost = 0;
        const monthNumber = Number(month.slice(5, 7));
        const seasonalFactor = strategyConfig.seasonalFactors[String(monthNumber)] || strategyConfig.seasonalFactors[monthNumber] || 1.0;
        const dcaIdx = group.length > strategyConfig.regularIdxDcaPure ? strategyConfig.regularIdxDcaPure : group.length - 1;
        const dcaDate = group[dcaIdx].Date;

        group.forEach((row, index) => {
          const price = row.Close;
          const vix = row.VIX;
          const rmdd = row.RMDD;

          if (row.Date === dcaDate) {
            sharesDca += strategyConfig.baseAmount / price;
            costDca += strategyConfig.baseAmount;
            cfDca.push({ date: row.Date, value: -strategyConfig.baseAmount });
          }

          let amount = 0;
          const notes = [];
          let isSniper = false;

          const activeStdIndices = strategyConfig.levelConfig
            .map((setting, idx) => ({ setting, idx }))
            .filter(({ setting, idx }) => rmdd <= setting.threshold && !stdTriggered[idx])
            .map(({ idx }) => idx);
          const newlyCrossedStdIndices = activeStdIndices.filter(idx => (
            row.PrevRMDD === null
            || row.PrevRMDD === undefined
            || Number.isNaN(row.PrevRMDD)
            || row.PrevRMDD > strategyConfig.levelConfig[idx].threshold
          ));
          const triggerStdIndices = newlyCrossedStdIndices.length > 0
            ? newlyCrossedStdIndices
            : (index === 0 && activeStdIndices.length > 0 ? [Math.max(...activeStdIndices)] : []);

          if (triggerStdIndices.length > 0) {
            const deepestTriggeredIdx = Math.max(...triggerStdIndices);
            triggerStdIndices.forEach(idx => {
              const setting = strategyConfig.levelConfig[idx];
              const add = strategyConfig.baseAmount * setting.units;
              amount += add;
              notes.push(`${setting.desc} (${setting.units}u)`);
            });
            for (let handledIdx = 0; handledIdx <= deepestTriggeredIdx; handledIdx += 1) {
              stdTriggered[handledIdx] = true;
            }
            monthHasBuy = true;
          }

          if (vix > strategyConfig.vixPanicThreshold && sniperIdx < strategyConfig.sniperConfig.length) {
            const setting = strategyConfig.sniperConfig[sniperIdx];
            if (rmdd <= setting.threshold) {
              const add = strategyConfig.baseAmount * setting.units;
              amount += add;
              sniperIdx += 1;
              monthHasBuy = true;
              isSniper = true;
              notes.push(`SNIPER (${setting.units}u)`);

              while (sniperIdx < strategyConfig.sniperConfig.length) {
                const nextSetting = strategyConfig.sniperConfig[sniperIdx];
                if (rmdd <= nextSetting.threshold) {
                  amount += strategyConfig.baseAmount * nextSetting.units;
                  sniperIdx += 1;
                  notes.push(`SNIPER GAP (${nextSetting.units}u)`);
                } else {
                  break;
                }
              }
            }
          }

          if (row.FallbackDay && !monthHasBuy) {
            amount += strategyConfig.baseAmount * seasonalFactor;
            monthHasBuy = true;
            let note = `Fallback Buy (${seasonalFactor}u)`;
            if (seasonalFactor !== 1.0) note += ' [Oct Boost]';
            notes.push(note);
          }

          if (amount > 0) {
            sharesStrat += amount / price;
            costStrat += amount;
            cfStrat.push({ date: row.Date, value: -amount });
            monthCost += amount;
            const buyType = notes.some(note => note.startsWith('Fallback'))
              ? 'Fallback Buy'
              : (isSniper ? 'Sniper Shot' : 'STD Level Buy');
            buys.push({
              Date: row.Date,
              Type: buyType,
              Notes: notes.join(' + '),
              Amount: amount,
              Price: price,
              RMDD: rmdd,
              VIX: vix,
              Strat_Val: sharesStrat * price,
            });
          }

          curve.push({
            Date: row.Date,
            Strat_Val: sharesStrat * price,
            DCA_Val: sharesDca * price,
            SOX: price,
            VIX: vix,
            RMDD: rmdd,
            Cum_Ret: row.CumRet,
          });
        });

        maxMonthlyCost = Math.max(maxMonthlyCost, monthCost);
      }

      const finalRow = rows[rows.length - 1];
      const finalPrice = finalRow.Close;
      const finalValStrat = sharesStrat * finalPrice;
      const finalValDca = sharesDca * finalPrice;
      const xirrStrat = xirr([...cfStrat, { date: finalRow.Date, value: finalValStrat }]);
      const xirrDca = xirr([...cfDca, { date: finalRow.Date, value: finalValDca }]);

      return {
        rows,
        curve,
        buys,
        metrics: {
          startDate: rows[0].Date,
          finalDate: finalRow.Date,
          finalPrice,
          costStrat,
          costDca,
          finalValStrat,
          finalValDca,
          xirrStrat,
          xirrDca,
          xirrDiff: xirrStrat - xirrDca,
          maxMonthlyCost,
        },
      };
    }

    function diffClass(value) {
      if (value > 0) return 'diff-positive';
      if (value < 0) return 'diff-negative';
      return 'diff-neutral';
    }

    function renderCards(result) {
      const latest = result.curve[result.curve.length - 1];
      document.querySelector('.cards').innerHTML = `
        <div class="card"><span>最新 SOX</span><b>${latest.SOX.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</b></div>
        <div class="card"><span>最新 VIX</span><b>${latest.VIX.toFixed(2)}</b></div>
        <div class="card"><span>目前 RMDD</span><b>${(latest.RMDD * 100).toFixed(2)}%</b></div>
      `;
    }

    function readableBuyNote(note) {
      return String(note || '')
        .split(' + ')
        .map(part => {
          const fallbackMatch = part.match(/Fallback Buy \(([-\d.]+)u\)/);
          if (fallbackMatch) return `fallback buy ${formatUnits(Number(fallbackMatch[1]))}`;
          const standardMatch = part.match(/(?:Std )?(Lv\d+) \(([-\d.]+)u\)/);
          if (standardMatch) return `${standardMatch[1]} ${formatUnits(Number(standardMatch[2]))}`;
          const sniperMatch = part.match(/SNIPER(?: GAP)? \(([-\d.]+)u\)/);
          if (sniperMatch) {
            const units = Number(sniperMatch[1]);
            const setting = strategyConfig.sniperConfig.find(item => Math.abs(item.units - units) < 0.0001);
            return `${setting ? setting.desc : 'Sniper Shot'} ${formatUnits(units)}`;
          }
          return part.replaceAll('u', ' 份');
        })
        .join('、');
    }

    function triggeredStandardSet(monthBuys) {
      const handled = new Set();
      monthBuys.forEach(buy => {
        strategyConfig.levelConfig.forEach((setting, idx) => {
          if (String(buy.Notes || '').includes(setting.desc)) {
            for (let handledIdx = 0; handledIdx <= idx; handledIdx += 1) {
              handled.add(strategyConfig.levelConfig[handledIdx].desc);
            }
          }
        });
      });
      return handled;
    }

    function countTriggeredSnipers(monthBuys) {
      return monthBuys.reduce((count, buy) => {
        const matches = String(buy.Notes || '').match(/SNIPER/g);
        return count + (matches ? matches.length : 0);
      }, 0);
    }

    function renderSignalPanels(result) {
      const m = result.metrics;
      const monthKey = m.finalDate.slice(0, 7);
      const monthBuys = result.buys.filter(buy => buy.Date.slice(0, 7) === monthKey);
      const totalAmount = monthBuys.reduce((sum, buy) => sum + buy.Amount, 0);
      const hasBuy = monthBuys.length > 0;
      const buyRows = hasBuy
        ? monthBuys.map(buy => `
            <li>
              <b>${htmlEscape(buy.Date)}</b>
              <span>${htmlEscape(readableBuyNote(buy.Notes))}</span>
              <strong>${money(buy.Amount)}</strong>
            </li>
          `).join('')
        : '<li><span>本月尚無策略買入紀錄。</span><strong>$0</strong></li>';

      const stdTriggered = triggeredStandardSet(monthBuys);
      const nextStd = strategyConfig.levelConfig.find(setting => !stdTriggered.has(setting.desc));
      const sniperCount = countTriggeredSnipers(monthBuys);
      const nextSniper = strategyConfig.sniperConfig[sniperCount];
      const nextStdText = nextStd
        ? `${nextStd.desc}: RMDD <= ${(nextStd.threshold * 100).toFixed(2)}%`
        : '本月標準防線已全部觸發';
      const nextSniperText = nextSniper
        ? `${nextSniper.desc}: VIX > ${strategyConfig.vixPanicThreshold} and RMDD <= ${(nextSniper.threshold * 100).toFixed(2)}%`
        : '本月恐慌狙擊已全部觸發';

      return `
        <div class="signal-grid">
          <article class="signal-card ${hasBuy ? 'active' : 'quiet'}">
            <span class="signal-pill">${hasBuy ? '本月已觸發買入' : '本月尚未觸發買入'}</span>
            <h3>目前建議買入金額：${money(totalAmount)}</h3>
            <ul class="signal-list">${buyRows}</ul>
          </article>
          <article class="signal-card expectation">
            <h3>下一個觸發條件</h3>
            <div class="expectation-lines">
              <p><b>標準加碼：</b>${htmlEscape(nextStdText)}</p>
              <p><b>恐慌狙擊：</b>${htmlEscape(nextSniperText)}</p>
            </div>
          </article>
        </div>
      `;
    }

    function renderPerformance(result) {
      const m = result.metrics;
      const years = yearsBetween(m.startDate, m.finalDate);
      const absoluteReturnStrat = (m.finalValStrat / m.costStrat - 1) * 100;
      const absoluteReturnDca = (m.finalValDca / m.costDca - 1) * 100;
      const annualizedAbsoluteStrat = (Math.pow(m.finalValStrat / m.costStrat, 1 / years) - 1) * 100;
      const annualizedAbsoluteDca = (Math.pow(m.finalValDca / m.costDca, 1 / years) - 1) * 100;
      const totalXirrStrat = (Math.pow(1 + m.xirrStrat / 100, years) - 1) * 100;
      const totalXirrDca = (Math.pow(1 + m.xirrDca / 100, years) - 1) * 100;
      const finalValueDiff = m.finalValStrat - m.finalValDca;
      const costDiff = m.costStrat - m.costDca;
      const maxMonthlyDiff = m.maxMonthlyCost - strategyConfig.baseAmount;
      const annualizedAbsoluteDiff = annualizedAbsoluteStrat - annualizedAbsoluteDca;
      const absoluteReturnDiff = absoluteReturnStrat - absoluteReturnDca;
      const totalXirrDiff = totalXirrStrat - totalXirrDca;
      const fallbackCount = result.buys.filter(buy => buy.Type === 'Fallback Buy').length;
      const standardCount = result.buys.filter(buy => buy.Type === 'STD Level Buy').length;
      const sniperCount = result.buys.filter(buy => buy.Type === 'Sniper Shot').length;
      const monthHasBuy = result.buys.some(buy => buy.Date.slice(0, 7) === m.finalDate.slice(0, 7));
      document.querySelector('.subtitle').textContent = `最新資料日：${m.finalDate}`;
      document.querySelector('#heroFinalValue').textContent = money(m.finalValStrat);
      document.querySelector('#heroCost').textContent = money(m.costStrat);
      document.querySelector('#heroXirr').textContent = pct(m.xirrStrat);
      document.querySelector('#heroBuyCount').textContent = intFmt.format(result.buys.length);

      document.querySelector('.performance').innerHTML = `
        ${renderSignalPanels(result)}
        <div class="section-heading">
          <div>
            <span class="eyebrow">策略績效摘要</span>
            <h2>Apex Predator 與純定期定額</h2>
          </div>
          <div class="period">${m.startDate} 至 ${m.finalDate}<span>約 ${years.toFixed(1)} 年</span></div>
        </div>
        <div class="performance-grid">
          <div class="summary-panel">
            <div class="outcome-grid">
              <article class="outcome-card primary">
                <span>最終資產差額</span>
                <b>${signedIntFmt.format(finalValueDiff)}</b>
                <small>Apex Predator ${money(m.finalValStrat)} / 純 DCA ${money(m.finalValDca)}</small>
              </article>
              <article class="outcome-card">
                <span>年化 XIRR 差距</span>
                <b class="${diffClass(m.xirrDiff)}">${signedPct(m.xirrDiff)}</b>
                <small>${pct(m.xirrStrat)} vs ${pct(m.xirrDca)}</small>
              </article>
              <article class="outcome-card">
                <span>總投入差額</span>
                <b>${signedIntFmt.format(costDiff)}</b>
                <small>${money(m.costStrat)} vs ${money(m.costDca)}</small>
              </article>
              <article class="outcome-card">
                <span>買入頻率</span>
                <b>${(result.buys.length / years).toFixed(1)} 次／年</b>
                <small>累計 ${intFmt.format(result.buys.length)} 次買入</small>
              </article>
            </div>
            <div class="compact-comparison">
              <div class="compact-row compact-head"><span>核心指標</span><strong>Apex Predator</strong><strong>純 DCA</strong><strong>差值</strong></div>
              <div class="compact-row"><span>最終資產</span><b>${money(m.finalValStrat)}</b><b>${money(m.finalValDca)}</b><b class="diff-neutral">${signedIntFmt.format(finalValueDiff)}</b></div>
              <div class="compact-row"><span>總投入</span><b>${money(m.costStrat)}</b><b>${money(m.costDca)}</b><b class="diff-neutral">${signedIntFmt.format(costDiff)}</b></div>
              <div class="compact-row emphasis"><span>年化 XIRR</span><b>${pct(m.xirrStrat)}</b><b>${pct(m.xirrDca)}</b><b class="${diffClass(m.xirrDiff)}">${signedPct(m.xirrDiff)}</b></div>
            </div>
          </div>
          <aside class="trade-summary">
            <div class="trade-title">交易訊號統計</div>
            <div class="trade-total"><div><b>${result.buys.length}</b><span>總買入次數</span></div></div>
            <div class="trade-list cumulative-list">
              <div class="trade-list-head"><span></span><span></span><small>累計</small></div>
              <div><i class="fallback-dot"></i><span>保底買入</span><b>${fallbackCount}</b></div>
              <div><i class="standard-dot"></i><span>標準防線</span><b>${standardCount}</b></div>
              <div><i class="sniper-dot"></i><span>狙擊觸發</span><b>${sniperCount}</b></div>
            </div>
            <p>狙擊條件：VIX &gt; ${strategyConfig.vixPanicThreshold} 且 RMDD 跌破對應防線。</p>
            <div class="annual-section">
              <div class="trade-title">交易訊號統計</div>
              <div class="annual-total"><b>${(result.buys.length / years).toFixed(1)}</b><span>次／年</span></div>
              <div class="trade-list annual-list">
                <div class="trade-list-head"><span></span><span></span><small>年均</small></div>
                <div><i class="fallback-dot"></i><span>保底買入</span><em>${(fallbackCount / years).toFixed(1)}</em></div>
                <div><i class="standard-dot"></i><span>標準防線</span><em>${(standardCount / years).toFixed(1)}</em></div>
                <div><i class="sniper-dot"></i><span>狙擊觸發</span><em>${(sniperCount / years).toFixed(1)}</em></div>
              </div>
            </div>
          </aside>
        </div>
      `;

      document.querySelector('.detailed-performance').innerHTML = `
        <div class="section-heading">
          <div>
            <span class="eyebrow">完整績效指標</span>
            <h2>Apex Predator 與純定期定額完整比較</h2>
          </div>
          <div class="period">${m.startDate} 至 ${m.finalDate}<span>約 ${years.toFixed(1)} 年</span></div>
        </div>
        <div class="comparison">
          <div class="comparison-head">
            <span>績效指標</span>
            <strong>Apex Predator</strong>
            <strong>純 DCA</strong>
            <strong>%／差值</strong>
          </div>
          <div class="comparison-row">
            <span>總投入成本</span><b>${money(m.costStrat)}</b><b>${money(m.costDca)}</b><b class="diff-neutral">${signedIntFmt.format(costDiff)}</b>
          </div>
          <div class="comparison-row">
            <span>最終資產價值</span><b>${money(m.finalValStrat)}</b><b>${money(m.finalValDca)}</b><b class="diff-neutral">${signedIntFmt.format(finalValueDiff)}</b>
          </div>
          <div class="comparison-row">
            <span>單月最高投入</span><b>${money(m.maxMonthlyCost)}</b><b>${money(strategyConfig.baseAmount)}</b><b class="diff-neutral">${signedIntFmt.format(maxMonthlyDiff)}</b>
          </div>
          <div class="comparison-row return-divider">
            <span>絕對績效</span><b>${pct(absoluteReturnStrat)}</b><b>${pct(absoluteReturnDca)}</b><b class="${diffClass(absoluteReturnDiff)}">${signedPct(absoluteReturnDiff)}</b>
          </div>
          <div class="comparison-row">
            <span>平均年化絕對報酬</span><b>${pct(annualizedAbsoluteStrat)}</b><b>${pct(annualizedAbsoluteDca)}</b><b class="${diffClass(annualizedAbsoluteDiff)}">${signedPct(annualizedAbsoluteDiff)}</b>
          </div>
          <div class="comparison-row">
            <span>總體 XIRR</span><b>${pct(totalXirrStrat)}</b><b>${pct(totalXirrDca)}</b><b class="${diffClass(totalXirrDiff)}">${signedPct(totalXirrDiff)}</b>
          </div>
          <div class="comparison-row emphasis">
            <span>年化報酬率 XIRR</span><b>${pct(m.xirrStrat)}</b><b>${pct(m.xirrDca)}</b><b class="${diffClass(m.xirrDiff)}">${signedPct(m.xirrDiff)}</b>
          </div>
          <div class="comparison-footnote">差值為 Apex Predator − 純 DCA。<br>XIRR（擴展內部報酬率）是專門用來計算「不定期、多筆現金流」的年化報酬率指標。<br>總體 XIRR 為年化 XIRR 按完整回測期間複利換算。</div>
        </div>
      `;
    }

    function renderRuleTable() {
      const base = strategyConfig.baseAmount;
      const formatUnits = value => `${Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 })} 份`;
      const htmlEscape = value => String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
      const bodyRows = [];

      function addGroup(rmdd, normal, panic, note, options = {}) {
        const rowCount = Math.max(normal.length, panic.length);
        const normalSpansGroup = normal.length === 1 && rowCount > 1;
        const panicSpansGroup = panic.length === 1 && rowCount > 1;
        for (let index = 0; index < rowCount; index += 1) {
          const cells = ['<tr>'];
          if (index === 0) cells.push(`<th rowspan="${rowCount}">${htmlEscape(rmdd)}</th>`);
          if (normalSpansGroup) {
            if (index === 0) {
              cells.push(`<td class="rule-cell grouped-cell" rowspan="${rowCount}">${htmlEscape(normal[0][0])}</td>`);
              cells.push(`<td class="amount-cell grouped-cell" rowspan="${rowCount}">${htmlEscape(normal[0][1])}</td>`);
            }
          } else {
            const item = normal[index] || ['', ''];
            cells.push(`<td class="rule-cell">${htmlEscape(item[0])}</td>`);
            cells.push(`<td class="amount-cell">${htmlEscape(item[1])}</td>`);
          }
          if (panicSpansGroup) {
            if (index === 0) {
              cells.push(`<td class="rule-cell grouped-cell" rowspan="${rowCount}">${htmlEscape(panic[0][0])}</td>`);
              cells.push(`<td class="amount-cell grouped-cell" rowspan="${rowCount}">${htmlEscape(panic[0][1])}</td>`);
            }
          } else {
            const item = panic[index] || ['', ''];
            cells.push(`<td class="rule-cell">${htmlEscape(item[0])}</td>`);
            cells.push(`<td class="amount-cell">${htmlEscape(item[1])}</td>`);
          }
          if (index === 0 && !options.suppressNote) {
            cells.push(`<td class="note-cell" rowspan="${options.noteRowspan || rowCount}">${htmlEscape(note)}</td>`);
          }
          cells.push('</tr>');
          bodyRows.push(cells.join(''));
        }
      }

      addGroup('月底仍未觸發任何加碼', [['fallback buy 1 份', money(base)]], [['fallback buy 1 份', money(base)]], '每月最多一次');
      addGroup('10 月月底仍未觸發任何加碼', [['fallback buy 1.5 份', money(base * 1.5)]], [['fallback buy 1.5 份', money(base * 1.5)]], '10 月保底為 1.5 份。');

      const thresholds = [...new Set([
        ...strategyConfig.levelConfig.map(item => item.threshold),
        ...strategyConfig.sniperConfig.map(item => item.threshold),
      ])].sort((a, b) => b - a);
      const levelMap = new Map(strategyConfig.levelConfig.map(item => [item.threshold, item]));
      const sniperMap = new Map(strategyConfig.sniperConfig.map(item => [item.threshold, item]));
      const thresholdGroups = [];

      thresholds.forEach(threshold => {
        const std = levelMap.get(threshold);
        const sniper = sniperMap.get(threshold);
        const normal = [];
        const panic = [];
        const notes = [];
        if (std) {
          const rule = `${std.desc} ${formatUnits(std.units)}`;
          normal.push([rule, money(base * std.units)]);
          panic.push([rule, money(base * std.units)]);
          notes.push('新跌破可連續觸發；跨月延續只觸發當下最深層。另外，由於季節效應，若訊號觸發月份為 10 月，則 Lv1 加碼份數為 1.5 份。');
        }
        if (sniper) {
          panic.push([`${sniper.desc} ${formatUnits(sniper.units)}`, money(base * sniper.units)]);
          notes.push('VIX > 32 才啟動；跳空大跌可連續觸發 gap sniper');
        }
        if (std && sniper) {
          panic.push([`合計 ${formatUnits(std.units + sniper.units)}`, money(base * (std.units + sniper.units))]);
        }
        const group = {
          rmdd: `RMDD <= ${(threshold * 100).toFixed(1).replace('.0', '')}%`,
          normal: normal.length ? normal : [['無', '—']],
          panic: panic.length ? panic : [['無', '—']],
          note: [...new Set(notes)].join('；'),
          mergeSniperNote: Boolean(std && sniper),
        };
        group.rowCount = Math.max(group.normal.length, group.panic.length);
        thresholdGroups.push(group);
      });

      const sniperNote = '新跌破可連續觸發；跨月延續只觸發當下最深層。另外，由於季節效應，若訊號觸發月份為 10 月，則 Lv1 加碼份數為 1.5 份。另外，VIX > 32 才啟動；跳空大跌可連續觸發 gap sniper';
      const sniperNoteRowspan = thresholdGroups
        .filter(group => group.mergeSniperNote)
        .reduce((total, group) => total + group.rowCount, 0);
      let sniperNoteRendered = false;

      thresholdGroups.forEach(group => {
        if (group.mergeSniperNote) {
          addGroup(group.rmdd, group.normal, group.panic, sniperNote, {
            noteRowspan: sniperNoteRendered ? undefined : sniperNoteRowspan,
            suppressNote: sniperNoteRendered,
          });
          sniperNoteRendered = true;
        } else {
          addGroup(group.rmdd, group.normal, group.panic, group.note);
        }
      });

      document.getElementById('ruleUnitHelp').textContent =
        `RMDD 是縱軸，VIX 是橫軸；表格列的是「該條件首次觸發時新增投入的份數」。1 份 = ${money(base)}。`;
      document.getElementById('ruleTableBody').innerHTML = bodyRows.join('');
    }

    function markerTrace(result, type, color, symbol, size) {
      const buys = result.buys.filter(buy => buy.Type === type);
      return {
        type: 'scatter',
        mode: 'markers',
        name: type,
        x: buys.map(buy => buy.Date),
        y: buys.map(buy => buy.Strat_Val),
        customdata: buys.map(buy => buy.Notes),
        marker: { color, symbol, size, line: { width: 1, color: 'white' } },
        hovertemplate: '%{x}<br>%{customdata}<br>Portfolio: $%{y:,.0f}<extra></extra>',
        xaxis: 'x',
        yaxis: 'y',
      };
    }

    function chartPayload(result) {
      const curve = result.curve;
      const endYear = Number(result.metrics.finalDate.slice(0, 4)) + 1;
      const chartRange = [result.metrics.startDate, `${endYear}-12-31`];
      const traces = [
        { type: 'scatter', mode: 'lines', name: 'Apex Predator', x: curve.map(r => r.Date), y: curve.map(r => r.Strat_Val), line: { color: '#62e6ff', width: 3.4 }, hovertemplate: '%{x}<br>Apex Predator: $%{y:,.0f}<extra></extra>', xaxis: 'x', yaxis: 'y' },
        { type: 'scatter', mode: 'lines', name: 'Pure DCA', x: curve.map(r => r.Date), y: curve.map(r => r.DCA_Val), line: { color: '#9db0c4', width: 2, dash: 'dash' }, hovertemplate: '%{x}<br>Pure DCA: $%{y:,.0f}<extra></extra>', xaxis: 'x', yaxis: 'y' },
        markerTrace(result, 'Fallback Buy', '#46e6a8', 'circle', 8),
        markerTrace(result, 'STD Level Buy', '#ff6b7a', 'triangle-up', 10),
        markerTrace(result, 'Sniper Shot', '#b38cff', 'star', 14),
        { type: 'scatter', mode: 'lines', name: 'VIX', x: curve.map(r => r.Date), y: curve.map(r => r.VIX), line: { color: '#ffd166', width: 1.8 }, hovertemplate: '%{x}<br>VIX: %{y:.2f}<extra></extra>', xaxis: 'x2', yaxis: 'y2' },
        { type: 'scatter', mode: 'lines', name: 'RMDD', x: curve.map(r => r.Date), y: curve.map(r => r.RMDD * 100), line: { color: '#66a6ff', width: 2.6 }, fill: 'tozeroy', fillcolor: 'rgba(102, 166, 255, 0.16)', hovertemplate: '%{x}<br>RMDD: %{y:.2f}%<extra></extra>', xaxis: 'x3', yaxis: 'y3' },
        { type: 'scatter', mode: 'lines', name: 'SOX Return', x: curve.map(r => r.Date), y: curve.map(r => r.Cum_Ret), line: { color: '#46e6a8', width: 2 }, hovertemplate: '%{x}<br>SOX Return: %{y:.2f}%<extra></extra>', xaxis: 'x4', yaxis: 'y4' },
      ];
      const shapes = [
        { type: 'line', xref: 'paper', x0: 0, x1: 0.83, yref: 'y2', y0: strategyConfig.vixPanicThreshold, y1: strategyConfig.vixPanicThreshold, line: { color: '#b38cff', width: 2, dash: 'dash' } },
        ...strategyConfig.levelConfig.map(setting => ({ type: 'line', xref: 'paper', x0: 0, x1: 0.83, yref: 'y3', y0: setting.threshold * 100, y1: setting.threshold * 100, line: { color: 'rgba(199, 215, 232, 0.54)', width: 1, dash: 'dot' } })),
        ...strategyConfig.sniperConfig.map(setting => ({ type: 'line', xref: 'paper', x0: 0, x1: 0.83, yref: 'y3', y0: setting.threshold * 100, y1: setting.threshold * 100, line: { color: '#ff6b7a', width: 1.5, dash: 'solid' } })),
      ];
      const layout = {
        title: { text: '', x: 0.02, font: { size: 28 } },
        paper_bgcolor: '#071019',
        plot_bgcolor: '#0d1a27',
        font: { family: 'Arial, Microsoft JhengHei, Noto Sans TC, sans-serif', color: '#c7d7e8', size: 14 },
        height: 1040,
        margin: { l: 74, r: 230, t: 82, b: 70 },
        hovermode: 'x unified',
        legend: { x: 1.02, y: 0.99, xanchor: 'left', yanchor: 'top', font: { size: 16, color: '#f4fbff' }, bgcolor: 'rgba(13, 26, 39, 0.72)', bordercolor: 'rgba(149, 179, 210, 0.22)', borderwidth: 1 },
        xaxis: { domain: [0, 0.84], anchor: 'y', range: chartRange, showgrid: true, gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)', showticklabels: false },
        yaxis: { domain: [0.60, 1.0], title: 'Portfolio Value ($)', gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)', tickformat: '~s' },
        xaxis2: { domain: [0, 0.84], anchor: 'y2', matches: 'x', showgrid: true, gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)', showticklabels: false },
        yaxis2: { domain: [0.40, 0.54], title: 'VIX', gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)' },
        xaxis3: { domain: [0, 0.84], anchor: 'y3', matches: 'x', showgrid: true, gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)', showticklabels: false },
        yaxis3: { domain: [0.21, 0.35], title: 'RMDD (%)', gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)', ticksuffix: '%' },
        xaxis4: { domain: [0, 0.84], anchor: 'y4', matches: 'x', range: chartRange, showgrid: true, gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)', rangeslider: { visible: true, thickness: 0.06, bgcolor: '#0d1a27', bordercolor: 'rgba(149, 179, 210, 0.28)' } },
        yaxis4: { domain: [0.02, 0.16], title: 'SOX Return (%)', gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)', ticksuffix: '%' },
        shapes,
        annotations: [
          { text: 'Apex Predator Strategy<br>Portfolio Value ($) & Buy Events', xref: 'paper', yref: 'paper', x: 0.42, y: 1.04, showarrow: false, font: { size: 24, color: '#f4fbff' } },
          { text: 'VIX Index', xref: 'paper', yref: 'paper', x: 0.42, y: 0.56, showarrow: false, font: { size: 22, color: '#f4fbff' } },
          { text: 'RMDD Zones', xref: 'paper', yref: 'paper', x: 0.42, y: 0.37, showarrow: false, font: { size: 22, color: '#f4fbff' } },
          { text: 'SOX Cumulative Return', xref: 'paper', yref: 'paper', x: 0.42, y: 0.18, showarrow: false, font: { size: 22, color: '#f4fbff' } },
          { text: 'Panic Threshold', xref: 'paper', yref: 'y2', x: 0.82, y: strategyConfig.vixPanicThreshold + 1.5, showarrow: false, font: { size: 14, color: '#ffd166' } },
        ],
      };
      return { traces, layout };
    }

    function buildParameterData(result) {
      const holdMonths = Array.from({ length: 12 }, (_, index) => index + 1);
      const holdDays = holdMonths.map(month => month * 21);
      const buckets = [];
      for (let value = -10; value >= -62.5; value -= 2.5) {
        buckets.push(Number(value.toFixed(1)));
      }
      const bucketLabels = buckets.map(value => `${value.toFixed(1)}%`);
      const rows = result.rows;
      const signals = [];

      rows.forEach((row, index) => {
        if (row.RMDD <= -0.10) {
          const bucket = Number((Math.ceil((row.RMDD * 100) / 2.5 - 0.0001) * 2.5).toFixed(1));
          if (buckets.includes(bucket)) {
            signals.push({ index, bucket });
          }
        }
      });

      const counts = buckets.map(bucket => signals.filter(signal => signal.bucket === bucket).length);
      const totalCount = counts.reduce((sum, value) => sum + value, 0);
      const probabilities = counts.map(count => totalCount ? (count / totalCount) * 100 : 0);
      const z = holdDays.map(days => (
        buckets.map(bucket => {
          const values = signals
            .filter(signal => signal.bucket === bucket && signal.index + days < rows.length)
            .map(signal => ((rows[signal.index + days].Close / rows[signal.index].Close) - 1) * 100)
            .filter(value => Number.isFinite(value));
          if (!values.length) return null;
          return values.reduce((sum, value) => sum + value, 0) / values.length;
        })
      ));

      return {
        holdLabels: holdMonths.map(month => `${month}M`),
        bucketLabels,
        buckets,
        counts,
        probabilities,
        totalCount,
        z,
        zForSurface: z.map(row => row.map(value => value === null ? 0 : value)),
      };
    }

    function renderParameterCharts(result) {
      const data = buildParameterData(result);
      const plateauColorscale = [
        [0.00, '#b8542f'],
        [0.06, '#df8b52'],
        [0.11, '#fff6b0'],
        [0.28, '#edf5a6'],
        [0.48, '#d4ec8f'],
        [0.68, '#9ccd73'],
        [0.84, '#669d5c'],
        [1.00, '#315f3b'],
      ];
      const heatmapTextColors = data.z.map(row => row.map(value => (
        value !== null && value >= 45 ? '#ffffff' : '#4b5563'
      )));
      const heatmapTrace = {
        type: 'heatmap',
        x: data.bucketLabels,
        y: data.holdLabels,
        z: data.z,
        colorscale: plateauColorscale,
        zmin: -8,
        zmax: 65,
        colorbar: {
          title: { text: '平均報酬率 (%)', side: 'right', font: { size: 15, color: '#4b5563' } },
          tickfont: { size: 14, color: '#4b5563' },
          thickness: 34,
          len: 0.88,
        },
        text: data.z.map(row => row.map(value => value === null ? '' : value.toFixed(1))),
        texttemplate: '%{text}',
        textfont: { size: 10, color: heatmapTextColors, family: 'Microsoft JhengHei, Noto Sans TC, Arial, sans-serif' },
        hovertemplate: 'RMDD: %{x}<br>持有期間: %{y}<br>平均報酬率: %{z:.2f}%<extra></extra>',
      };
      const sharedLayout = {
        paper_bgcolor: '#071019',
        plot_bgcolor: '#0d1a27',
        font: { family: 'Microsoft JhengHei, Noto Sans TC, Arial, sans-serif', color: '#c7d7e8', size: 13 },
        margin: { l: 58, r: 24, t: 54, b: 70 },
      };
      Plotly.react('parameterHeatmap', [heatmapTrace], {
        ...sharedLayout,
        margin: { l: 66, r: 82, t: 58, b: 82 },
        title: {
          text: '費半 SOX 報酬參數高原圖 (2D 俯視熱區)',
          font: { size: 18, color: '#f4fbff', family: 'Microsoft JhengHei, Noto Sans TC, Arial, sans-serif' },
          x: 0.5,
          xanchor: 'center',
        },
        xaxis: {
          title: { text: 'X軸: RMDD 買進條件區間', font: { size: 14, color: '#c7d7e8' } },
          tickangle: -45,
          categoryorder: 'array',
          categoryarray: data.bucketLabels,
          autorange: 'reversed',
          tickfont: { size: 12, color: '#c7d7e8' },
          gridcolor: 'rgba(149, 179, 210, 0.16)',
          zeroline: false,
        },
        yaxis: {
          title: { text: 'Y軸: 持有期間(月)', font: { size: 14, color: '#c7d7e8' } },
          categoryorder: 'array',
          categoryarray: data.holdLabels,
          autorange: 'reversed',
          tickfont: { size: 13, color: '#c7d7e8' },
          gridcolor: 'rgba(149, 179, 210, 0.16)',
          zeroline: false,
        },
      }, { responsive: true, displaylogo: false });

      const surfaceTrace = {
        type: 'surface',
        x: data.bucketLabels,
        y: data.holdLabels,
        z: data.zForSurface,
        colorscale: plateauColorscale,
        cmid: 0,
        cmin: -5,
        cmax: 65,
        colorbar: { title: '平均報酬率 (%)', len: 0.72, tickfont: { color: '#c7d7e8' }, titlefont: { color: '#c7d7e8' } },
        contours: { z: { show: true, usecolormap: true, highlightcolor: '#62e6ff', project: { z: true } } },
        hovertemplate: 'RMDD: %{x}<br>持有: %{y}<br>平均報酬率: %{z:.2f}%<extra></extra>',
      };
      Plotly.react('parameterSurface', [surfaceTrace], {
        ...sharedLayout,
        title: { text: '費半 SOX 參數高原表面 (3D Surface)', font: { size: 16, color: '#f4fbff' } },
        height: 470,
        scene: {
          bgcolor: '#0d1a27',
          xaxis: { title: 'RMDD 回撤區間', tickangle: -45, color: '#c7d7e8', gridcolor: 'rgba(149, 179, 210, 0.22)', zerolinecolor: 'rgba(149, 179, 210, 0.22)' },
          yaxis: { title: '持有期間', color: '#c7d7e8', gridcolor: 'rgba(149, 179, 210, 0.22)', zerolinecolor: 'rgba(149, 179, 210, 0.22)' },
          zaxis: { title: '平均報酬率 (%)', color: '#c7d7e8', gridcolor: 'rgba(149, 179, 210, 0.22)', zerolinecolor: 'rgba(149, 179, 210, 0.22)' },
          camera: { eye: { x: 1.45, y: 1.45, z: 0.9 } },
        },
      }, { responsive: true, displaylogo: false, scrollZoom: true });

      const barTrace = {
        type: 'bar',
        x: data.bucketLabels,
        y: data.probabilities,
        marker: { color: 'rgba(98, 230, 255, 0.78)', line: { color: '#071019', width: 1 } },
        text: data.probabilities.map(value => value ? `${value.toFixed(1)}%` : ''),
        textposition: 'outside',
        customdata: data.counts,
        hovertemplate: 'RMDD: %{x}<br>發生機率: %{y:.2f}%<br>樣本數: %{customdata} 天<extra></extra>',
      };
      Plotly.react('rmddSampleDistribution', [barTrace], {
        ...sharedLayout,
        height: 380,
        title: { text: '各 RMDD 回撤區間發生機率分布圖', font: { size: 16, color: '#f4fbff' } },
        xaxis: { title: 'RMDD 買進條件區間', tickangle: -45, gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)' },
        yaxis: { title: '發生機率（%）', gridcolor: 'rgba(149, 179, 210, 0.16)', zerolinecolor: 'rgba(149, 179, 210, 0.22)', rangemode: 'tozero', ticksuffix: '%' },
      }, { responsive: true, displaylogo: false });
    }

    function applyDates(pushState = true) {
      const startInput = document.getElementById('startDate');
      const endInput = document.getElementById('endDate');
      const baseInput = document.getElementById('baseAmount');
      const status = document.getElementById('dateStatus');
      try {
        if (!startInput.value || !endInput.value) throw new Error('請先輸入起訖日。');
        if (startInput.value > endInput.value) throw new Error('起始日不能晚於結束日。');
        const nextBaseAmount = Number(baseInput.value);
        if (!Number.isFinite(nextBaseAmount)) throw new Error('請輸入一份金額。');
        if (nextBaseAmount < 500 || nextBaseAmount > 50000) throw new Error('一份金額需介於 500 到 50,000。');
        strategyConfig.baseAmount = nextBaseAmount;
        const result = runStrategy(startInput.value, endInput.value);
        renderCards(result);
        renderPerformance(result);
        renderRuleTable();
        const payload = chartPayload(result);
        Plotly.react('chart', payload.traces, payload.layout, { responsive: true, displaylogo: false, scrollZoom: true });
        renderParameterCharts(result);
        if (pushState) {
          const params = new URLSearchParams(window.location.search);
          params.set('start', startInput.value);
          params.set('end', endInput.value);
          params.set('unit', String(strategyConfig.baseAmount));
          window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`);
        }
        status.textContent = '';
      } catch (error) {
        status.textContent = error.message;
      }
    }

    const params = new URLSearchParams(window.location.search);
    document.getElementById('startDate').value = params.get('start') || strategyConfig.startDate;
    document.getElementById('endDate').value = params.get('end') || strategyConfig.endDate;
    const unitParam = Number(params.get('unit') || defaultBaseAmount);
    document.getElementById('baseAmount').value = Number.isFinite(unitParam) ? Math.min(50000, Math.max(500, unitParam)) : defaultBaseAmount;
    document.getElementById('applyDates').addEventListener('click', () => applyDates(true));
    document.getElementById('baseAmount').addEventListener('keydown', event => {
      if (event.key === 'Enter') applyDates(true);
    });
    document.getElementById('resetDates').addEventListener('click', () => {
      document.getElementById('startDate').value = strategyConfig.startDate;
      document.getElementById('endDate').value = strategyConfig.endDate;
      document.getElementById('baseAmount').value = defaultBaseAmount;
      applyDates(true);
    });
    applyDates(false);
  </script>
"""
    return (
        script.replace("__MARKET_ROWS__", json.dumps(market_rows, ensure_ascii=False))
        .replace("__CLIENT_CONFIG__", json.dumps(client_config, ensure_ascii=False))
        .replace("__SOURCE_LABEL__", json.dumps(source_label, ensure_ascii=False))
        .replace("__TICKERS__", json.dumps(tickers, ensure_ascii=False))
    )


def build_html(result: dict, tickers: list[str], source_label: str, market_rows: list[dict]) -> str:
    curve = result["curve"]
    buys = result["buys"]
    metrics = result["metrics"]
    latest = curve.iloc[-1]
    latest_date = curve.index[-1].strftime("%Y-%m-%d")
    interactive_script = build_interactive_script(market_rows, build_client_config(CONFIG), tickers, source_label)
    chart_start = max(pd.Timestamp(CONFIG["CHART_START_DATE"]), result["data"].index[0])
    end_date = pd.Timestamp(CONFIG["END_DATE"])
    chart_end = pd.Timestamp(year=end_date.year + 1, month=12, day=31)
    chart_range = [chart_start.strftime("%Y-%m-%d"), chart_end.strftime("%Y-%m-%d")]

    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Apex Predator",
            "x": js_list(curve.index),
            "y": js_list(curve["Strat_Val"]),
            "line": {"color": "#62e6ff", "width": 3.4},
            "hovertemplate": "%{x}<br>Apex Predator: $%{y:,.0f}<extra></extra>",
            "xaxis": "x",
            "yaxis": "y",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Pure DCA",
            "x": js_list(curve.index),
            "y": js_list(curve["DCA_Val"]),
            "line": {"color": "#9db0c4", "width": 2, "dash": "dash"},
            "hovertemplate": "%{x}<br>Pure DCA: $%{y:,.0f}<extra></extra>",
            "xaxis": "x",
            "yaxis": "y",
        },
        make_marker_trace(buys, "Fallback Buy", "#46e6a8", "circle", 8),
        make_marker_trace(buys, "STD Level Buy", "#ff6b7a", "triangle-up", 10),
        make_marker_trace(buys, "Sniper Shot", "#b38cff", "star", 14),
        {
            "type": "scatter",
            "mode": "lines",
            "name": "VIX",
            "x": js_list(curve.index),
            "y": js_list(curve["VIX"]),
            "line": {"color": "#ffd166", "width": 1.8},
            "hovertemplate": "%{x}<br>VIX: %{y:.2f}<extra></extra>",
            "xaxis": "x2",
            "yaxis": "y2",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "RMDD",
            "x": js_list(curve.index),
            "y": js_list(curve["RMDD"] * 100),
            "line": {"color": "#66a6ff", "width": 2.6},
            "fill": "tozeroy",
            "fillcolor": "rgba(102, 166, 255, 0.16)",
            "hovertemplate": "%{x}<br>RMDD: %{y:.2f}%<extra></extra>",
            "xaxis": "x3",
            "yaxis": "y3",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "SOX Return",
            "x": js_list(curve.index),
            "y": js_list(curve["Cum_Ret"]),
            "line": {"color": "#46e6a8", "width": 2},
            "hovertemplate": "%{x}<br>SOX Return: %{y:.2f}%<extra></extra>",
            "xaxis": "x4",
            "yaxis": "y4",
        },
    ]

    shapes = [
        {"type": "line", "xref": "paper", "x0": 0, "x1": 0.83, "yref": "y2", "y0": CONFIG["VIX_PANIC_THRESHOLD"], "y1": CONFIG["VIX_PANIC_THRESHOLD"], "line": {"color": "#b38cff", "width": 2, "dash": "dash"}},
    ]
    for setting in CONFIG["LEVEL_CONFIG"]:
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 0.83,
                "yref": "y3",
                "y0": setting["threshold"] * 100,
                "y1": setting["threshold"] * 100,
                "line": {"color": "rgba(199, 215, 232, 0.54)", "width": 1, "dash": "dot"},
            }
        )
    for setting in CONFIG["SNIPER_CONFIG"]:
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 0.83,
                "yref": "y3",
                "y0": setting["threshold"] * 100,
                "y1": setting["threshold"] * 100,
                "line": {"color": "#ff6b7a", "width": 1.5, "dash": "solid"},
            }
        )

    layout = {
        "title": {"text": "", "x": 0.02, "font": {"size": 28}},
        "paper_bgcolor": "#071019",
        "plot_bgcolor": "#0d1a27",
        "font": {"family": "Arial, Microsoft JhengHei, Noto Sans TC, sans-serif", "color": "#c7d7e8", "size": 14},
        "height": 1040,
        "margin": {"l": 74, "r": 230, "t": 82, "b": 70},
        "hovermode": "x unified",
        "legend": {"x": 1.02, "y": 0.99, "xanchor": "left", "yanchor": "top", "font": {"size": 16, "color": "#f4fbff"}, "bgcolor": "rgba(13, 26, 39, 0.72)", "bordercolor": "rgba(149, 179, 210, 0.22)", "borderwidth": 1},
        "xaxis": {"domain": [0, 0.84], "anchor": "y", "range": chart_range, "showgrid": True, "gridcolor": "rgba(149, 179, 210, 0.16)", "zerolinecolor": "rgba(149, 179, 210, 0.22)", "showticklabels": False},
        "yaxis": {"domain": [0.60, 1.0], "title": "Portfolio Value ($)", "gridcolor": "rgba(149, 179, 210, 0.16)", "zerolinecolor": "rgba(149, 179, 210, 0.22)", "tickformat": "~s"},
        "xaxis2": {"domain": [0, 0.84], "anchor": "y2", "matches": "x", "showgrid": True, "gridcolor": "rgba(149, 179, 210, 0.16)", "zerolinecolor": "rgba(149, 179, 210, 0.22)", "showticklabels": False},
        "yaxis2": {"domain": [0.40, 0.54], "title": "VIX", "gridcolor": "rgba(149, 179, 210, 0.16)", "zerolinecolor": "rgba(149, 179, 210, 0.22)"},
        "xaxis3": {"domain": [0, 0.84], "anchor": "y3", "matches": "x", "showgrid": True, "gridcolor": "rgba(149, 179, 210, 0.16)", "zerolinecolor": "rgba(149, 179, 210, 0.22)", "showticklabels": False},
        "yaxis3": {"domain": [0.21, 0.35], "title": "RMDD (%)", "gridcolor": "rgba(149, 179, 210, 0.16)", "zerolinecolor": "rgba(149, 179, 210, 0.22)", "ticksuffix": "%"},
        "xaxis4": {"domain": [0, 0.84], "anchor": "y4", "matches": "x", "range": chart_range, "showgrid": True, "gridcolor": "rgba(149, 179, 210, 0.16)", "zerolinecolor": "rgba(149, 179, 210, 0.22)", "rangeslider": {"visible": True, "thickness": 0.06, "bgcolor": "#0d1a27", "bordercolor": "rgba(149, 179, 210, 0.28)"}},
        "yaxis4": {"domain": [0.02, 0.16], "title": "SOX Return (%)", "gridcolor": "rgba(149, 179, 210, 0.16)", "zerolinecolor": "rgba(149, 179, 210, 0.22)", "ticksuffix": "%"},
        "shapes": shapes,
        "annotations": [
            {"text": "Apex Predator Strategy<br>Portfolio Value ($) & Buy Events", "xref": "paper", "yref": "paper", "x": 0.42, "y": 1.04, "showarrow": False, "font": {"size": 24, "color": "#f4fbff"}},
            {"text": "VIX Index", "xref": "paper", "yref": "paper", "x": 0.42, "y": 0.56, "showarrow": False, "font": {"size": 22, "color": "#f4fbff"}},
            {"text": "RMDD Zones", "xref": "paper", "yref": "paper", "x": 0.42, "y": 0.37, "showarrow": False, "font": {"size": 22, "color": "#f4fbff"}},
            {"text": "SOX Cumulative Return", "xref": "paper", "yref": "paper", "x": 0.42, "y": 0.18, "showarrow": False, "font": {"size": 22, "color": "#f4fbff"}},
            {"text": "Panic Threshold", "xref": "paper", "yref": "y2", "x": 0.82, "y": CONFIG["VIX_PANIC_THRESHOLD"] + 1.5, "showarrow": False, "font": {"size": 14, "color": "#ffd166"}},
        ],
    }

    subtitle = f"最新資料日：{metrics['final_date'].date()}"
    source_details = f"資料來源：{escape(source_label)}｜原始 ticker：{escape(', '.join(map(str, tickers)))}"

    metrics_cards = f"""
      <div class="card"><span>最新 SOX</span><b>{latest['SOX']:,.2f}</b></div>
      <div class="card"><span>最新 VIX</span><b>{latest['VIX']:.2f}</b></div>
      <div class="card"><span>目前 RMDD</span><b>{latest['RMDD'] * 100:.2f}%</b></div>
    """
    start_date = result["data"].index[0]
    years = (metrics["final_date"] - start_date).days / 365.25
    absolute_return_strat = (metrics["final_val_strat"] / metrics["cost_strat"] - 1) * 100
    absolute_return_dca = (metrics["final_val_dca"] / metrics["cost_dca"] - 1) * 100
    annualized_absolute_strat = (
        (metrics["final_val_strat"] / metrics["cost_strat"]) ** (1 / years) - 1
    ) * 100
    annualized_absolute_dca = (
        (metrics["final_val_dca"] / metrics["cost_dca"]) ** (1 / years) - 1
    ) * 100
    total_xirr_strat = ((1 + metrics["xirr_strat"] / 100) ** years - 1) * 100
    total_xirr_dca = ((1 + metrics["xirr_dca"] / 100) ** years - 1) * 100
    cost_diff = metrics["cost_strat"] - metrics["cost_dca"]
    final_value_diff = metrics["final_val_strat"] - metrics["final_val_dca"]
    max_monthly_diff = metrics["max_monthly_cost"] - CONFIG["BASE_AMOUNT"]
    annualized_absolute_diff = annualized_absolute_strat - annualized_absolute_dca
    absolute_return_diff = absolute_return_strat - absolute_return_dca
    total_xirr_diff = total_xirr_strat - total_xirr_dca

    def difference_class(value: float) -> str:
        if value > 0:
            return "diff-positive"
        if value < 0:
            return "diff-negative"
        return "diff-neutral"

    fallback_count = int((buys["Type"] == "Fallback Buy").sum())
    standard_count = int((buys["Type"] == "STD Level Buy").sum())
    sniper_count = int((buys["Type"] == "Sniper Shot").sum())
    report_summary = f"""
  <header class="report-hero">
    <div class="hero-inner">
      <div class="hero-copy">
        <span class="report-kicker">SOX 回測研究報告</span>
        <h1>Apex Predator Strategy</h1>
        <p>這是一套以 SOX 為標的、根據市場回撤與恐慌程度動態加碼的長期投資策略。</p>
        <div class="hero-meta">
          <span class="subtitle">{subtitle}</span>
          <span>{start_date.date()} 至 {metrics['final_date'].date()}</span>
          <span>約 {years:.1f} 年</span>
        </div>
      </div>
      <div class="hero-stats" aria-label="核心績效數字">
        <div class="hero-stat">
          <span>Final Portfolio</span>
          <b id="heroFinalValue">{money(metrics['final_val_strat'])}</b>
        </div>
        <div class="hero-stat">
          <span>Total Invested</span>
          <b id="heroCost">{money(metrics['cost_strat'])}</b>
        </div>
        <div class="hero-stat">
          <span>Annualized XIRR</span>
          <b id="heroXirr">{pct(metrics['xirr_strat'])}</b>
        </div>
        <div class="hero-stat">
          <span>Buy Events</span>
          <b id="heroBuyCount">{len(buys):,}</b>
        </div>
      </div>
    </div>
  </header>
  <section class="strategy-thesis">
    <div class="thesis-copy">
      <span class="eyebrow">策略邏輯</span>
      <h2>用回撤決定加碼節奏，用恐慌辨識狙擊時點。</h2>
      <p>這份 Dashboard 把策略績效、買入事件與市場狀態放在同一張互動圖表中，方便快速比較 Apex Predator 與純定期定額的長期結果。</p>
    </div>
    <div class="factor-grid">
      <article class="factor-card">
        <span>01</span>
        <h3>RMDD</h3>
        <p>跌破預設回撤防線時，啟動標準加碼。</p>
      </article>
      <article class="factor-card">
        <span>02</span>
        <h3>VIX</h3>
        <p>市場進入恐慌區時，搭配深層回撤觸發狙擊買入。</p>
      </article>
      <article class="factor-card">
        <span>03</span>
        <h3>Seasonality</h3>
        <p>10 月保底買入提高至 1.5 份，反映季節性回落特徵。</p>
      </article>
    </div>
  </section>
"""
    strategy_details = """
  <section class="strategy-details">
    <div class="details-inner">
      <span class="eyebrow">策略詳細說明</span>
      <h2>Apex Predator 策略架構</h2>
      <p>Apex Predator 是一套改良版的 SOX 定期定額加碼策略（DCA），共有三大因子：</p>
      <p><b>1) RMDD</b>－追蹤 SOX 指數 1 年交易日的高點的回撤，跌破設定的防線（Level）時啟動標準加碼（STD Level Buy）</p>
      <p><b>2) S&amp;P500 恐慌指數 (VIX)</b>－判斷市場系統性恐慌程度，當 VIX 大於一定程度（市場很恐慌），且 RMDD 跌破深層防線時（且市場跌很深），將額外啟動狙擊加碼（Sniper Shot）</p>
      <p><b>3) 季節效應</b>－根據歷史資料回測，發現 10 月股市的回落程度顯著，因此在該月月底的基礎加碼（Fallback buy）將上調至 1.5 份。</p>
    </div>
  </section>
"""
    detailed_performance_summary = f"""
  <section class="detailed-performance">
    <div class="section-heading">
      <div>
        <span class="eyebrow">完整績效指標</span>
        <h2>Apex Predator 與純定期定額完整比較</h2>
      </div>
      <div class="period">{start_date.date()} 至 {metrics['final_date'].date()}<span>約 {years:.1f} 年</span></div>
    </div>
    <div class="comparison">
      <div class="comparison-head">
        <span>績效指標</span>
        <strong>Apex Predator</strong>
        <strong>純 DCA</strong>
        <strong>%／差值</strong>
      </div>
      <div class="comparison-row">
        <span>總投入成本</span>
        <b>{money(metrics['cost_strat'])}</b>
        <b>{money(metrics['cost_dca'])}</b>
        <b class="diff-neutral">{cost_diff:+,.0f}</b>
      </div>
      <div class="comparison-row">
        <span>最終資產價值</span>
        <b>{money(metrics['final_val_strat'])}</b>
        <b>{money(metrics['final_val_dca'])}</b>
        <b class="diff-neutral">{final_value_diff:+,.0f}</b>
      </div>
      <div class="comparison-row">
        <span>單月最高投入</span>
        <b>{money(metrics['max_monthly_cost'])}</b>
        <b>{money(CONFIG['BASE_AMOUNT'])}</b>
        <b class="diff-neutral">{max_monthly_diff:+,.0f}</b>
      </div>
      <div class="comparison-row return-divider">
        <span>絕對績效</span>
        <b>{absolute_return_strat:.2f}%</b>
        <b>{absolute_return_dca:.2f}%</b>
        <b class="{difference_class(absolute_return_diff)}">{absolute_return_diff:+.2f}%</b>
      </div>
      <div class="comparison-row">
        <span>平均年化絕對報酬</span>
        <b>{annualized_absolute_strat:.2f}%</b>
        <b>{annualized_absolute_dca:.2f}%</b>
        <b class="{difference_class(annualized_absolute_diff)}">{annualized_absolute_diff:+.2f}%</b>
      </div>
      <div class="comparison-row">
        <span>總體 XIRR</span>
        <b>{total_xirr_strat:.2f}%</b>
        <b>{total_xirr_dca:.2f}%</b>
        <b class="{difference_class(total_xirr_diff)}">{total_xirr_diff:+.2f}%</b>
      </div>
      <div class="comparison-row emphasis">
        <span>年化報酬率 XIRR</span>
        <b>{pct(metrics['xirr_strat'])}</b>
        <b>{pct(metrics['xirr_dca'])}</b>
        <b class="{difference_class(metrics['xirr_diff'])}">{metrics['xirr_diff']:+.2f}%</b>
      </div>
      <div class="comparison-footnote">差值為 Apex Predator − 純 DCA。<br>XIRR（擴展內部報酬率）是專門用來計算「不定期、多筆現金流」的年化報酬率指標。<br>總體 XIRR 為年化 XIRR 按完整回測期間複利換算。</div>
    </div>
  </section>
"""
    performance_summary = f"""
  <section class="performance">
    <div class="section-heading">
      <div>
        <span class="eyebrow">策略績效摘要</span>
        <h2>Apex Predator 與純定期定額</h2>
      </div>
      <div class="period">{start_date.date()} 至 {metrics['final_date'].date()}<span>約 {years:.1f} 年</span></div>
    </div>
    <div class="performance-grid">
      <div class="summary-panel">
        <div class="outcome-grid">
          <article class="outcome-card primary">
            <span>最終資產差額</span>
            <b>{final_value_diff:+,.0f}</b>
            <small>Apex Predator {money(metrics['final_val_strat'])} / 純 DCA {money(metrics['final_val_dca'])}</small>
          </article>
          <article class="outcome-card">
            <span>年化 XIRR 差距</span>
            <b class="{difference_class(metrics['xirr_diff'])}">{metrics['xirr_diff']:+.2f}%</b>
            <small>{pct(metrics['xirr_strat'])} vs {pct(metrics['xirr_dca'])}</small>
          </article>
          <article class="outcome-card">
            <span>總投入差額</span>
            <b>{cost_diff:+,.0f}</b>
            <small>{money(metrics['cost_strat'])} vs {money(metrics['cost_dca'])}</small>
          </article>
          <article class="outcome-card">
            <span>買入頻率</span>
            <b>{len(buys) / years:.1f} 次／年</b>
            <small>累計 {len(buys):,} 次買入</small>
          </article>
        </div>
        <div class="compact-comparison">
          <div class="compact-row compact-head"><span>核心指標</span><strong>Apex Predator</strong><strong>純 DCA</strong><strong>差值</strong></div>
          <div class="compact-row"><span>最終資產</span><b>{money(metrics['final_val_strat'])}</b><b>{money(metrics['final_val_dca'])}</b><b class="diff-neutral">{final_value_diff:+,.0f}</b></div>
          <div class="compact-row"><span>總投入</span><b>{money(metrics['cost_strat'])}</b><b>{money(metrics['cost_dca'])}</b><b class="diff-neutral">{cost_diff:+,.0f}</b></div>
          <div class="compact-row emphasis"><span>年化 XIRR</span><b>{pct(metrics['xirr_strat'])}</b><b>{pct(metrics['xirr_dca'])}</b><b class="{difference_class(metrics['xirr_diff'])}">{metrics['xirr_diff']:+.2f}%</b></div>
        </div>
      </div>
      <aside class="trade-summary">
        <div class="trade-title">交易訊號統計</div>
        <div class="trade-total">
          <div><b>{len(buys)}</b><span>總買入次數</span></div>
        </div>
        <div class="trade-list cumulative-list">
          <div class="trade-list-head"><span></span><span></span><small>累計</small></div>
          <div><i class="fallback-dot"></i><span>保底買入</span><b>{fallback_count}</b></div>
          <div><i class="standard-dot"></i><span>標準防線</span><b>{standard_count}</b></div>
          <div><i class="sniper-dot"></i><span>狙擊觸發</span><b>{sniper_count}</b></div>
        </div>
        <p>狙擊條件：VIX &gt; {CONFIG['VIX_PANIC_THRESHOLD']} 且 RMDD 跌破對應防線。</p>
        <div class="annual-section">
          <div class="trade-title">交易訊號統計</div>
          <div class="annual-total"><b>{len(buys) / years:.1f}</b><span>次／年</span></div>
          <div class="trade-list annual-list">
            <div class="trade-list-head"><span></span><span></span><small>年均</small></div>
            <div><i class="fallback-dot"></i><span>保底買入</span><em>{fallback_count / years:.1f}</em></div>
            <div><i class="standard-dot"></i><span>標準防線</span><em>{standard_count / years:.1f}</em></div>
            <div><i class="sniper-dot"></i><span>狙擊觸發</span><em>{sniper_count / years:.1f}</em></div>
          </div>
        </div>
      </aside>
    </div>
  </section>
"""
    rule_table = build_rule_table(CONFIG)

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Apex Predator Strategy</title>
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
  <link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
  <link rel="manifest" href="site.webmanifest">
  <meta name="theme-color" content="#071019">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    :root {{
      --bg: #071019;
      --panel: #0d1a27;
      --panel-2: #112235;
      --panel-3: #14283d;
      --line: rgba(149, 179, 210, 0.22);
      --line-strong: rgba(116, 232, 255, 0.46);
      --text: #f4fbff;
      --muted: #9db0c4;
      --soft: #c7d7e8;
      --cyan: #62e6ff;
      --blue: #66a6ff;
      --amber: #ffd166;
      --green: #46e6a8;
      --red: #ff6b7a;
      --purple: #b38cff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, 'Microsoft JhengHei', 'Noto Sans TC', sans-serif; background: var(--bg); color: var(--text); }}
    body::before {{ content: ""; position: fixed; inset: 0; z-index: -1; background: linear-gradient(118deg, rgba(98, 230, 255, 0.14) 0%, transparent 28%, rgba(255, 209, 102, 0.08) 62%, transparent 100%), linear-gradient(180deg, #071019 0%, #091827 48%, #071019 100%); }}
    .report-hero {{ position: relative; overflow: hidden; padding: 68px 28px 42px; background: linear-gradient(135deg, rgba(7, 16, 25, 0.92), rgba(11, 31, 48, 0.94)); border-bottom: 1px solid var(--line); }}
    .report-hero::after {{ content: ""; position: absolute; inset: auto 0 0; height: 1px; background: linear-gradient(90deg, transparent, var(--cyan), var(--amber), transparent); opacity: 0.85; }}
    .hero-inner {{ position: relative; display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(380px, 0.95fr); gap: 36px; align-items: end; max-width: 1180px; margin: 0 auto; }}
    .report-kicker {{ display: inline-block; margin-bottom: 14px; color: var(--cyan); font-size: 12px; font-weight: 700; letter-spacing: 0; text-transform: uppercase; }}
    h1 {{ margin: 0; color: var(--text); font-size: 58px; line-height: 0.98; letter-spacing: 0; }}
    .hero-copy p {{ max-width: 690px; margin: 22px 0 0; color: var(--soft); font-size: 20px; line-height: 1.7; }}
    .hero-meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 28px; color: var(--muted); font-size: 13px; }}
    .hero-meta span {{ padding: 8px 11px; background: rgba(13, 26, 39, 0.86); border: 1px solid var(--line); border-radius: 6px; }}
    .subtitle {{ color: var(--cyan); font-weight: 700; }}
    .hero-stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .hero-stat {{ min-height: 118px; padding: 18px; background: linear-gradient(180deg, rgba(20, 40, 61, 0.92), rgba(10, 24, 37, 0.92)); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 18px 50px rgba(0, 0, 0, 0.32); }}
    .hero-stat:nth-child(1), .hero-stat:nth-child(3) {{ border-color: rgba(98, 230, 255, 0.34); }}
    .hero-stat span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .hero-stat b {{ display: block; margin-top: 15px; color: var(--text); font-size: 28px; line-height: 1.05; font-variant-numeric: tabular-nums; }}
    .hero-stat:nth-child(1) b, .hero-stat:nth-child(3) b {{ color: var(--cyan); }}
    .strategy-thesis {{ display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr); gap: 24px; max-width: 1180px; margin: 30px auto 0; padding: 0 28px; }}
    .thesis-copy h2 {{ margin: 4px 0 12px; color: var(--text); font-size: 34px; line-height: 1.18; }}
    .thesis-copy p {{ margin: 0; color: var(--soft); font-size: 15px; line-height: 1.85; }}
    .factor-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .factor-card {{ min-height: 166px; padding: 20px; background: linear-gradient(180deg, rgba(17, 34, 53, 0.98), rgba(10, 22, 35, 0.98)); border: 1px solid var(--line); border-radius: 8px; }}
    .factor-card span {{ color: var(--amber); font-size: 12px; font-weight: 700; }}
    .factor-card h3 {{ margin: 24px 0 10px; color: var(--text); font-size: 22px; }}
    .factor-card p {{ margin: 0; color: var(--soft); font-size: 14px; line-height: 1.7; }}
    .date-controls {{ max-width: 1180px; margin: 0 auto; padding: 18px 28px 0; }}
    .date-panel {{ display: flex; align-items: end; gap: 12px; flex-wrap: wrap; background: rgba(13, 26, 39, 0.92); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .date-panel label {{ display: grid; gap: 6px; color: var(--muted); font-size: 12px; }}
    .date-panel input {{ height: 36px; min-width: 150px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; color: var(--text); font: inherit; font-size: 14px; background: rgba(7, 16, 25, 0.9); }}
    .date-panel button {{ height: 36px; border: 1px solid rgba(98, 230, 255, 0.42); border-radius: 6px; padding: 0 14px; color: #031018; background: var(--cyan); font: inherit; font-size: 13px; font-weight: 700; cursor: pointer; }}
    .date-panel button.secondary {{ background: transparent; color: var(--soft); border-color: var(--line); }}
    .date-status {{ margin-left: auto; color: var(--muted); font-size: 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; max-width: 1180px; margin: 0 auto; padding: 16px 28px 0; }}
    .card {{ background: rgba(13, 26, 39, 0.92); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .card span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 7px; }}
    .card b {{ color: var(--text); font-size: 22px; }}
    .performance {{ max-width: 1180px; margin: 22px auto 4px; padding: 0 28px; }}
    .signal-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; margin-bottom: 16px; }}
    .signal-card {{ position: relative; min-height: 176px; background: rgba(13, 26, 39, 0.94); border: 1px solid var(--line); border-radius: 8px; padding: 26px 28px; overflow: hidden; }}
    .signal-card.active::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 8px; background: var(--amber); }}
    .signal-card.quiet::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 8px; background: var(--cyan); }}
    .signal-pill {{ display: inline-flex; align-items: center; min-height: 30px; padding: 0 14px; border-radius: 999px; background: rgba(255, 209, 102, 0.14); color: var(--amber); font-size: 15px; font-weight: 700; }}
    .signal-card.quiet .signal-pill {{ background: rgba(98, 230, 255, 0.12); color: var(--cyan); }}
    .signal-card h3 {{ margin: 28px 0 12px; color: var(--text); font-size: 26px; line-height: 1.2; letter-spacing: 0; }}
    .signal-card.expectation h3 {{ margin-top: 4px; font-size: 24px; }}
    .signal-list {{ display: grid; gap: 7px; margin: 0; padding: 0; list-style: none; }}
    .signal-list li {{ display: grid; grid-template-columns: 86px minmax(0, 1fr) auto; gap: 10px; align-items: center; color: var(--soft); font-size: 14px; }}
    .signal-list li b {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
    .signal-list li span {{ min-width: 0; line-height: 1.4; }}
    .signal-list li strong {{ color: var(--text); font-size: 15px; font-variant-numeric: tabular-nums; }}
    .expectation-lines {{ display: grid; gap: 13px; margin-top: 28px; }}
    .expectation-lines p {{ margin: 0; color: var(--text); font-size: 18px; line-height: 1.35; }}
    .expectation-lines b {{ font-weight: 700; }}
    .section-heading {{ display: flex; justify-content: space-between; align-items: end; gap: 24px; margin-bottom: 12px; }}
    .section-heading h2 {{ margin: 3px 0 0; font-size: 25px; color: var(--text); }}
    .eyebrow {{ color: var(--cyan); font-size: 12px; font-weight: 700; letter-spacing: 0; }}
    .period {{ color: var(--soft); font-size: 14px; text-align: right; }}
    .period span {{ display: block; margin-top: 3px; color: var(--muted); font-size: 12px; }}
    .performance-grid {{ display: grid; grid-template-columns: minmax(0, 2.25fr) minmax(240px, 0.75fr); gap: 12px; }}
    .summary-panel {{ display: grid; gap: 12px; }}
    .outcome-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .outcome-card {{ min-height: 134px; padding: 18px; background: rgba(13, 26, 39, 0.94); border: 1px solid var(--line); border-radius: 8px; }}
    .outcome-card.primary {{ border-color: rgba(98, 230, 255, 0.46); background: linear-gradient(180deg, rgba(98, 230, 255, 0.13), rgba(13, 26, 39, 0.94)); }}
    .outcome-card span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; }}
    .outcome-card b {{ display: block; margin-top: 14px; color: var(--text); font-size: 25px; line-height: 1.1; font-variant-numeric: tabular-nums; }}
    .outcome-card.primary b {{ color: var(--cyan); }}
    .outcome-card small {{ display: block; margin-top: 12px; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    .compact-comparison {{ background: rgba(13, 26, 39, 0.94); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .compact-row {{ display: grid; grid-template-columns: minmax(120px, 1fr) minmax(126px, 1fr) minmax(126px, 1fr) minmax(92px, 0.72fr); align-items: center; gap: 14px; padding: 13px 16px; border-top: 1px solid rgba(149, 179, 210, 0.14); color: var(--soft); font-size: 14px; }}
    .compact-row:first-child {{ border-top: 0; }}
    .compact-head {{ background: rgba(98, 230, 255, 0.09); color: var(--muted); font-size: 13px; }}
    .compact-row b, .compact-row strong {{ color: var(--text); font-variant-numeric: tabular-nums; }}
    .compact-row.emphasis {{ background: rgba(98, 230, 255, 0.08); }}
    .compact-row.emphasis b {{ color: var(--cyan); font-size: 17px; }}
    .comparison, .trade-summary {{ background: rgba(13, 26, 39, 0.94); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .comparison-head, .comparison-row {{ display: grid; grid-template-columns: minmax(140px, 1.15fr) minmax(130px, 1fr) minmax(130px, 1fr) minmax(100px, 0.75fr); align-items: center; column-gap: 16px; padding: 11px 16px; }}
    .comparison-head {{ background: rgba(98, 230, 255, 0.09); color: var(--soft); font-size: 13px; }}
    .comparison-head strong {{ font-size: 14px; }}
    .comparison-row {{ min-height: 28px; border-top: 1px solid rgba(149, 179, 210, 0.14); font-size: 14px; }}
    .comparison-row > span:first-child {{ color: var(--muted); }}
    .comparison-row b {{ color: var(--text); font-size: 15px; font-variant-numeric: tabular-nums; }}
    .comparison-row.emphasis {{ background: rgba(98, 230, 255, 0.08); }}
    .comparison-row.emphasis b {{ font-size: 18px; color: var(--cyan); }}
    .comparison-row.return-divider {{ border-top: 2px solid rgba(98, 230, 255, 0.28); }}
    .comparison-row .diff-positive {{ color: var(--red); }}
    .comparison-row .diff-negative {{ color: var(--green); }}
    .comparison-row .diff-neutral {{ color: var(--soft); font-size: 15px; font-weight: 700; font-family: inherit; font-variant-numeric: tabular-nums; }}
    .comparison-footnote {{ padding: 10px 16px 12px; border-top: 1px solid rgba(149, 179, 210, 0.14); color: var(--muted); font-size: 11px; line-height: 1.5; }}
    .trade-summary {{ display: flex; flex-direction: column; padding: 16px; }}
    .trade-title {{ color: var(--soft); font-size: 13px; font-weight: 700; }}
    .trade-total {{ padding: 12px 0 14px; border-bottom: 1px solid rgba(149, 179, 210, 0.14); }}
    .trade-total > div {{ display: flex; align-items: baseline; gap: 7px; min-width: 0; }}
    .trade-total b {{ font-size: 34px; color: var(--cyan); }}
    .trade-total span {{ color: var(--muted); font-size: 12px; }}
    .trade-list {{ padding: 8px 0; }}
    .trade-list div {{ display: grid; grid-template-columns: 12px minmax(82px, 1fr) 38px; align-items: center; gap: 7px; min-height: 30px; color: var(--soft); font-size: 14px; }}
    .trade-list .trade-list-head {{ min-height: 20px; color: var(--muted); font-size: 10px; text-align: right; }}
    .trade-list small {{ font-size: 10px; font-weight: 400; }}
    .trade-list i {{ width: 8px; height: 8px; border-radius: 50%; }}
    .fallback-dot {{ background: var(--green); }}
    .standard-dot {{ background: var(--red); }}
    .sniper-dot {{ background: var(--purple); }}
    .trade-list b {{ color: var(--text); font-size: 15px; text-align: right; }}
    .trade-list em {{ color: var(--cyan); font-size: 13px; font-style: normal; font-weight: 700; text-align: right; }}
    .trade-summary p {{ margin: 6px 0 0; color: var(--muted); font-size: 12px; line-height: 1.55; }}
    .annual-section {{ margin-top: 30px; padding-top: 14px; border-top: 1px solid rgba(149, 179, 210, 0.14); }}
    .annual-total {{ display: inline-flex; align-items: baseline; gap: 6px; margin: 9px 0 8px; padding: 7px 9px; background: rgba(98, 230, 255, 0.09); }}
    .annual-total b {{ color: var(--cyan); font-size: 30px; line-height: 1; }}
    .annual-total span {{ color: var(--muted); font-size: 12px; }}
    .annual-list {{ padding-bottom: 0; border-top: 1px solid rgba(149, 179, 210, 0.14); }}
    #chart {{ width: min(1280px, 100vw); height: 1040px; margin: 4px auto 0; }}
    .note {{ max-width: 1180px; margin: 0 auto 28px; padding: 0 28px; color: var(--muted); font-size: 13px; }}
    .rules {{ max-width: 1180px; margin: 0 auto 34px; padding: 0 28px; }}
    .rules h2 {{ margin: 0 0 8px; font-size: 24px; color: var(--text); }}
    .rule-help {{ margin: 0 0 12px; color: var(--muted); font-size: 14px; }}
    .rule-scroll {{ overflow-x: auto; background: rgba(13, 26, 39, 0.94); border: 1px solid var(--line); border-radius: 8px; }}
    .rule-table {{ width: 100%; border-collapse: collapse; min-width: 1040px; table-layout: fixed; font-size: 14px; }}
    .rule-table th, .rule-table td {{ padding: 7px 7px; border-right: 1px solid rgba(149, 179, 210, 0.18); border-bottom: 1px solid rgba(149, 179, 210, 0.18); text-align: left; vertical-align: middle; }}
    .rule-table thead th {{ background: rgba(98, 230, 255, 0.1); color: var(--text); font-size: 15px; }}
    .rule-table thead .rule-group-head th {{ text-align: center; vertical-align: middle; border-right: 1px solid rgba(149, 179, 210, 0.18); }}
    .rule-table thead .rule-group-head th:last-child {{ text-align: left; border-left: 1px solid rgba(149, 179, 210, 0.18); border-right: 0; }}
    .rule-table thead .rule-group-head th {{ height: 54px; }}
    .rule-table thead .rule-subhead th {{ height: 28px; padding-top: 4px; padding-bottom: 4px; text-align: center; background: rgba(20, 40, 61, 0.92); border-right: 1px solid rgba(149, 179, 210, 0.18); }}
    .rule-table .rmdd-col {{ width: 17%; }}
    .rule-table .rule-col {{ width: 15%; }}
    .rule-table .amount-col {{ width: 10.5%; }}
    .rule-table .note-col {{ width: 32%; }}
    .rule-table tbody th {{ background: rgba(20, 40, 61, 0.72); color: var(--text); font-weight: 700; text-align: center; vertical-align: middle; line-height: 1.3; }}
    .rule-table tbody td {{ height: 22px; line-height: 1.3; }}
    .rule-table .rule-cell {{ color: var(--soft); }}
    .rule-table .amount-cell {{ color: var(--cyan); font-weight: 700; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .rule-table .grouped-cell {{ vertical-align: middle; }}
    .rule-table .note-cell {{ border-left: 1px solid rgba(149, 179, 210, 0.18); color: var(--soft); line-height: 1.35; }}
    .rule-table span {{ color: var(--muted); font-weight: 400; font-size: 14px; }}
    .rule-table th:last-child, .rule-table td:last-child {{ border-right: 0; }}
    .rule-table tr:last-child th, .rule-table tr:last-child td {{ border-bottom: 0; }}
    .parameter-analysis {{ max-width: 1180px; margin: 0 auto 42px; padding: 0 28px; }}
    .parameter-analysis h2 {{ margin: 0 0 8px; font-size: 24px; color: var(--text); }}
    .parameter-help {{ margin: 0 0 12px; color: var(--muted); font-size: 14px; }}
    .parameter-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }}
    .parameter-chart {{ min-height: 470px; background: rgba(13, 26, 39, 0.94); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .parameter-chart.wide {{ grid-column: 1 / -1; min-height: 380px; }}
    .strategy-details {{ margin-top: 10px; padding: 38px 28px 50px; background: #09131f; border-top: 1px solid var(--line); }}
    .details-inner {{ max-width: 900px; margin: 0 auto; }}
    .strategy-details h2 {{ margin: 4px 0 14px; color: var(--text); font-size: 24px; }}
    .strategy-details p {{ margin: 0 0 9px; color: var(--soft); font-size: 15px; line-height: 1.8; }}
    .strategy-details b {{ color: var(--cyan); }}
    .detailed-performance {{ max-width: 1180px; margin: 0 auto 54px; padding: 30px 28px 0; }}
    @media (max-width: 900px) {{
      .report-hero {{ padding: 34px 18px 24px; }}
      .hero-inner {{ grid-template-columns: 1fr; gap: 18px; }}
      h1 {{ font-size: 36px; }}
      .hero-copy p {{ font-size: 16px; }}
      .strategy-thesis {{ grid-template-columns: 1fr; padding: 0 12px; }}
      .cards {{ grid-template-columns: repeat(3, minmax(0, 1fr)); padding: 12px; }}
      .date-controls {{ padding: 12px; }}
      .date-status {{ margin-left: 0; flex-basis: 100%; }}
      .performance {{ padding: 0 12px; }}
      .performance-grid {{ grid-template-columns: 1fr; }}
      .outcome-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .signal-grid {{ grid-template-columns: 1fr; }}
      .parameter-analysis {{ padding: 0 12px; }}
      .detailed-performance {{ padding: 24px 12px 0; }}
      .parameter-grid {{ grid-template-columns: 1fr; }}
      .section-heading {{ align-items: start; flex-direction: column; gap: 8px; }}
      .period {{ text-align: left; }}
    }}
    @media (max-width: 620px) {{
      h1 {{ font-size: 31px; }}
      .report-hero {{ padding: 28px 18px 22px; }}
      .hero-copy p {{ margin-top: 14px; font-size: 15px; line-height: 1.65; }}
      .hero-meta {{ gap: 8px; margin-top: 18px; }}
      .hero-meta span {{ padding: 6px 9px; }}
      .hero-stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
      .hero-stat {{ min-height: 94px; padding: 13px; }}
      .hero-stat span {{ font-size: 10px; }}
      .hero-stat b {{ margin-top: 12px; font-size: 20px; }}
      .factor-grid {{ grid-template-columns: 1fr; }}
      .outcome-grid {{ grid-template-columns: 1fr; }}
      .compact-comparison {{ overflow-x: auto; }}
      .compact-row {{ min-width: 620px; }}
      .cards {{ grid-template-columns: 1fr; }}
      .signal-card {{ padding: 20px 18px 20px 24px; }}
      .signal-card h3, .signal-card.expectation h3 {{ font-size: 21px; }}
      .signal-list li {{ grid-template-columns: 1fr; gap: 4px; }}
      .comparison {{ overflow-x: auto; }}
      .comparison-head, .comparison-row {{ min-width: 620px; }}
    }}
  </style>
</head>
<body>
  {report_summary}
  <section class="date-controls">
    <div class="date-panel">
      <label>起始日期<input id="startDate" type="date"></label>
      <label>結束日期<input id="endDate" type="date"></label>
      <label>一份金額<input id="baseAmount" type="number" min="500" max="50000" step="500" inputmode="numeric"></label>
      <button id="applyDates" type="button">套用設定</button>
      <button id="resetDates" class="secondary" type="button">回到預設</button>
      <span id="dateStatus" class="date-status"></span>
    </div>
  </section>
  <section class="cards">{metrics_cards}</section>
  {performance_summary}
  <div id="chart"></div>
  <div class="note">互動提示：拖曳可放大區間，雙擊可重設；右側圖例可點擊開關線條。｜{source_details}</div>
  {rule_table}
  <section class="parameter-analysis">
    <h2>參數高原分析</h2>
    <p class="parameter-help">依目前選取日期區間重新計算：RMDD &lt;= -10% 的買進樣本，以 2.5% 回撤區間分組，觀察 1M 到 12M 後續平均報酬與發生機率。</p>
    <div class="parameter-grid">
      <div id="parameterHeatmap" class="parameter-chart"></div>
      <div id="parameterSurface" class="parameter-chart"></div>
      <div id="rmddSampleDistribution" class="parameter-chart wide"></div>
    </div>
  </section>
  {strategy_details}
  {detailed_performance_summary}
  {interactive_script}
</body>
</html>"""


def main() -> None:
    args = parse_args()
    pd.Timestamp(args.start_date)
    CONFIG["START_DATE"] = args.start_date

    merged, tickers, source_label = load_market_data()
    end_date_arg = str(args.end_date).strip()
    if end_date_arg.lower() == "auto":
        CONFIG["END_DATE"] = merged.index.max().strftime("%Y-%m-%d")
    else:
        pd.Timestamp(end_date_arg)
        CONFIG["END_DATE"] = end_date_arg

    result = run_strategy(merged, CONFIG)
    market_rows = build_market_rows(merged, CONFIG)
    html = build_html(result, tickers, source_label, market_rows)
    OUT_FILE.write_text(html, encoding="utf-8")
    PAGES_FILE.write_text(html, encoding="utf-8")
    metrics = result["metrics"]
    buys = result["buys"]
    print(f"Wrote: {OUT_FILE}")
    print(f"Wrote: {PAGES_FILE}")
    print(f"Latest data date: {metrics['final_date'].date()}")
    print(f"Buy rows: {len(buys)}")
    print(f"XIRR diff: {metrics['xirr_diff']:.2f}%")


if __name__ == "__main__":
    main()
