import streamlit as st
import io
import math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------
# Dependencies / versions:
# pandas, numpy, xlrd<2.0, openpyxl, xlsxwriter, streamlit
# --------------------------

# ------------------------------------------------------------
# Preferred header rows (0-based) based on your samples.
# (App will still auto-detect, but prefers these.)
# ------------------------------------------------------------
PREFERRED_HEADER_ROW = {"ws": 3, "spr": 3, "map": 0}

EXPECTED = {
    "ws":  ["Item Code", "Item Name", "ML"],
    "spr": ["Item Name", "Net Sales Qty", "Outlet Name", "Current Stock"],  # Item Code may be absent
    "map": ["Bulk Item Code", "Bulk Item Name", "Retail Item Code", "Retail Item Name",
            "Unit Factor (bulk units per 1 retail unit)"],
}

COLS = {
    "ws": {"code": "Item Code", "name": "Item Name", "qty": "ML"},
    "spr": {
        "code":   "Item Code",        # may not exist
        "name":   "Item Name",
        "outlet": "Outlet Name",
        "net":    "Net Sales Qty",
        "stock":  "Current Stock",
    },
    "map": {
        "bulk_code": "Bulk Item Code",
        "bulk_name": "Bulk Item Name",
        "ret_code":  "Retail Item Code",
        "ret_name":  "Retail Item Name",
        "conv":     "Unit Factor (bulk units per 1 retail unit)",
    }
}

WAREHOUSE_KEYWORDS = ["warehouse", "wh", "godown", "dc", "central"]  # used to filter SPR rows

def _engine_for_filename(name: str) -> str:
    suf = Path(name).suffix.lower()
    if suf == ".xlsx":
        return "openpyxl"
    if suf == ".xls":
        return "xlrd"   # requires xlrd<2.0
    raise ValueError(f"Unsupported file type: {suf}")

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum() or ch.isspace())

def _detect_header_row(file, kind: str) -> int:
    """Scan first 15 rows to find header row."""
    expected_cols = EXPECTED[kind]
    prefer = PREFERRED_HEADER_ROW.get(kind, None)
    engine = _engine_for_filename(file.name)
    sample = pd.read_excel(file, header=None, nrows=15, engine=engine)
    file.seek(0)

    want = [_norm(c) for c in expected_cols]
    def score_row(r):
        vals = sample.iloc[r].tolist()
        row_norm = [_norm(v) for v in vals]
        return sum(1 for w in want if w in row_norm)

    if prefer is not None and 0 <= prefer < sample.shape[0]:
        if score_row(prefer) >= max(2, min(3, len(want))):
            return prefer

    best_row, best_score = None, -1
    for r in range(sample.shape[0]):
        sc = score_row(r)
        if sc > best_score:
            best_row, best_score = r, sc
    return best_row if best_row is not None else (prefer if prefer is not None else 3)

def read_table(file, kind: str) -> pd.DataFrame:
    engine = _engine_for_filename(file.name)
    header = _detect_header_row(file, kind)
    df = pd.read_excel(file, header=header, engine=engine)
    file.seek(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_ws(ws_df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in [COLS["ws"]["code"], COLS["ws"]["name"], COLS["ws"]["qty"]] if c in ws_df.columns]
    ws = ws_df[keep].copy()
    ws[COLS["ws"]["qty"]] = pd.to_numeric(ws[COLS["ws"]["qty"]], errors="coerce").fillna(0)
    if COLS["ws"]["code"] in ws.columns:
        ws_agg = (
            ws.groupby([COLS["ws"]["code"], COLS["ws"]["name"]], dropna=False, as_index=False)[COLS["ws"]["qty"]]
              .sum()
              .rename(columns={COLS["ws"]["qty"]: "Warehouse_Qty"})
        )
    else:
        ws_agg = ws.groupby([COLS["ws"]["name"]], dropna=False, as_index=False)[COLS["ws"]["qty"]].sum()
        ws_agg = ws_agg.rename(columns={COLS["ws"]["name"]: "Item Name", COLS["ws"]["qty"]: "Warehouse_Qty"})
    return ws_agg

def load_spr(spr_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    keep = [c for c in [COLS["spr"]["code"], COLS["spr"]["name"], COLS["spr"]["outlet"], COLS["spr"]["net"], COLS["spr"]["stock"]] if c in spr_df.columns]
    spr = spr_df[keep].copy()
    if COLS["spr"]["net"] in spr.columns:
        spr[COLS["spr"]["net"]] = pd.to_numeric(spr[COLS["spr"]["net"]], errors="coerce").fillna(0)
    if COLS["spr"]["stock"] in spr.columns:
        spr[COLS["spr"]["stock"]] = pd.to_numeric(spr[COLS["spr"]["stock"]], errors="coerce").fillna(0)

    has_outlet = COLS["spr"]["outlet"] in spr.columns
    if has_outlet:
        oc = COLS["spr"]["outlet"]
        def is_wh(x):
            s = str(x).strip().lower()
            return any(k in s for k in WAREHOUSE_KEYWORDS)
        spr = spr[~spr[oc].astype(str).map(is_wh)].copy()

    spr["Outlet_Retail_Need"] = (spr.get(COLS["spr"]["net"], 0) - spr.get(COLS["spr"]["stock"], 0)).clip(lower=0)
    return spr, (COLS["spr"]["code"] in spr.columns)

def load_map(map_df: pd.DataFrame) -> pd.DataFrame:
    need = [COLS["map"]["bulk_code"], COLS["map"]["bulk_name"], COLS["map"]["ret_code"], COLS["map"]["ret_name"], COLS["map"]["conv"]]
    keep = [c for c in need if c in map_df.columns]
    m = map_df[keep].copy()
    if COLS["map"]["conv"] not in m.columns:
        m[COLS["map"]["conv"]] = 1.0
    m[COLS["map"]["conv"]] = pd.to_numeric(m[COLS["map"]["conv"]], errors="coerce").fillna(1.0)
    keys = [c for c in [COLS["map"]["bulk_code"], COLS["map"]["ret_code"]] if c in m.columns]
    if keys:
        m = m.drop_duplicates(subset=keys, keep="first")
    return m

def compute_retail_needs(spr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = []
    if COLS["spr"]["code"] in spr.columns:
        group_cols.append(COLS["spr"]["code"])
    group_cols.append(COLS["spr"]["name"])
    if COLS["spr"]["outlet"] in spr.columns:
        group_cols.append(COLS["spr"]["outlet"])

    per_outlet = spr.groupby(group_cols, dropna=False, as_index=False)["Outlet_Retail_Need"].sum()

    rename_map = {}
    if COLS["spr"]["code"] in spr.columns: rename_map[COLS["spr"]["code"]] = "Retail_Item_Code"
    rename_map[COLS["spr"]["name"]] = "Retail_Item_Name"
    if COLS["spr"]["outlet"] in spr.columns: rename_map[COLS["spr"]["outlet"]] = "Outlet"
    per_outlet = per_outlet.rename(columns=rename_map)
    if "Outlet" not in per_outlet.columns:
        per_outlet["Outlet"] = "ALL"

    totals = (
        per_outlet.groupby([c for c in ["Retail_Item_Code", "Retail_Item_Name"] if c in per_outlet.columns],
                           dropna=False, as_index=False)["Outlet_Retail_Need"]
        .sum().rename(columns={"Outlet_Retail_Need": "Retail_Need_Total"})
    )
    return totals, per_outlet

def compute_retail_sales_from_spr(spr: pd.DataFrame) -> pd.DataFrame:
    code_col = COLS["spr"]["code"]
    name_col = COLS["spr"]["name"]
    net_col  = COLS["spr"]["net"]
    if net_col not in spr.columns:
        raise ValueError(f"SPR missing '{net_col}' to compute sales.")
    grp_cols = []
    if code_col in spr.columns:
        grp_cols.append(code_col)
    grp_cols.append(name_col)
    sales = spr.groupby(grp_cols, dropna=False, as_index=False)[net_col].sum().rename(columns={net_col: "Recent_Sales"})
    ren = {}
    if code_col in spr.columns: ren[code_col] = "Retail_Item_Code"
    ren[name_col] = "Retail_Item_Name"
    sales = sales.rename(columns=ren)
    sales["Recent_Sales"] = pd.to_numeric(sales["Recent_Sales"], errors="coerce").fillna(0)
    return sales

def get_retail_stock_in_wh(ws_agg: pd.DataFrame, retail_totals: pd.DataFrame) -> pd.DataFrame:
    if "Retail_Item_Code" in retail_totals.columns and "Item Code" in ws_agg.columns:
        ws_retail = ws_agg.rename(columns={"Item Code": "Retail_Item_Code",
                                           "Item Name": "Retail_Item_Name",
                                           "Warehouse_Qty": "Retail_Stock_in_WH"})
        ws_retail = ws_retail[["Retail_Item_Code", "Retail_Item_Name", "Retail_Stock_in_WH"]]
        ws_retail = ws_retail[ws_retail["Retail_Item_Code"].notna()]
    else:
        if "Item Name" not in ws_agg.columns:
            raise ValueError("WS missing 'Item Name' for name-based join.")
        ws_retail = ws_agg.rename(columns={"Item Name": "Retail_Item_Name", "Warehouse_Qty": "Retail_Stock_in_WH"})
        ws_retail = ws_retail[["Retail_Item_Name", "Retail_Stock_in_WH"]]
    ws_retail["Retail_Stock_in_WH"] = pd.to_numeric(ws_retail["Retail_Stock_in_WH"], errors="coerce").fillna(0)
    return ws_retail

def build_schedule_retail_level_sales_based(ws_agg: pd.DataFrame,
                                            retail_totals: pd.DataFrame,
                                            ws_retail_wh: pd.DataFrame,
                                            mapping: pd.DataFrame,
                                            retail_sales: pd.DataFrame) -> pd.DataFrame:
    m = mapping.rename(columns={
        COLS["map"]["bulk_code"]: "Bulk_Item_Code",
        COLS["map"]["bulk_name"]: "Bulk_Item_Name",
        COLS["map"]["ret_code"]:  "Retail_Item_Code",
        COLS["map"]["ret_name"]:  "Retail_Item_Name",
        COLS["map"]["conv"]:      "Unit_Factor_BulkPerRetail",
    }).copy()
    m["Unit_Factor_BulkPerRetail"] = pd.to_numeric(m["Unit_Factor_BulkPerRetail"], errors="coerce").fillna(1.0)

    need = retail_totals.copy()
    dm = need.merge(ws_retail_wh, on=[c for c in ["Retail_Item_Code", "Retail_Item_Name"] if c in need.columns], how="left")
    dm["Retail_Stock_in_WH"] = dm.get("Retail_Stock_in_WH", 0).fillna(0)
    dm["Retail_Shortfall"] = (dm["Retail_Need_Total"] - dm["Retail_Stock_in_WH"]).clip(lower=0)

    key_cols = [c for c in ["Retail_Item_Code", "Retail_Item_Name"] if c in dm.columns]
    dm = dm.merge(m, on=key_cols, how="left")
    dm = dm.merge(retail_sales, on=[c for c in ["Retail_Item_Code", "Retail_Item_Name"] if c in dm.columns], how="left")
    dm["Recent_Sales"] = pd.to_numeric(dm.get("Recent_Sales", 0), errors="coerce").fillna(0)

    unmapped = dm[dm["Bulk_Item_Code"].isna()]
    if not unmapped.empty:
        st.warning(f"{len(unmapped)} retail items lack bulk mapping and are excluded from repack.")
    dm = dm.dropna(subset=["Bulk_Item_Code"]).copy()

    dm["Bulk_Units_Required"] = (dm["Retail_Shortfall"] * dm["Unit_Factor_BulkPerRetail"]).astype(float)

    if "Item Code" in ws_agg.columns:
        ws_bulk = ws_agg.rename(columns={
            "Item Code": "Bulk_Item_Code",
            "Item Name": "Bulk_Item_Name",
            "Warehouse_Qty": "Bulk_Available_in_WH"
        })[["Bulk_Item_Code", "Bulk_Item_Name", "Bulk_Available_in_WH"]].drop_duplicates()
    else:
        ws_bulk = ws_agg.rename(columns={
            "Item Name": "Bulk_Item_Name",
            "Warehouse_Qty": "Bulk_Available_in_WH"
        })[["Bulk_Item_Name", "Bulk_Available_in_WH"]].drop_duplicates()

    dm = dm.merge(ws_bulk, on=[c for c in ["Bulk_Item_Code", "Bulk_Item_Name"] if c in dm.columns], how="left")
    dm["Bulk_Available_in_WH"] = pd.to_numeric(dm["Bulk_Available_in_WH"], errors="coerce").fillna(0)

    def allocate_group(g: pd.DataFrame) -> pd.DataFrame:
        avail = float(g["Bulk_Available_in_WH"].iloc[0])
        need_bulk = g["Bulk_Units_Required"].to_numpy(dtype=float)
        sales = g["Recent_Sales"].to_numpy(dtype=float)
        total_needed = float(np.nansum(need_bulk))
        if avail <= 0 or total_needed <= 0:
            g["Bulk_To_Consume"] = 0.0
            return g
        sales_sum = float(np.nansum(sales))
        weights = (sales / sales_sum) if sales_sum > 0 else (need_bulk / total_needed)
        alloc = np.minimum(weights * avail, need_bulk)
        for _ in range(5):
            leftover = avail - float(np.nansum(alloc))
            if leftover <= 1e-9:
                break
            room = need_bulk - alloc
            mask = room > 1e-9
            if not np.any(mask):
                break
            w = np.where(mask, weights, 0.0)
            w_sum = float(np.nansum(w))
            extra = (leftover / np.sum(mask)) * np.where(mask, 1.0, 0.0) if w_sum <= 0 else leftover * (w / w_sum)
            alloc = np.minimum(alloc + extra, need_bulk)
        g["Bulk_To_Consume"] = np.ceil(alloc).astype(float)
        return g

    group_keys = [c for c in ["Bulk_Item_Code", "Bulk_Item_Name"] if c in dm.columns]
    dm = dm.groupby(group_keys, group_keys=False, dropna=False).apply(allocate_group)

    dm["Retail_To_Repack"] = (dm["Bulk_To_Consume"] / dm["Unit_Factor_BulkPerRetail"]).round(0)
    dm["Retail_To_Repack"] = dm[["Retail_To_Repack", "Retail_Shortfall"]].min(axis=1)
    dm["Retail_Unmet"] = (dm["Retail_Shortfall"] - dm["Retail_To_Repack"]).clip(lower=0)

    cols = [c for c in [
        "Retail_Item_Code", "Retail_Item_Name",
        "Bulk_Item_Code", "Bulk_Item_Name",
        "Recent_Sales",
        "Retail_Need_Total", "Retail_Stock_in_WH", "Retail_Shortfall",
        "Unit_Factor_BulkPerRetail",
        "Bulk_Available_in_WH", "Bulk_Units_Required", "Bulk_To_Consume",
        "Retail_To_Repack", "Retail_Unmet"
    ] if c in dm.columns]
    dm = dm[cols].sort_values(["Retail_Shortfall", "Recent_Sales", "Retail_Item_Name"],
                               ascending=[False, False, True]).reset_index(drop=True)
    return dm

def run_pipeline(ws_file, spr_file, map_file):
    ws_df  = read_table(ws_file, "ws")
    spr_df = read_table(spr_file, "spr")
    map_df = read_table(map_file, "map")

    ws_agg, (spr, _) = load_ws(ws_df), load_spr(spr_df)
    retail_totals, per_outlet = compute_retail_needs(spr)
    ws_retail_wh = get_retail_stock_in_wh(ws_agg, retail_totals)
    retail_sales = compute_retail_sales_from_spr(spr)
    schedule = build_schedule_retail_level_sales_based(ws_agg, retail_totals, ws_retail_wh, map_df, retail_sales)

    # Simplified shortlist sheet
    shortlist = schedule.rename(columns={
        "Retail_Item_Code": "Item Code",
        "Retail_Item_Name": "Item Name",
        "Retail_To_Repack": "Required"
    })[["Item Code", "Item Name", "Required"]].sort_values("Required", ascending=False)

    # Write to in-memory xlsx
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as xlw:
        schedule.to_excel(xlw, index=False, sheet_name="Repack_Schedule")
        per_outlet.to_excel(xlw, index=False, sheet_name="Outlet_Retail_Need")
        retail_totals.to_excel(xlw, index=False, sheet_name="Retail_Need_Total")
        ws_agg.to_excel(xlw, index=False, sheet_name="WS_Aggregated")
        shortlist.to_excel(xlw, index=False, sheet_name="Repack_Shortlist")
    bio.seek(0)
    return bio, schedule.shape[0], shortlist.shape[0]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Repacking Planner", layout="centered")
st.title("Repacking Schedule Generator")

st.markdown("Upload **WS.xls**, **SPR.xls**, and **bulk_to_retail_map.xlsx** and download the plan.")

c1, c2, c3 = st.columns(3)
with c1:
    ws_file = st.file_uploader("WS (warehouse stock)", type=["xls","xlsx"], key="ws")
with c2:
    spr_file = st.file_uploader("SPR (outlet sales & stock)", type=["xls","xlsx"], key="spr")
with c3:
    map_file = st.file_uploader("Mapping (bulk→retail)", type=["xlsx","xls"], key="map")

btn = st.button("Generate Plan", type="primary", disabled=not (ws_file and spr_file and map_file))

if btn:
    try:
        output_bytes, n_sched, n_short = run_pipeline(ws_file, spr_file, map_file)
        st.success(f"Generated plan ✅  Repack_Schedule rows: {n_sched} | Repack_Shortlist rows: {n_short}")
        st.download_button(
            label="Download repacking_plan.xlsx",
            data=output_bytes,
            file_name="repacking_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
