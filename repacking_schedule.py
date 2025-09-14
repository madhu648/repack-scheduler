import math
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
FILES = {
    "ws": "WS.xls",                     # warehouse stocks (.xls in production)
    "spr": "SPR.xls",                   # outlet sales & stocks (.xls in production)
    "map": "bulk_to_retail_map.xlsx",   # mapping (xlsx or xls)
}

# Preferred header rows (0-based) seen in your samples; auto-detect still runs.
PREFERRED_HEADER_ROW = {"ws": 3, "spr": 3, "map": 0}
# Count stockout only by Current Stock (ignore demand)?
STOCKOUT_REQUIRES_DEMAND = False  # set True if you want "stockout with demand"

EXPECTED = {
    "ws":  ["Item Code", "Item Name", "ML"],
    "spr": ["Item Name", "Net Sales Qty", "Outlet Name", "Current Stock"],  # Item Code may be absent
    "map": ["Bulk Item Code", "Bulk Item Name", "Retail Item Code", "Retail Item Name",
            "Unit Factor (bulk units per 1 retail unit)"],
}

COLS = {
    "ws": {"code": "Item Code", "name": "Item Name", "qty": "ML"},
    "spr": {
        "code":   "Item Code",        # may not exist; we fallback to name
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
        "conv":     "Unit Factor (bulk units per 1 retail unit)",  # bulk units per 1 retail unit
    },
}

# Heuristics to filter warehouse rows in SPR
WAREHOUSE_KEYWORDS = ["warehouse", "wh", "godown", "dc", "central"]

# Priority settings (used for coverage → daily sales)
SALES_WINDOW_DAYS = 14          # recent sales window used to compute Daily_Sales

# ============================================================
# Helpers: engine selection, header detection, robust reading
# ============================================================
def _engine_for(path: str) -> str | None:
    suf = Path(path).suffix.lower()
    if suf == ".xlsx":
        return "openpyxl"
    if suf == ".xls":
        return "xlrd"   # requires xlrd<2.0
    return None

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum() or ch.isspace())

def _detect_header_row(path: str, expected_cols: list[str], prefer: int | None) -> int:
    engine = _engine_for(path)
    sample = pd.read_excel(path, header=None, nrows=15, engine=engine)
    want = [_norm(c) for c in expected_cols]

    def row_score(r: int) -> int:
        vals = sample.iloc[r].tolist()
        row_norm = [_norm(v) for v in vals]
        return sum(1 for w in want if w in row_norm)

    if prefer is not None and 0 <= prefer < sample.shape[0]:
        if row_score(prefer) >= max(2, min(3, len(want))):
            return prefer

    best_row, best_score = None, -1
    for r in range(sample.shape[0]):
        sc = row_score(r)
        if sc > best_score:
            best_row, best_score = r, sc
    return best_row if best_row is not None else (prefer if prefer is not None else 3)

def read_table(kind: str, path: str) -> pd.DataFrame:
    engine = _engine_for(path)
    if engine is None:
        raise ValueError(f"Unsupported file type: {path}")
    header = _detect_header_row(path, EXPECTED[kind], PREFERRED_HEADER_ROW.get(kind))
    df = pd.read_excel(path, header=header, engine=engine)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ============================================================
# Loaders
# ============================================================
def load_ws(path_ws: str) -> pd.DataFrame:
    ws = read_table("ws", path_ws)
    keep = [c for c in [COLS["ws"]["code"], COLS["ws"]["name"], COLS["ws"]["qty"]] if c in ws.columns]
    ws = ws[keep].copy()
    ws[COLS["ws"]["qty"]] = pd.to_numeric(ws[COLS["ws"]["qty"]], errors="coerce").fillna(0)

    # Aggregate duplicates by code if present, else by name
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

def load_spr(path_spr: str) -> tuple[pd.DataFrame, bool]:
    spr = read_table("spr", path_spr)
    keep = [c for c in [COLS["spr"]["code"], COLS["spr"]["name"], COLS["spr"]["outlet"], COLS["spr"]["net"], COLS["spr"]["stock"]] if c in spr.columns]
    spr = spr[keep].copy()

    # Numerics
    if COLS["spr"]["net"] in spr.columns:
        spr[COLS["spr"]["net"]] = pd.to_numeric(spr[COLS["spr"]["net"]], errors="coerce").fillna(0)
    if COLS["spr"]["stock"] in spr.columns:
        spr[COLS["spr"]["stock"]] = pd.to_numeric(spr[COLS["spr"]["stock"]], errors="coerce").fillna(0)

    # Filter warehouse lines when Outlet Name available
    has_outlet = COLS["spr"]["outlet"] in spr.columns
    if has_outlet:
        oc = COLS["spr"]["outlet"]
        def is_wh(x):
            s = str(x).strip().lower()
            return any(k in s for k in WAREHOUSE_KEYWORDS)
        spr = spr[~spr[oc].astype(str).map(is_wh)].copy()

    # Per-row outlet need
    spr["Outlet_Retail_Need"] = (spr.get(COLS["spr"]["net"], 0) - spr.get(COLS["spr"]["stock"], 0)).clip(lower=0)
    return spr, (COLS["spr"]["code"] in spr.columns)

def load_map(path_map: str) -> pd.DataFrame:
    m = read_table("map", path_map)
    need = [COLS["map"]["bulk_code"], COLS["map"]["bulk_name"], COLS["map"]["ret_code"], COLS["map"]["ret_name"], COLS["map"]["conv"]]
    keep = [c for c in need if c in m.columns]
    m = m[keep].copy()
    if COLS["map"]["conv"] not in m.columns:
        m[COLS["map"]["conv"]] = 1.0
    m[COLS["map"]["conv"]] = pd.to_numeric(m[COLS["map"]["conv"]], errors="coerce").fillna(1.0)
    keys = [c for c in [COLS["map"]["bulk_code"], COLS["map"]["ret_code"]] if c in m.columns]
    if keys:
        m = m.drop_duplicates(subset=keys, keep="first")
    return m

# ============================================================
# Derived tables
# ============================================================
def compute_retail_needs(spr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # per-outlet group
    group_cols = []
    if COLS["spr"]["code"] in spr.columns:
        group_cols.append(COLS["spr"]["code"])
    group_cols.append(COLS["spr"]["name"])
    if COLS["spr"]["outlet"] in spr.columns:
        group_cols.append(COLS["spr"]["outlet"])

    per_outlet = spr.groupby(group_cols, dropna=False, as_index=False)["Outlet_Retail_Need"].sum()

    rename_map = {}
    if COLS["spr"]["code"] in spr.columns:
        rename_map[COLS["spr"]["code"]] = "Retail_Item_Code"
    rename_map[COLS["spr"]["name"]] = "Retail_Item_Name"
    if COLS["spr"]["outlet"] in spr.columns:
        rename_map[COLS["spr"]["outlet"]] = "Outlet"
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
    if code_col in spr.columns:
        ren[code_col] = "Retail_Item_Code"
    ren[name_col] = "Retail_Item_Name"
    sales = sales.rename(columns=ren)
    sales["Recent_Sales"] = pd.to_numeric(sales["Recent_Sales"], errors="coerce").fillna(0)
    return sales

def compute_total_outlet_current_stock(spr: pd.DataFrame) -> pd.DataFrame:
    """Sum of outlets' current stock per retail SKU (ignores warehouse rows)."""
    grp_cols = []
    if COLS["spr"]["code"] in spr.columns:
        grp_cols.append(COLS["spr"]["code"])
    grp_cols.append(COLS["spr"]["name"])
    tot = spr.groupby(grp_cols, dropna=False, as_index=False)[COLS["spr"]["stock"]].sum().rename(
        columns={COLS["spr"]["stock"]: "Total_Outlet_Current_Stock"}
    )
    ren = {}
    if COLS["spr"]["code"] in spr.columns:
        ren[COLS["spr"]["code"]] = "Retail_Item_Code"
    ren[COLS["spr"]["name"]] = "Retail_Item_Name"
    return tot.rename(columns=ren)

# Optional switch near your config block:
STOCKOUT_REQUIRES_DEMAND = False  # set True to require Net Sales > 0 as well

# Optional switch near config:
STOCKOUT_REQUIRES_DEMAND = False  # True = require demand as well

def compute_outlet_stockouts(spr: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per retail SKU the number of DISTINCT outlets in stockout.
    Stockout per outlet is decided AFTER aggregating all rows for that outlet:
      - max(Current Stock) <= 0  => outlet is stocked-out
      - If STOCKOUT_REQUIRES_DEMAND=True, also require sum(Net Sales Qty) > 0
    """
    df = spr.copy()

    # Ensure numeric
    df[COLS["spr"]["stock"]] = pd.to_numeric(df[COLS["spr"]["stock"]], errors="coerce").fillna(0)
    if STOCKOUT_REQUIRES_DEMAND:
        df[COLS["spr"]["net"]] = pd.to_numeric(df.get(COLS["spr"]["net"], 0), errors="coerce").fillna(0)

    # Keys
    item_keys = []
    if COLS["spr"]["code"] in df.columns:
        item_keys.append(COLS["spr"]["code"])
    item_keys.append(COLS["spr"]["name"])

    has_outlet = COLS["spr"]["outlet"] in df.columns
    if has_outlet:
        # Normalize outlet id to avoid case/spacing dupes
        df[COLS["spr"]["outlet"]] = df[COLS["spr"]["outlet"]].astype(str).str.strip().str.lower()

        # Collapse to (item, outlet): max stock and (optional) demand per outlet
        agg_dict = {COLS["spr"]["stock"]: "max"}
        if STOCKOUT_REQUIRES_DEMAND:
            agg_dict[COLS["spr"]["net"]] = "sum"

        outlet_level = (
            df.groupby(item_keys + [COLS["spr"]["outlet"]], dropna=False)
              .agg(agg_dict).reset_index()
        )

        # Decide stockout at outlet-level
        outlet_level["__outlet_stockout__"] = outlet_level[COLS["spr"]["stock"]] <= 0
        if STOCKOUT_REQUIRES_DEMAND:
            outlet_level["__outlet_stockout__"] &= outlet_level[COLS["spr"]["net"]] > 0

        # Count outlets in stockout per item
        per_item = (
            outlet_level.groupby(item_keys, dropna=False)["__outlet_stockout__"]
                        .sum().astype(int).reset_index(name="Outlets_StockedOut_Now")
        )
    else:
        # No outlet column → can only check if ANY row is stockout for the item
        exists_any = (
            df.groupby(item_keys, dropna=False)[COLS["spr"]["stock"]].max().reset_index()
        )
        exists_any["Outlets_StockedOut_Now"] = (exists_any[COLS["spr"]["stock"]] <= 0).astype(int)
        per_item = exists_any[item_keys + ["Outlets_StockedOut_Now"]]

    # Rename to unified columns
    ren = {}
    if COLS["spr"]["code"] in df.columns:
        ren[COLS["spr"]["code"]] = "Retail_Item_Code"
    ren[COLS["spr"]["name"]] = "Retail_Item_Name"
    return per_item.rename(columns=ren)

def get_retail_stock_in_wh(ws_agg: pd.DataFrame, retail_totals: pd.DataFrame) -> pd.DataFrame:
    if "Retail_Item_Code" in retail_totals.columns and "Item Code" in ws_agg.columns:
        ws_retail = ws_agg.rename(columns={
            "Item Code": "Retail_Item_Code",
            "Item Name": "Retail_Item_Name",
            "Warehouse_Qty": "Retail_Stock_in_WH",
        })
        ws_retail = ws_retail[["Retail_Item_Code", "Retail_Item_Name", "Retail_Stock_in_WH"]]
        ws_retail = ws_retail[ws_retail["Retail_Item_Code"].notna()]
    else:
        if "Item Name" not in ws_agg.columns:
            raise ValueError("Warehouse table missing 'Item Name' for name-based join.")
        ws_retail = ws_agg.rename(columns={
            "Item Name": "Retail_Item_Name",
            "Warehouse_Qty": "Retail_Stock_in_WH",
        })
        ws_retail = ws_retail[["Retail_Item_Name", "Retail_Stock_in_WH"]]

    ws_retail["Retail_Stock_in_WH"] = pd.to_numeric(ws_retail["Retail_Stock_in_WH"], errors="coerce").fillna(0)
    return ws_retail

# ============================================================
# Sales-based proportional allocation at retail level
# ============================================================
def build_schedule_retail_level_sales_based(
    ws_agg: pd.DataFrame,
    retail_totals: pd.DataFrame,
    ws_retail_wh: pd.DataFrame,
    mapping: pd.DataFrame,
    retail_sales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Output: one row per mapped retail item.
    When bulk is short, available bulk is allocated PROPORTIONALLY TO RECENT SALES.
    """
    # Normalize mapping
    m = mapping.rename(columns={
        COLS["map"]["bulk_code"]: "Bulk_Item_Code",
        COLS["map"]["bulk_name"]: "Bulk_Item_Name",
        COLS["map"]["ret_code"]:  "Retail_Item_Code",
        COLS["map"]["ret_name"]:  "Retail_Item_Name",
        COLS["map"]["conv"]:      "Unit_Factor_BulkPerRetail",   # bulk units needed per 1 retail unit
    }).copy()
    m["Unit_Factor_BulkPerRetail"] = pd.to_numeric(m["Unit_Factor_BulkPerRetail"], errors="coerce").fillna(1.0)

    # Need + retail WH stock
    need = retail_totals.copy()
    dm = need.merge(
        ws_retail_wh,
        on=[c for c in ["Retail_Item_Code", "Retail_Item_Name"] if c in need.columns],
        how="left",
    )
    dm["Retail_Stock_in_WH"] = dm.get("Retail_Stock_in_WH", 0).fillna(0)
    dm["Retail_Shortfall"] = (dm["Retail_Need_Total"] - dm["Retail_Stock_in_WH"]).clip(lower=0)

    # Join mapping (retail→bulk) and recent sales (weights)
    key_cols = [c for c in ["Retail_Item_Code", "Retail_Item_Name"] if c in dm.columns]
    dm = dm.merge(m, on=key_cols, how="left")
    dm = dm.merge(
        retail_sales,
        on=[c for c in ["Retail_Item_Code", "Retail_Item_Name"] if c in dm.columns],
        how="left",
    )
    dm["Recent_Sales"] = pd.to_numeric(dm.get("Recent_Sales", 0), errors="coerce").fillna(0)

    # Unmapped retail items cannot be repacked
    dm = dm.dropna(subset=["Bulk_Item_Code"]).copy()

    # Required bulk per retail
    dm["Unit_Factor_BulkPerRetail"] = dm["Unit_Factor_BulkPerRetail"].replace(0, 1.0)
    dm["Bulk_Units_Required"] = (dm["Retail_Shortfall"] * dm["Unit_Factor_BulkPerRetail"]).astype(float)

    # Bulk availability from WS
    if "Item Code" in ws_agg.columns:
        ws_bulk = ws_agg.rename(columns={
            "Item Code": "Bulk_Item_Code",
            "Item Name": "Bulk_Item_Name",
            "Warehouse_Qty": "Bulk_Available_in_WH",
        })[["Bulk_Item_Code", "Bulk_Item_Name", "Bulk_Available_in_WH"]].drop_duplicates()
    else:
        ws_bulk = ws_agg.rename(columns={
            "Item Name": "Bulk_Item_Name",
            "Warehouse_Qty": "Bulk_Available_in_WH",
        })[["Bulk_Item_Name", "Bulk_Available_in_WH"]].drop_duplicates()

    dm = dm.merge(ws_bulk, on=[c for c in ["Bulk_Item_Code", "Bulk_Item_Name"] if c in dm.columns], how="left")
    dm["Bulk_Available_in_WH"] = pd.to_numeric(dm["Bulk_Available_in_WH"], errors="coerce").fillna(0)

    # Allocation per bulk SKU (sales-proportional with caps)
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

        # Initial proportional allocation with capping
        alloc = np.minimum(weights * avail, need_bulk)

        # Redistribute leftover a few passes
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

        g["Bulk_To_Consume"] = np.ceil(alloc).astype(float)  # integer bulk units if needed
        return g

    group_keys = [c for c in ["Bulk_Item_Code", "Bulk_Item_Name"] if c in dm.columns]
    dm = dm.groupby(group_keys, group_keys=False, dropna=False).apply(allocate_group)

    # Retail produced from allocated bulk (cap by retail shortfall)
    dm["Retail_To_Repack"] = (dm["Bulk_To_Consume"] / dm["Unit_Factor_BulkPerRetail"]).round(0)
    dm["Retail_To_Repack"] = dm[["Retail_To_Repack", "Retail_Shortfall"]].min(axis=1)

    # Unmet retail
    dm["Retail_Unmet"] = (dm["Retail_Shortfall"] - dm["Retail_To_Repack"]).clip(lower=0)

    # Final column order (one row per retail item)
    cols = [c for c in [
        "Retail_Item_Code", "Retail_Item_Name",
        "Bulk_Item_Code", "Bulk_Item_Name",
        "Recent_Sales",
        "Retail_Need_Total", "Retail_Stock_in_WH", "Retail_Shortfall",
        "Unit_Factor_BulkPerRetail",
        "Bulk_Available_in_WH", "Bulk_Units_Required", "Bulk_To_Consume",
        "Retail_To_Repack", "Retail_Unmet",
    ] if c in dm.columns]

    dm = dm[cols].sort_values(
        ["Retail_Shortfall", "Recent_Sales", "Retail_Item_Name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return dm

# ============================================================
# Priority column (integer, dense increasing)
# ============================================================
def add_priority(schedule_df: pd.DataFrame, spr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unique Priority (1..N), strictly following:
      1) Items with any outlet stockout come first (Has_Stockouts=1 before 0)
      2) Within stockouts: Outlets_StockedOut_Now (desc)
      3) Within non-stockouts: Coverage_Days (asc; inf last)
      4) Tie-breakers everywhere: Retail_Shortfall (desc), Recent_Sales (desc),
         Retail_Item_Code (asc), Retail_Item_Name (asc)
    """
    df = schedule_df.copy()

    # Merge outlet totals and stockout counts
    tot_outlet_stock  = compute_total_outlet_current_stock(spr_df)
    outlet_stockouts  = compute_outlet_stockouts(spr_df)
    key_cols = [c for c in ["Retail_Item_Code", "Retail_Item_Name"] if c in df.columns]
    df = df.merge(tot_outlet_stock, on=key_cols, how="left")
    df = df.merge(outlet_stockouts, on=key_cols, how="left")

    # Ensure fields exist & numeric
    for col, default in [
        ("Total_Outlet_Current_Stock", 0),
        ("Outlets_StockedOut_Now",     0),
        ("Recent_Sales",               0),
        ("Retail_Shortfall",           0),
        ("Retail_Stock_in_WH",         0),
    ]:
        df[col] = pd.to_numeric(df.get(col, default), errors="coerce").fillna(default)

    if "Retail_Item_Code" not in df.columns: df["Retail_Item_Code"] = ""
    if "Retail_Item_Name" not in df.columns: df["Retail_Item_Name"] = ""

    # Coverage computation (safe)
    SALES_DAYS = max(SALES_WINDOW_DAYS, 1)
    df["Daily_Sales"] = df["Recent_Sales"] / SALES_DAYS
    total_stock = df["Retail_Stock_in_WH"] + df["Total_Outlet_Current_Stock"]

    cov_vals = []
    for s, d in zip(total_stock.to_numpy(float), df["Daily_Sales"].to_numpy(float)):
        if d <= 0:
            cov_vals.append(float("inf") if s > 0 else 0.0)
        else:
            cov_vals.append(s / d)
    df["Coverage_Days"] = cov_vals
    df["Coverage_Sort"] = df["Coverage_Days"].replace([np.inf, -np.inf], np.inf)

    # Explicit stockout flag
    df["Has_Stockouts"] = (df["Outlets_StockedOut_Now"] > 0).astype(int)

    # Build a single **global** order that encodes your algorithm
    # Note: For non-stockouts, Outlets_StockedOut_Now is 0 for all; they will then be ordered by Coverage_Sort.
    df = df.sort_values(
        by=[
            "Has_Stockouts",          # 1 first (True), then 0
            "Outlets_StockedOut_Now", # more stockouts first
            "Coverage_Sort",          # lower coverage first (inf last)
            "Retail_Shortfall",       # bigger shortfall first
            "Recent_Sales",           # higher sales first
            "Retail_Item_Code",       # stable
            "Retail_Item_Name",       # stable
        ],
        ascending=[False,             False,                 True,               False,              False,            True,               True],
        kind="mergesort",  # stable sort
    ).reset_index(drop=True)

    # Unique integer priority 1..N
    df["Priority"] = np.arange(1, len(df) + 1, dtype=int)

    # Cleanup helper
    df = df.drop(columns=["Coverage_Sort"])
    return df

# ============================================================
# Pipeline
# ============================================================
def generate_repacking_plan(ws_file=FILES["ws"], spr_file=FILES["spr"], map_file=FILES["map"]):
    ws_agg = load_ws(ws_file)
    spr, _ = load_spr(spr_file)
    mapping = load_map(map_file)

    retail_totals, per_outlet = compute_retail_needs(spr)
    ws_retail_wh = get_retail_stock_in_wh(ws_agg, retail_totals)
    retail_sales = compute_retail_sales_from_spr(spr)  # weights

    # Build schedule (retail-level rows)
    schedule_retail = build_schedule_retail_level_sales_based(
        ws_agg, retail_totals, ws_retail_wh, mapping, retail_sales
    )

    # Add Priority column
    schedule_retail = add_priority(schedule_retail, spr)

    return {
        "schedule_retail_level": schedule_retail,   # includes Priority
        "retail_need_per_outlet": per_outlet,
        "retail_need_total": retail_totals,
        "ws_aggregated": ws_agg,
    }

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    out = generate_repacking_plan()
    out_path = Path("repacking_plan.xlsx")

    sched = out["schedule_retail_level"]  # includes Priority

    # Shortlist: only actionable, Priority as last column
    shortlist = (
        sched.rename(columns={
            "Retail_Item_Code": "Item Code",
            "Retail_Item_Name": "Item Name",
            "Retail_To_Repack": "Required",
        })[["Item Code", "Item Name", "Required", "Priority"]]
    )
    shortlist["Required"] = pd.to_numeric(shortlist["Required"], errors="coerce").fillna(0)
    shortlist = shortlist[shortlist["Required"] > 0].sort_values(
        ["Priority", "Required"], ascending=[True, False]
    )

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xlw:
        sched.to_excel(xlw, index=False, sheet_name="Repack_Schedule")
        out["retail_need_per_outlet"].to_excel(xlw, index=False, sheet_name="Outlet_Retail_Need")
        out["retail_need_total"].to_excel(xlw, index=False, sheet_name="Retail_Need_Total")
        out["ws_aggregated"].to_excel(xlw, index=False, sheet_name="WS_Aggregated")
        shortlist.to_excel(xlw, index=False, sheet_name="Repack_Shortlist")

        for sheet in ["Repack_Schedule","Outlet_Retail_Need","Retail_Need_Total","WS_Aggregated","Repack_Shortlist"]:
            xlw.sheets[sheet].set_column(0, 30, 18)

    print(f"Written: {out_path.resolve()}")
