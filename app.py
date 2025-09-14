import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

# your module with generate_repacking_plan(...)
import repacking_schedule as rs

st.set_page_config(page_title="Repacking Planner", layout="centered")
st.title("📦 Repacking Planner")

st.markdown("Upload the three files and click **Generate**:")
c1, c2, c3 = st.columns(3)
with c1:
    ws_file = st.file_uploader("WS (.xls or .xlsx)", type=["xls","xlsx"], key="ws")
with c2:
    spr_file = st.file_uploader("SPR (.xls or .xlsx)", type=["xls","xlsx"], key="spr")
with c3:
    map_file = st.file_uploader("Bulk→Retail Map (.xlsx or .xls)", type=["xlsx","xls"], key="map")

def _save_to_tmp(uploaded, suffix):
    """Persist uploaded file to disk so pandas/xlrd/openpyxl can read it."""
    if not uploaded:
        return None
    data = uploaded.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name

st.divider()
if st.button("Generate", type="primary", disabled=not (ws_file and spr_file and map_file)):
    try:
        ws_path  = _save_to_tmp(ws_file,  Path(ws_file.name).suffix or ".xls")
        spr_path = _save_to_tmp(spr_file, Path(spr_file.name).suffix or ".xls")
        map_path = _save_to_tmp(map_file, Path(map_file.name).suffix or ".xlsx")

        with st.spinner("Crunching numbers…"):
            out = rs.generate_repacking_plan(ws_file=ws_path, spr_file=spr_path, map_file=map_path)

            # Build final workbook in memory
            out_path = Path(tempfile.gettempdir()) / "repacking_plan.xlsx"

            # Shortlist with Priority last & Required>0 (matches your latest request)
            sched = out["schedule_retail_level"]
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

        st.success("Done! Download your workbook below.")
        with open(out_path, "rb") as fh:
            st.download_button(
                label="⬇️ Download repacking_plan.xlsx",
                data=fh.read(),
                file_name="repacking_plan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception as e:
        st.error(f"Error: {e}")
