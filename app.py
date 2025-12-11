# app.py - Antenna Cloud Demo 3 (Fully Loaded) — UPDATED
# Added: interactive axis controls (X/Y sliders) + Auto-scale toggle for Viewer and Compare modes.

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config("Antenna Cloud Demo3", layout="wide", initial_sidebar_state="expanded")

# ---------- Utilities ----------
def read_sparam_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Normalize column names (strip)
    colmap = {c: c.strip() for c in df.columns}
    df.rename(columns=colmap, inplace=True)
    # try to find frequency column
    freq_cols = [c for c in df.columns if "freq" in c.lower() or "frequency" in c.lower()]
    if not freq_cols:
        # also accept first column as freq
        freq_col = df.columns[0]
    else:
        freq_col = freq_cols[0]
    df = df.rename(columns={freq_col: "frequency_Hz"})
    df["frequency_Hz"] = pd.to_numeric(df["frequency_Hz"], errors="coerce")
    # find s-params columns and standardize names
    for expected in ["s11", "s21", "s12", "s22"]:
        candidates = [c for c in df.columns if expected in c.lower().replace(" ", "")]
        if candidates:
            df[expected.upper() + "_dB"] = pd.to_numeric(df[candidates[0]], errors="coerce")
    return df

def ensure_db_cols(df):
    for p in ["S11", "S21", "S12", "S22"]:
        col = p + "_dB"
        if col not in df.columns:
            if p in df.columns:
                df[col] = 20 * np.log10(np.abs(df[p].astype(float)))
    return df

def magdb_to_linear(db):
    return 10 ** (db/20.0)

def compute_vswr_from_S11_db(s11_db):
    gamma = magdb_to_linear(s11_db)
    gamma_clamped = np.minimum(np.maximum(gamma, 0.0), 0.999999)
    vswr = (1+gamma_clamped) / (1-gamma_clamped)
    return vswr

def find_resonant_and_bw(df):
    out = {}
    if "S11_dB" not in df.columns:
        return out
    s11 = df["S11_dB"].to_numpy()
    freq = df["frequency_Hz"].to_numpy()
    idx_min = np.nanargmin(s11)
    out["resonant_freq_Hz"] = float(freq[idx_min])
    out["resonant_S11_dB"] = float(s11[idx_min])
    mask = s11 < -10
    bw_ranges = []
    if mask.any():
        i = 0
        n = len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j+1 < n and mask[j+1]:
                    j += 1
                bw_ranges.append((freq[i], freq[j]))
                i = j+1
            else:
                i += 1
    out["bandwidth_ranges_Hz"] = bw_ranges
    return out

def prepare_summary(df):
    df = ensure_db_cols(df)
    summary = {}
    if "S11_dB" in df.columns:
        metrics = find_resonant_and_bw(df)
        summary.update(metrics)
        summary["min_S11_dB"] = float(df["S11_dB"].min())
        summary["max_S21_dB"] = float(df["S21_dB"].max()) if "S21_dB" in df.columns else None
        summary["vswr_mean"] = float(compute_vswr_from_S11_db(df["S11_dB"]).mean())
    summary["n_points"] = int(len(df))
    return summary

def make_plot_sparams(df, title="S-Parameters"):
    df = ensure_db_cols(df)
    freq = df["frequency_Hz"]
    traces = []
    for p in ["S11_dB", "S21_dB", "S12_dB", "S22_dB"]:
        if p in df.columns:
            traces.append(go.Scatter(x=freq/1e9, y=df[p], mode="lines", name=p))
    fig = go.Figure(traces)
    fig.update_layout(title=title, xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)", template="plotly_white")
    return fig

def make_vswr_plot(df, title="VSWR"):
    df = ensure_db_cols(df)
    if "S11_dB" not in df.columns:
        return None
    vswr = compute_vswr_from_S11_db(df["S11_dB"].to_numpy())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["frequency_Hz"]/1e9, y=vswr, mode="lines", name="VSWR"))
    fig.update_layout(title=title, xaxis_title="Frequency (GHz)", yaxis_title="VSWR", template="plotly_white")
    return fig

def export_summary_csv(summary: dict):
    buf = io.StringIO()
    pd.DataFrame([summary]).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def try_create_pdf_report(title, summary, figs):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception:
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height-50, title)
    c.setFont("Helvetica", 10)
    y = height - 80
    for k, v in summary.items():
        line = f"{k}: {v}"
        c.drawString(40, y, line)
        y -= 12
        if y < 100:
            c.showPage()
            y = height - 50
    for fig in figs:
        try:
            img_bytes = fig_to_png_bytes(fig)
            if img_bytes:
                img = ImageReader(io.BytesIO(img_bytes))
                iw, ih = img.getSize()
                scale = min((width-80)/iw, (height-200)/ih)
                w_img = iw*scale
                h_img = ih*scale
                if y - h_img < 50:
                    c.showPage()
                    y = height - 50
                c.drawImage(img, 40, y-h_img, width=w_img, height=h_img)
                y = y - h_img - 20
        except Exception:
            continue
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

def fig_to_png_bytes(fig):
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", width=1200, height=700, scale=1)
        return png_bytes
    except Exception:
        return None

# ---------- Sidebar navigation ----------
st.sidebar.title("Antenna Cloud Demo3")
mode = st.sidebar.selectbox("Choose mode", ["S-Parameter Viewer", "Compare Antennas", "3D Radiation Pattern", "Summary / Report", "About"])

if "last_df" not in st.session_state:
    st.session_state["last_df"] = None
if "last_summary" not in st.session_state:
    st.session_state["last_summary"] = None
if "last_figs" not in st.session_state:
    st.session_state["last_figs"] = {}

# ---------- Mode 1: S-Parameter Viewer ----------
if mode == "S-Parameter Viewer":
    st.header("📈 S-Parameter Viewer & Sweep Tool")
    st.write("Upload a CSV containing frequency and S-parameters (S11, S21, S12, S22). Column names can contain 'freq' and 's11' etc.")

    uploaded_file = st.file_uploader("Upload S-parameter CSV file", type=["csv"], key="viewer_file")
    if uploaded_file is not None:
        df = read_sparam_csv(uploaded_file)
        df = ensure_db_cols(df)
        st.session_state["last_df"] = df
        st.write("Preview:")
        st.dataframe(df.head())

        # plotting controls
        st.subheader("Plot settings")
        db_columns = [c for c in df.columns if c.endswith("_dB")]
        show_sparams = st.multiselect("Select parameters to show (dB)", db_columns, default=db_columns[:2])

        # Axis controls
        st.subheader("Axis range controls")
        auto_scale = st.checkbox("Auto-scale axes (use Plotly autoscale)", value=True)
        # compute sensible defaults
        freq_ghz = df["frequency_Hz"].to_numpy() / 1e9
        f_min_default = float(np.nanmin(freq_ghz))
        f_max_default = float(np.nanmax(freq_ghz))
        if db_columns:
            all_db = np.hstack([df[c].dropna().to_numpy() for c in db_columns])
            y_default_min = float(np.nanmin(all_db)) - 5.0
            y_default_max = float(np.nanmax(all_db)) + 2.0
        else:
            y_default_min, y_default_max = -60.0, 5.0

        col_a, col_b = st.columns([2,2])
        with col_a:
            x_min, x_max = st.slider(
                "X-axis range (GHz)",
                min_value=0.0,
                max_value=max(5.0, f_max_default),
                value=(max(0.0, f_min_default), max(f_max_default, min(5.0, f_max_default))),
                step=0.1
            )
        with col_b:
            y_min, y_max = st.slider(
                "Y-axis range (dB)",
                min_value=-100.0,
                max_value=+20.0,
                value=(y_default_min, y_default_max),
                step=0.5
            )

        if show_sparams:
            fig = go.Figure()
            for p in show_sparams:
                if p in df.columns:
                    fig.add_trace(go.Scatter(x=df["frequency_Hz"]/1e9, y=df[p], name=p, mode="lines"))
            # apply axis ranges if not autoscale
            layout_kwargs = dict(title="S-Parameters", xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)", template="plotly_white")
            if not auto_scale:
                layout_kwargs["xaxis"] = dict(range=[x_min, x_max])
                layout_kwargs["yaxis"] = dict(range=[y_min, y_max])
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)
            st.session_state["last_figs"]["sparams"] = fig

        # VSWR plot
        st.subheader("VSWR")
        vswr_fig = make_vswr_plot(df)
        if vswr_fig:
            if not auto_scale:
                # apply same x-range to vswr
                vswr_fig.update_layout(xaxis=dict(range=[x_min, x_max]))
            st.plotly_chart(vswr_fig, use_container_width=True)
            st.session_state["last_figs"]["vswr"] = vswr_fig

        # summary metrics
        st.subheader("Auto Metrics")
        summary = prepare_summary(df)
        st.session_state["last_summary"] = summary
        st.json(summary)

        # downloads
        st.subheader("Download")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed CSV", csv_bytes, "processed_sparams.csv", "text/csv")
        summary_csv = export_summary_csv(summary)
        st.download_button("Download summary CSV", summary_csv, "summary.csv", "text/csv")

        st.write("Export plots as PNG (best-effort).")
        png_sparams = fig_to_png_bytes(st.session_state["last_figs"].get("sparams")) if st.session_state["last_figs"].get("sparams") else None
        png_vswr = fig_to_png_bytes(st.session_state["last_figs"].get("vswr")) if st.session_state["last_figs"].get("vswr") else None
        if png_sparams:
            st.download_button("Download S-params PNG", png_sparams, "sparams.png", "image/png")
        else:
            st.info("PNG export requires 'kaleido' in the environment. If you want PNG export, add 'kaleido' to requirements.txt.")

# ---------- Mode 2: Compare Antennas ----------
elif mode == "Compare Antennas":
    st.header("🔁 Antenna Comparison Tool")
    st.write("Upload two S-parameter CSV files to compare S11 and VSWR side-by-side.")
    col1, col2 = st.columns(2)
    file1 = col1.file_uploader("Upload file A", type=["csv"], key="cmp_a")
    file2 = col2.file_uploader("Upload file B", type=["csv"], key="cmp_b")

    # Axis controls for compare mode
    st.subheader("Comparison Axis Controls")
    auto_scale_cmp = st.checkbox("Auto-scale axes for comparison", value=True, key="auto_cmp")
    # We'll compute defaults only if files uploaded
    x_min_cmp = 0.0
    x_max_cmp = 3.0
    y_min_cmp = -60.0
    y_max_cmp = 5.0
    if file1 or file2:
        # estimate from whichever file is present
        tmp = None
        if file1:
            tmp = ensure_db_cols(read_sparam_csv(file1))
        elif file2:
            tmp = ensure_db_cols(read_sparam_csv(file2))
        if tmp is not None:
            freq_tmp = tmp["frequency_Hz"].to_numpy()/1e9
            x_min_cmp = float(np.nanmin(freq_tmp))
            x_max_cmp = float(np.nanmax(freq_tmp))
            db_cols_tmp = [c for c in tmp.columns if c.endswith("_dB")]
            if db_cols_tmp:
                all_db_tmp = np.hstack([tmp[c].dropna().to_numpy() for c in db_cols_tmp])
                y_min_cmp = float(np.nanmin(all_db_tmp)) - 5.0
                y_max_cmp = float(np.nanmax(all_db_tmp)) + 2.0

    col_c1, col_c2 = st.columns([2,2])
    with col_c1:
        x_min_c, x_max_c = st.slider("X-axis range (GHz) - Compare", min_value=0.0, max_value=max(5.0, x_max_cmp), value=(x_min_cmp, max(x_max_cmp, 3.0)), step=0.1)
    with col_c2:
        y_min_c, y_max_c = st.slider("Y-axis range (dB) - Compare", min_value=-100.0, max_value=20.0, value=(y_min_cmp, y_max_cmp), step=0.5)

    if file1 and file2:
        df1 = ensure_db_cols(read_sparam_csv(file1))
        df2 = ensure_db_cols(read_sparam_csv(file2))
        st.write("Preview A")
        st.dataframe(df1.head())
        st.write("Preview B")
        st.dataframe(df2.head())

        common_freq = np.linspace(max(df1["frequency_Hz"].min(), df2["frequency_Hz"].min()),
                                  min(df1["frequency_Hz"].max(), df2["frequency_Hz"].max()), 512)
        interp1 = {}
        interp2 = {}
        for col in ["S11_dB", "S21_dB", "S12_dB", "S22_dB"]:
            if col in df1.columns:
                interp1[col] = np.interp(common_freq, df1["frequency_Hz"], df1[col])
            if col in df2.columns:
                interp2[col] = np.interp(common_freq, df2["frequency_Hz"], df2[col])

        fig = go.Figure()
        if "S11_dB" in interp1:
            fig.add_trace(go.Scatter(x=common_freq/1e9, y=interp1["S11_dB"], name="A: S11"))
        if "S11_dB" in interp2:
            fig.add_trace(go.Scatter(x=common_freq/1e9, y=interp2["S11_dB"], name="B: S11"))
        layout_kwargs = dict(title="Comparison: S11 (A vs B)", xaxis_title="Freq (GHz)", yaxis_title="S11 (dB)", template="plotly_white")
        if not auto_scale_cmp:
            layout_kwargs["xaxis"] = dict(range=[x_min_c, x_max_c])
            layout_kwargs["yaxis"] = dict(range=[y_min_c, y_max_c])
        fig.update_layout(**layout_kwargs)
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["last_figs"]["compare_s11"] = fig

        vswr1 = compute_vswr_from_S11_db(interp1["S11_dB"]) if "S11_dB" in interp1 else None
        vswr2 = compute_vswr_from_S11_db(interp2["S11_dB"]) if "S11_dB" in interp2 else None
        fig2 = go.Figure()
        if vswr1 is not None:
            fig2.add_trace(go.Scatter(x=common_freq/1e9, y=vswr1, name="A: VSWR"))
        if vswr2 is not None:
            fig2.add_trace(go.Scatter(x=common_freq/1e9, y=vswr2, name="B: VSWR"))
        if not auto_scale_cmp:
            fig2.update_layout(xaxis=dict(range=[x_min_c, x_max_c]), yaxis=dict(range=[None, None]))
        fig2.update_layout(title="Comparison: VSWR (A vs B)", xaxis_title="Freq (GHz)", yaxis_title="VSWR", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
        st.session_state["last_figs"]["compare_vswr"] = fig2

        summary_a = prepare_summary(df1)
        summary_b = prepare_summary(df2)
        st.subheader("Summary Comparison")
        comp_df = pd.DataFrame({"Metric": list(summary_a.keys()),
                                "A": list(summary_a.values()),
                                "B": [summary_b.get(k, None) for k in summary_a.keys()]})
        st.dataframe(comp_df)

        with io.BytesIO() as mem_zip:
            with zipfile.ZipFile(mem_zip, mode="w") as zf:
                zf.writestr("summary_A.csv", pd.DataFrame([summary_a]).to_csv(index=False))
                zf.writestr("summary_B.csv", pd.DataFrame([summary_b]).to_csv(index=False))
                png1 = fig_to_png_bytes(st.session_state["last_figs"].get("compare_s11"))
                png2 = fig_to_png_bytes(st.session_state["last_figs"].get("compare_vswr"))
                if png1:
                    zf.writestr("compare_s11.png", png1)
                if png2:
                    zf.writestr("compare_vswr.png", png2)
            mem_zip.seek(0)
            st.download_button("Download comparison ZIP", mem_zip.read(), "comparison_results.zip", "application/zip")

# ---------- Mode 3: 3D Radiation Pattern ----------
elif mode == "3D Radiation Pattern":
    st.header("🌐 3D Radiation Pattern Viewer")
    st.write("Upload a radiation CSV with columns: theta, phi, gain_dBi  OR use a synthetic sample to preview.")

    rad_file = st.file_uploader("Upload radiation CSV (theta,phi,gain_dBi)", type=["csv"], key="rad_file")
    use_synthetic = st.checkbox("Use synthetic sample pattern (if no file uploaded)", value=True)

    if rad_file is not None:
        rad_df = pd.read_csv(rad_file)
        if not set(["theta", "phi", "gain_dbi"]).issubset({c.lower() for c in rad_df.columns}):
            st.warning("CSV should contain columns named theta, phi, gain_dBi (case-insensitive). Attempting to map...")
            lower_map = {c.lower(): c for c in rad_df.columns}
            if "theta" in lower_map and "phi" in lower_map and "gain_dbi" in lower_map:
                rad_df = rad_df.rename(columns={lower_map["theta"]:"theta", lower_map["phi"]:"phi", lower_map["gain_dbi"]:"gain_dBi"})
            else:
                st.error("Couldn't find required columns.")
                rad_df = None
    else:
        rad_df = None

    if rad_df is None and use_synthetic:
        st.info("Using synthetic radiation pattern.")
        thetas = np.linspace(0, np.pi, 61)
        phis = np.linspace(0, 2*np.pi, 73)
        TH, PH = np.meshgrid(thetas, phis)
        n = 8
        G = (np.cos(TH - np.pi/2) ** 2) * (1 / (1 + 0.2*(np.sin(PH*2))))
        G = np.maximum(G, 1e-4)
        Gd = 10 * np.log10(G / G.max()) + 8
        X = (1 + G/np.max(G)) * np.sin(TH) * np.cos(PH)
        Y = (1 + G/np.max(G)) * np.sin(TH) * np.sin(PH)
        Z = (1 + G/np.max(G)) * np.cos(TH)
        surf = go.Surface(x=X, y=Y, z=Z, surfacecolor=Gd, colorscale="Viridis")
        fig3d = go.Figure(data=[surf])
        fig3d.update_layout(title="Synthetic 3D Radiation Pattern (surfacecolor = gain dB)", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        st.plotly_chart(fig3d, use_container_width=True, height=700)
        st.session_state["last_figs"]["3d"] = fig3d
    elif rad_df is not None:
        rad_df = rad_df.rename(columns={c:c.strip() for c in rad_df.columns})
        lower_map = {c.lower():c for c in rad_df.columns}
        theta_col = lower_map.get("theta")
        phi_col = lower_map.get("phi")
        gain_col = lower_map.get("gain_dbi") or lower_map.get("gain_db") or lower_map.get("gain")
        rad_df = rad_df.rename(columns={theta_col:"theta", phi_col:"phi", gain_col:"gain_dBi"})
        thetas = np.unique(rad_df["theta"])
        phis = np.unique(rad_df["phi"])
        try:
            grid = rad_df.pivot(index="phi", columns="theta", values="gain_dBi")
            TH, PH = np.meshgrid(grid.columns*np.pi/180.0, grid.index*np.pi/180.0)
            R = 1 + (grid.values - np.nanmin(grid.values)) / (np.nanmax(grid.values) - np.nanmin(grid.values) + 1e-6)
            X = R * np.sin(TH) * np.cos(PH)
            Y = R * np.sin(TH) * np.sin(PH)
            Z = R * np.cos(TH)
            surf = go.Surface(x=X, y=Y, z=Z, surfacecolor=grid.values, colorscale="Viridis")
            fig3d = go.Figure(data=[surf])
            fig3d.update_layout(title="3D Radiation Pattern (from uploaded data)", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
            st.plotly_chart(fig3d, use_container_width=True, height=700)
            st.session_state["last_figs"]["3d"] = fig3d
        except Exception as e:
            st.error("Failed to convert uploaded radiation data to 3D surface. Make sure your CSV contains a regular theta/phi grid.")
            st.write(e)

# ---------- Mode 4: Summary / Report ----------
elif mode == "Summary / Report":
    st.header("🧾 Summary & Report Generation")
    st.write("Use the last loaded S-parameter dataset (from Viewer) or upload a file now to produce a report.")

    df_file = st.file_uploader("Upload S-parameter CSV for report (optional, uses last loaded if left empty)", type=["csv"], key="report_file")
    if df_file is None and st.session_state.get("last_df") is not None:
        df = st.session_state["last_df"]
    elif df_file is not None:
        df = ensure_db_cols(read_sparam_csv(df_file))
    else:
        df = None

    if df is not None:
        st.subheader("Computed Summary")
        summary = prepare_summary(df)
        st.json(summary)
        st.session_state["last_summary"] = summary

        fig_s = make_plot_sparams(df, title="S-Parameters for Report")
        fig_v = make_vswr_plot(df, title="VSWR for Report")
        figs = [f for f in [fig_s, fig_v] if f is not None]

        summary_csv = export_summary_csv(summary)
        st.download_button("Download summary CSV", summary_csv, "demo3_summary.csv", "text/csv")

        pdf_bytes = try_create_pdf_report("Antenna Report (Demo3)", summary, figs)
        if pdf_bytes:
            st.download_button("Download PDF Report", pdf_bytes, "antenna_report.pdf", "application/pdf")
        else:
            st.info("Automatic PDF creation requires 'reportlab' and 'kaleido' to embed images. Falling back to ZIP.")
            with io.BytesIO() as mem_zip:
                with zipfile.ZipFile(mem_zip, mode="w") as zf:
                    zf.writestr("summary.csv", summary_csv)
                    for i, f in enumerate(figs):
                        png = fig_to_png_bytes(f)
                        if png:
                            zf.writestr(f"figure_{i}.png", png)
                mem_zip.seek(0)
                st.download_button("Download ZIP (summary + figures)", mem_zip.read(), "report_assets.zip", "application/zip")

    else:
        st.info("No data available for report. Load a file in 'S-Parameter Viewer' first or upload one here.")

# ---------- Mode 5: About ----------
elif mode == "About":
    st.header("About Antenna Cloud Demo3")
    st.markdown("""
    **Antenna Cloud Demo3** — Fully loaded demonstration app.
    
    Features:
    - S-Parameter viewer + VSWR and metrics
    - Comparison tool for two datasets
    - 3D radiation pattern viewer (synthetic or from uploaded data)
    - Summary and report generation (CSV, ZIP; PDF if optional deps available)
    
    Notes:
    - For best results, upload CSVs with columns like:
      `frequency_Hz,S11_dB,S21_dB,S12_dB,S22_dB`
    - Radiation pattern CSV should contain `theta,phi,gain_dBi`.
    - Optional packages for full functionality:
      - `kaleido` (Plotly -> PNG)
      - `reportlab` (PDF creation)
    """)
