# app.py - Antenna Cloud Demo3 (Robust Reader + Inspector)
import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from pathlib import Path
from pandas.errors import EmptyDataError
import plotly.graph_objects as go

st.set_page_config("Antenna Cloud Demo3", layout="wide", initial_sidebar_state="expanded")

# ---------- Robust CSV reader ----------
def read_sparam_csv(file) -> pd.DataFrame:
    """
    Robust CSV reader for Streamlit UploadedFile or a local path.
    Returns DataFrame with 'frequency_Hz' and *_dB columns if present.
    Raises ValueError on failure with friendly message.
    """
    # quick size check (Streamlit UploadedFile has .size attr)
    try:
        size = getattr(file, "size", None)
    except Exception:
        size = None
    if size == 0:
        raise ValueError("Uploaded file is empty (0 bytes). Please re-upload a valid CSV.")

    # read raw bytes
    try:
        if hasattr(file, "getvalue"):         # UploadedFile
            raw = file.getvalue()
        elif isinstance(file, (str, Path)):   # path string
            with open(file, "rb") as fh:
                raw = fh.read()
        else:
            raw = file.read()
    except Exception as e:
        raise ValueError(f"Could not read uploaded file: {e}")

    if not raw:
        raise ValueError("Uploaded file appears empty. Please check the file and try again.")

    # decode text (try utf-8 then latin1)
    txt = None
    for enc in ("utf-8", "latin1"):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            continue
    if txt is None:
        raise ValueError("Uploaded file encoding not supported (not utf-8/latin1).")

    # attempt parsing with common separators
    df = None
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(txt), sep=sep)
            if df.shape[1] >= 1 and df.shape[0] >= 1:
                break
        except EmptyDataError:
            df = None
            break
        except Exception:
            df = None
            continue

    if df is None or df.shape[1] == 0:
        raise ValueError("No columns found in CSV. Ensure it has headers like frequency_Hz,S11_dB,... and is not empty.")

    # normalize columns
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    # find frequency column
    freq_cols = [c for c in df.columns if "freq" in c.lower() or "frequency" in c.lower()]
    freq_col = freq_cols[0] if freq_cols else df.columns[0]
    df = df.rename(columns={freq_col: "frequency_Hz"})
    df["frequency_Hz"] = pd.to_numeric(df["frequency_Hz"], errors="coerce")

    # map s-params to *_dB columns if present
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

def fig_to_png_bytes(fig):
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", width=1200, height=700, scale=1)
        return png_bytes
    except Exception:
        return None

def export_summary_csv(summary: dict):
    buf = io.StringIO()
    pd.DataFrame([summary]).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ---------- UI & Modes ----------
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
    uploaded_file = st.file_uploader("Upload S-parameter CSV file", type=["csv"], key="viewer_file")
    if uploaded_file is not None:
        try:
            df = read_sparam_csv(uploaded_file)
            df = ensure_db_cols(df)
        except ValueError as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.session_state["last_df"] = df
        st.write("Preview:")
        st.dataframe(df.head())

        st.subheader("Plot settings")
        db_columns = [c for c in df.columns if c.endswith("_dB")]
        show_sparams = st.multiselect("Select parameters to show (dB)", db_columns, default=db_columns[:2])

        # Axis controls
        st.subheader("Axis range controls")
        auto_scale = st.checkbox("Auto-scale axes (Plotly autoscale)", value=True)
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
            layout_kwargs = dict(title="S-Parameters", xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)", template="plotly_white")
            if not auto_scale:
                layout_kwargs["xaxis"] = dict(range=[x_min, x_max])
                layout_kwargs["yaxis"] = dict(range=[y_min, y_max])
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)
            st.session_state["last_figs"]["sparams"] = fig

        # VSWR
        st.subheader("VSWR")
        vswr_fig = None
        if "S11_dB" in df.columns:
            vswr_fig = go.Figure()
            vswr = compute_vswr_from_S11_db(df["S11_dB"].to_numpy())
            vswr_fig.add_trace(go.Scatter(x=df["frequency_Hz"]/1e9, y=vswr, mode="lines", name="VSWR"))
            if not auto_scale:
                vswr_fig.update_layout(xaxis=dict(range=[x_min, x_max]))
            vswr_fig.update_layout(title="VSWR", xaxis_title="Frequency (GHz)", yaxis_title="VSWR", template="plotly_white")
            st.plotly_chart(vswr_fig, use_container_width=True)
            st.session_state["last_figs"]["vswr"] = vswr_fig

        st.subheader("Auto Metrics")
        summary = prepare_summary(df)
        st.session_state["last_summary"] = summary
        st.json(summary)

        st.subheader("Download")
        st.download_button("Download processed CSV", df.to_csv(index=False).encode("utf-8"), "processed_sparams.csv", "text/csv")
        st.download_button("Download summary CSV", export_summary_csv(summary), "summary.csv", "text/csv")

# ---------- Mode 2: Compare Antennas ----------
elif mode == "Compare Antennas":
    st.header("🔁 Antenna Comparison Tool")
    st.write("Upload two S-parameter CSV files to compare S11 and VSWR side-by-side.")
    c1, c2 = st.columns(2)
    file1 = c1.file_uploader("Upload file A", type=["csv"], key="cmp_a")
    file2 = c2.file_uploader("Upload file B", type=["csv"], key="cmp_b")

    # --- inspector (temporary helpful debug) ---
    def inspect_upload(f, label):
        if f is None:
            st.write(f"{label}: not uploaded")
            return
        st.write(f"{label} name:", getattr(f, "name", "n/a"))
        st.write(f"{label} size (bytes):", getattr(f, "size", "n/a"))
        try:
            preview = f.getvalue()[:1000]
            preview_text = preview.decode("utf-8", errors="replace")
            st.text_area(f"{label} preview (first 1000 bytes)", preview_text, height=160)
        except Exception as e:
            st.write(f"{label} preview error:", e)

    inspect_upload(file1, "File A")
    inspect_upload(file2, "File B")
    # --- end inspector ---

    # axis controls for compare
    st.subheader("Comparison Axis Controls")
    auto_scale_cmp = st.checkbox("Auto-scale axes for comparison", value=True, key="auto_cmp")
    x_min_c, x_max_c = 0.0, 3.0
    y_min_c, y_max_c = -60.0, 5.0

    if file1 and file2:
        try:
            df1 = ensure_db_cols(read_sparam_csv(file1))
        except ValueError as e:
            st.error(f"Error reading File A: {e}")
            st.stop()
        try:
            df2 = ensure_db_cols(read_sparam_csv(file2))
        except ValueError as e:
            st.error(f"Error reading File B: {e}")
            st.stop()

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

        # VSWR compare
        vswr1 = compute_vswr_from_S11_db(interp1["S11_dB"]) if "S11_dB" in interp1 else None
        vswr2 = compute_vswr_from_S11_db(interp2["S11_dB"]) if "S11_dB" in interp2 else None
        fig2 = go.Figure()
        if vswr1 is not None:
            fig2.add_trace(go.Scatter(x=common_freq/1e9, y=vswr1, name="A: VSWR"))
        if vswr2 is not None:
            fig2.add_trace(go.Scatter(x=common_freq/1e9, y=vswr2, name="B: VSWR"))
        st.plotly_chart(fig2, use_container_width=True)
        st.session_state["last_figs"]["compare_vswr"] = fig2

        summary_a = prepare_summary(df1)
        summary_b = prepare_summary(df2)
        st.subheader("Summary Comparison")
        comp_df = pd.DataFrame({"Metric": list(summary_a.keys()),
                                "A": list(summary_a.values()),
                                "B": [summary_b.get(k, None) for k in summary_a.keys()]})
        st.dataframe(comp_df)

        # zip results
        with io.BytesIO() as mem_zip:
            with zipfile.ZipFile(mem_zip, mode="w") as zf:
                zf.writestr("summary_A.csv", pd.DataFrame([summary_a]).to_csv(index=False))
                zf.writestr("summary_B.csv", pd.DataFrame([summary_b]).to_csv(index=False))
            mem_zip.seek(0)
            st.download_button("Download comparison ZIP", mem_zip.read(), "comparison_results.zip", "application/zip")

# ---------- Mode 3: 3D Radiation Pattern ----------
elif mode == "3D Radiation Pattern":
    st.header("🌐 3D Radiation Pattern Viewer")
    st.write("Upload a radiation CSV with columns: theta, phi, gain_dBi OR use synthetic sample.")
    rad_file = st.file_uploader("Upload radiation CSV (theta,phi,gain_dBi)", type=["csv"], key="rad_file")
    use_synthetic = st.checkbox("Use synthetic sample pattern (if no file uploaded)", value=True)

    rad_df = None
    if rad_file is not None:
        try:
            txt = rad_file.getvalue().decode("utf-8")
            rad_df = pd.read_csv(io.StringIO(txt))
            # normalize
            if not set(["theta", "phi", "gain_dbi"]).issubset({c.lower() for c in rad_df.columns}):
                st.warning("CSV should contain theta, phi, gain_dBi (case-insensitive). Attempting to map...")
                lower_map = {c.lower(): c for c in rad_df.columns}
                if "theta" in lower_map and "phi" in lower_map and "gain_dbi" in lower_map:
                    rad_df = rad_df.rename(columns={lower_map["theta"]:"theta", lower_map["phi"]:"phi", lower_map["gain_dbi"]:"gain_dBi"})
                else:
                    st.error("Couldn't find required columns.")
                    rad_df = None
        except Exception as e:
            st.error(f"Error reading radiation CSV: {e}")
            rad_df = None

    if rad_df is None and use_synthetic:
        st.info("Using synthetic radiation pattern.")
        thetas = np.linspace(0, np.pi, 61)
        phis = np.linspace(0, 2*np.pi, 73)
        TH, PH = np.meshgrid(thetas, phis)
        G = (np.cos(TH - np.pi/2) ** 2) * (1 / (1 + 0.2*(np.sin(PH*2))))
        G = np.maximum(G, 1e-4)
        Gd = 10 * np.log10(G / G.max()) + 8
        X = (1 + G/np.max(G)) * np.sin(TH) * np.cos(PH)
        Y = (1 + G/np.max(G)) * np.sin(TH) * np.sin(PH)
        Z = (1 + G/np.max(G)) * np.cos(TH)
        surf = go.Surface(x=X, y=Y, z=Z, surfacecolor=Gd, colorscale="Viridis")
        fig3d = go.Figure(data=[surf])
        fig3d.update_layout(title="Synthetic 3D Radiation Pattern", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        st.plotly_chart(fig3d, use_container_width=True, height=700)
        st.session_state["last_figs"]["3d"] = fig3d

# ---------- Mode 4: Summary / Report ----------
elif mode == "Summary / Report":
    st.header("🧾 Summary & Report Generation")
    st.write("Use the last loaded S-parameter dataset or upload a file now.")
    df_file = st.file_uploader("Upload S-parameter CSV for report (optional)", type=["csv"], key="report_file")
    if df_file is None and st.session_state.get("last_df") is not None:
        df = st.session_state["last_df"]
    elif df_file is not None:
        try:
            df = read_sparam_csv(df_file)
            df = ensure_db_cols(df)
        except ValueError as e:
            st.error(f"Error reading file for report: {e}")
            df = None
    else:
        df = None

    if df is not None:
        summary = prepare_summary(df)
        st.json(summary)
        st.download_button("Download summary CSV", export_summary_csv(summary), "demo3_summary.csv", "text/csv")

# ---------- Mode 5: About ----------
elif mode == "About":
    st.header("About Antenna Cloud Demo3")
    st.markdown("""
    **Antenna Cloud Demo3** — Robust version with friendly CSV validation.
    - S-Parameter viewer + VSWR and metrics
    - Comparison with upload inspector and safe error handling
    - 3D radiation pattern viewer (synthetic or from uploaded data)
    - Summary generation and downloads
    Notes:
    - For best results, upload CSVs with columns like: `frequency_Hz,S11_dB,S21_dB,S12_dB,S22_dB`
    - Radiation CSV: `theta,phi,gain_dBi`
    """)

