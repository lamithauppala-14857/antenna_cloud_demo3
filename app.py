# ========================= app.py - Antenna Cloud Demo3 ============================
# FULL FILE (ALL 5 MODULES) WITH UPDATED VIEWER MODE (Option B-3 polynomial extrapolation)
# ===================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
from pathlib import Path
from pandas.errors import EmptyDataError
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.set_page_config("Antenna Cloud Demo3", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# ROBUST CSV READER
# --------------------------------------------------------------------------
def read_csv_bytes(file) -> str:
    try:
        size = getattr(file, "size", None)
    except Exception:
        size = None
    if size == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    try:
        if hasattr(file, "getvalue"):
            raw = file.getvalue()
        elif isinstance(file, (str, Path)):
            with open(file, "rb") as fh:
                raw = fh.read()
        else:
            raw = file.read()
    except Exception as e:
        raise ValueError(f"Could not read uploaded file: {e}")

    if not raw:
        raise ValueError("Uploaded file appears empty.")

    for enc in ("utf-8", "latin1"):
        try:
            txt = raw.decode(enc)
            return txt
        except Exception:
            continue
    raise ValueError("Encoding not supported.")

def smart_read_dataframe(file) -> pd.DataFrame:
    txt = read_csv_bytes(file)
    df = None
    for sep in [",", ";", "\t", "|"]:
        try:
            df_try = pd.read_csv(io.StringIO(txt), sep=sep)
            if df_try.shape[1] >= 1 and df_try.shape[0] >= 1:
                df = df_try
                break
        except Exception:
            df = None
            continue

    if df is None or df.shape[1] == 0:
        raise ValueError("CSV has no usable columns.")

    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    return df

# --------------------------------------------------------------------------
# S-PARAM UTILITIES
# --------------------------------------------------------------------------
def read_sparam_csv(file) -> pd.DataFrame:
    df = smart_read_dataframe(file)
    freq_cols = [c for c in df.columns if "freq" in c.lower()]
    freq_col = freq_cols[0] if freq_cols else df.columns[0]
    df = df.rename(columns={freq_col: "frequency_Hz"})

    df["frequency_Hz"] = pd.to_numeric(df["frequency_Hz"], errors="coerce")

    for expected in ["s11", "s21", "s12", "s22"]:
        cand = [c for c in df.columns if expected in c.lower().replace(" ", "")]
        if cand:
            df[expected.upper()+"_dB"] = pd.to_numeric(df[cand[0]], errors="coerce")
    return df

def ensure_db_cols(df):
    for p in ["S11","S21","S12","S22"]:
        if p+"_dB" not in df.columns and p in df.columns:
            df[p+"_dB"] = 20*np.log10(np.abs(df[p].astype(float)))
    return df

def magdb_to_linear(db):
    return 10**(db/20)

def compute_vswr_from_S11_db(s11_db):
    gamma = magdb_to_linear(s11_db)
    gamma = np.clip(gamma,0,0.999999)
    return (1+gamma)/(1-gamma)

def find_resonant_and_bw(df):
    out = {}
    if "S11_dB" not in df.columns:
        return out
    s11 = df["S11_dB"].to_numpy()
    freq = df["frequency_Hz"].to_numpy()
    idx = np.nanargmin(s11)
    out["resonant_freq_Hz"] = float(freq[idx])
    out["resonant_S11_dB"] = float(s11[idx])

    mask = s11 < -10
    ranges=[]
    i=0
    while i < len(mask):
        if mask[i]:
            j=i
            while j+1 < len(mask) and mask[j+1]:
                j+=1
            ranges.append((freq[i],freq[j]))
            i=j+1
        else:
            i+=1
    out["bandwidth_ranges_Hz"] = ranges
    return out

def prepare_summary(df):
    df = ensure_db_cols(df)
    summary={}
    if "S11_dB" in df.columns:
        m=find_resonant_and_bw(df)
        summary.update(m)
        summary["min_S11_dB"] = float(df["S11_dB"].min())
        summary["max_S21_dB"] = float(df["S21_dB"].max()) if "S21_dB" in df.columns else None
        summary["vswr_mean"] = float(compute_vswr_from_S11_db(df["S11_dB"]).mean())
    summary["n_points"]=len(df)
    return summary

# --------------------------------------------------------------------------
# NEW FUNCTION: Polynomial extrapolation (B-3)
# --------------------------------------------------------------------------
def poly_extrapolate(x, y, degree, x_new_min, x_new_max):
    """Fit polynomial & generate extended Y for extended X range."""
    try:
        coeff = np.polyfit(x, y, degree)
        poly = np.poly1d(coeff)
        x_ext = np.linspace(x_new_min, x_new_max, 2000)
        y_ext = poly(x_ext)
        return x_ext, y_ext
    except:
        return x, y

# --------------------------------------------------------------------------
# 3D RADIATION BUILDER
# (unchanged)
# --------------------------------------------------------------------------
def build_3d_pattern_from_rad_df(rad_df, grid_n_theta=61, grid_n_phi=73):
    col = {c.lower().strip():c for c in rad_df.columns}
    theta_col = col.get("theta")
    phi_col = col.get("phi")
    gain_col = col.get("gain_dbi") or col.get("gain")

    if theta_col is None or phi_col is None or gain_col is None:
        raise ValueError("Radiation CSV must contain theta, phi, gain_dBi.")

    thetas = pd.to_numeric(rad_df[theta_col], errors="coerce").to_numpy()
    phis = pd.to_numeric(rad_df[phi_col], errors="coerce").to_numpy()
    gains = pd.to_numeric(rad_df[gain_col], errors="coerce").to_numpy()

    mask = ~(np.isnan(thetas)|np.isnan(phis)|np.isnan(gains))
    thetas,phis,gains = thetas[mask], phis[mask], gains[mask]

    unique_t = np.unique(thetas)
    unique_p = np.unique(phis)
    regular = (len(unique_t)*len(unique_p)==len(gains))

    if regular:
        try:
            df2=rad_df.copy()
            df2[theta_col]=pd.to_numeric(df2[theta_col])
            df2[phi_col]=pd.to_numeric(df2[phi_col])
            df2[gain_col]=pd.to_numeric(df2[gain_col])
            grid = df2.pivot(index=phi_col, columns=theta_col, values=gain_col)
            TH_deg = grid.columns.values.astype(float)
            PH_deg = grid.index.values.astype(float)
            TH,PH = np.meshgrid(TH_deg*np.pi/180, PH_deg*np.pi/180)
            G = grid.values
        except:
            regular=False

    if not regular:
        θ = np.linspace(0,180,grid_n_theta)
        φ = np.linspace(0,360,grid_n_phi)
        TH_deg,PH_deg=np.meshgrid(θ,φ)
        pts=np.vstack([thetas,phis]).T
        G = griddata(pts,gains,(TH_deg,PH_deg),'linear')
        nanmask=np.isnan(G)
        if nanmask.any():
            G2 = griddata(pts,gains,(TH_deg,PH_deg),'nearest')
            G[nanmask]=G2[nanmask]
        TH = TH_deg*np.pi/180
        PH = PH_deg*np.pi/180

    Gv=np.nan_to_num(G,nan=np.nanmin(G))
    R = 1 + (Gv-Gv.min())/(Gv.max()-Gv.min()+1e-9)
    X = R*np.sin(TH)*np.cos(PH)
    Y = R*np.sin(TH)*np.sin(PH)
    Z = R*np.cos(TH)
    surf = go.Surface(x=X,y=Y,z=Z,surfacecolor=Gv,colorscale="Viridis")
    fig = go.Figure(data=[surf])
    fig.update_layout(title="3D Radiation Pattern",scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z'))
    return fig

# --------------------------------------------------------------------------
# UI START
# --------------------------------------------------------------------------
st.sidebar.title("Antenna Cloud Demo3")
mode = st.sidebar.selectbox("Choose mode", ["S-Parameter Viewer","Compare Antennas","3D Radiation Pattern","Summary / Report","About"])

if "last_df" not in st.session_state:
    st.session_state["last_df"]=None

# --------------------------------------------------------------------------
# MODE 1: UPDATED S-PARAMETER VIEWER (Option B-3)
# --------------------------------------------------------------------------
if mode=="S-Parameter Viewer":
    st.header("📈 S-Parameter Viewer")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        try:
            df = read_sparam_csv(file)
            df = ensure_db_cols(df)
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.session_state["last_df"]=df
        st.dataframe(df.head())

        cols = [c for c in df.columns if c.endswith("_dB")]
        to_show = st.multiselect("Select parameters", cols, default=cols[:2])

        auto = st.checkbox("Auto-scale", True)

        freq = df["frequency_Hz"]/1e9
        xmin=float(freq.min())
        xmax=float(freq.max())

        x_user_min, x_user_max = st.slider(
            "X Range (GHz)", min_value=0.0, max_value=max(10.0,xmax),
            value=(xmin,xmax), step=0.1
        )

        y_min, y_max = st.slider(
            "Y Range (dB)", -100.0, 20.0,
            value=(-60.0,5.0), step=0.5
        )

        degree = 3   # as you selected B-3
        fig = go.Figure()

        for p in to_show:
            y = df[p].to_numpy()
            x = freq.to_numpy()

            # Extrapolate beyond original data
            x_ext, y_ext = poly_extrapolate(x, y, degree, x_user_min, x_user_max)

            fig.add_trace(go.Scatter(x=x_ext, y=y_ext, mode="lines", name=f"{p} (extended)"))

        layout = dict(
            title="S-Parameters (Extended Lines)",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Magnitude (dB)",
            template="plotly_white",
        )

        if not auto:
            layout["xaxis"]={"range":[x_user_min,x_user_max]}
            layout["yaxis"]={"range":[y_min,y_max]}

        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        # VSWR (unchanged, but also extrapolated)
        if "S11_dB" in df.columns:
            y = compute_vswr_from_S11_db(df["S11_dB"])
            x = freq.to_numpy()
            x_ext, y_ext = poly_extrapolate(x, y, 3, x_user_min, x_user_max)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x_ext, y=y_ext, mode="lines", name="VSWR"))
            fig2.update_layout(
                title="VSWR (Extended)",
                xaxis_title="GHz",
                yaxis_title="VSWR",
                template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Auto Metrics")
        st.json(prepare_summary(df))

# --------------------------------------------------------------------------
# MODE 2: COMPARE (UNCHANGED)
# --------------------------------------------------------------------------
elif mode=="Compare Antennas":
    st.header("🔁 Compare Antennas")
    st.info("This module is unchanged from your original code.")
    st.write("Upload two files...")
    # -------------- unchanged code --------------
    # Keeping content identical so your app behavior stays same
    c1,c2=st.columns(2)
    f1=c1.file_uploader("File A",type=["csv"])
    f2=c2.file_uploader("File B",type=["csv"])
    if f1 and f2:
        try:
            df1=ensure_db_cols(read_sparam_csv(f1))
            df2=ensure_db_cols(read_sparam_csv(f2))
        except Exception as e:
            st.error(str(e)); st.stop()

        st.write("Preview A:"); st.dataframe(df1.head())
        st.write("Preview B:"); st.dataframe(df2.head())

        common = np.linspace(max(df1.frequency_Hz.min(), df2.frequency_Hz.min()),
                             min(df1.frequency_Hz.max(), df2.frequency_Hz.max()),500)
        fig=go.Figure()
        if "S11_dB" in df1: fig.add_trace(go.Scatter(x=common/1e9,y=np.interp(common,df1.frequency_Hz,df1["S11_dB"]),name="A S11"))
        if "S11_dB" in df2: fig.add_trace(go.Scatter(x=common/1e9,y=np.interp(common,df2.frequency_Hz,df2["S11_dB"]),name="B S11"))
        fig.update_layout(title="Compare S11",template="plotly_white")
        st.plotly_chart(fig,use_container_width=True)

# --------------------------------------------------------------------------
# MODE 3: RADIATION (UNCHANGED)
# --------------------------------------------------------------------------
elif mode=="3D Radiation Pattern":
    st.header("🌐 3D Radiation Pattern Viewer")
    rad = st.file_uploader("Upload radiation CSV", type=["csv"])
    use_syn = st.checkbox("Use synthetic example", True)

    if rad:
        try:
            d=smart_read_dataframe(rad)
            fig=build_3d_pattern_from_rad_df(d)
            st.plotly_chart(fig,use_container_width=True)
        except Exception as e:
            st.error(str(e))
    else:
        if use_syn:
            st.info("Synthetic pattern shown.")
            θ=np.linspace(0,np.pi,61)
            φ=np.linspace(0,2*np.pi,73)
            TH,PH=np.meshgrid(θ,φ)
            G = (np.cos(TH-np.pi/2)**2)
            Gd = 10*np.log10(G/G.max()+1e-9)
            X=np.sin(TH)*np.cos(PH)
            Y=np.sin(TH)*np.sin(PH)
            Z=np.cos(TH)
            surf=go.Surface(x=X,y=Y,z=Z,surfacecolor=Gd,colorscale="Viridis")
            fig=go.Figure(data=[surf])
            st.plotly_chart(fig,use_container_width=True)

# --------------------------------------------------------------------------
# MODE 4: SUMMARY (UNCHANGED)
# --------------------------------------------------------------------------
elif mode=="Summary / Report":
    st.header("🧾 Summary")
    f=st.file_uploader("Upload CSV",type=["csv"])
    if f:
        try:
            df=ensure_db_cols(read_sparam_csv(f))
            st.json(prepare_summary(df))
        except Exception as e:
            st.error(str(e))

# --------------------------------------------------------------------------
# MODE 5: ABOUT
# --------------------------------------------------------------------------
else:
    st.header("About Demo3")
    st.markdown("""
    ✔ S-Parameter Viewer  
    ✔ Comparison  
    ✔ 3D Radiation  
    ✔ Summary & Report  
    **Antenna Cloud Demo3** — Robust version with improved 3D radiation handling. 
    - S-Parameter viewer + VSWR and metrics 
    - Comparison tool with upload inspector and safe handling 
    - 3D radiation pattern viewer (accepts grid or scattered points; will interpolate) 
    - Summary generation and downloads Notes: 
    - For best results, upload radiation CSVs with columns: theta (deg), phi (deg), gain_dBi 
    - If the pattern doesn't show, try exporting HFSS radiation on a regular theta x phi grid.
    """)

# --------------------------------------------------------------------------
# END OF FILE
# --------------------------------------------------------------------------




