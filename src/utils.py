"""
src/utils.py
─────────────────────────────────────────────────────────────────────────────
Shared helper functions for the Market Demand Trend Analysis project.
All notebooks import from this module to keep code DRY and consistent.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─── Project Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "Dataset"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_REP    = ROOT / "outputs" / "reports"

OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_REP.mkdir(parents=True, exist_ok=True)

# ─── Plot Theme ───────────────────────────────────────────────────────────────
PALETTE = {
    "primary"   : "#6C63FF",
    "secondary" : "#F5A623",
    "success"   : "#2ECC71",
    "danger"    : "#E74C3C",
    "info"      : "#3498DB",
    "dark"      : "#2D2D2D",
    "light"     : "#F8F9FA",
    "muted"     : "#6C757D",
}

PLOTLY_TEMPLATE = "plotly_dark"

def set_style() -> None:
    """Apply consistent matplotlib style across all notebooks."""
    plt.rcParams.update({
        "figure.facecolor"  : "#1E1E2E",
        "axes.facecolor"    : "#252535",
        "axes.edgecolor"    : "#444466",
        "axes.labelcolor"   : "#DEDEDE",
        "axes.titlesize"    : 14,
        "axes.labelsize"    : 11,
        "axes.grid"         : True,
        "grid.color"        : "#333355",
        "grid.linestyle"    : "--",
        "grid.alpha"        : 0.5,
        "xtick.color"       : "#AAAACC",
        "ytick.color"       : "#AAAACC",
        "text.color"        : "#DEDEDE",
        "font.family"       : "DejaVu Sans",
        "figure.titlesize"  : 16,
        "legend.facecolor"  : "#252535",
        "legend.edgecolor"  : "#444466",
        "legend.fontsize"   : 9,
        "savefig.dpi"       : 150,
        "savefig.bbox"      : "tight",
        "savefig.facecolor" : "#1E1E2E",
    })


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def load_postings() -> pd.DataFrame:
    """Load and lightly clean job_postings.csv."""
    df = pd.read_csv(DATA_DIR / "job_postings.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")
    df["last_processed_time"] = pd.to_datetime(df["last_processed_time"], errors="coerce")
    df = df.drop_duplicates(subset=["job_link"])
    return df


def load_skills() -> pd.DataFrame:
    """Load and clean job_skills.csv."""
    df = pd.read_csv(DATA_DIR / "job_skills.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates(subset=["job_link"])
    return df


def load_summary() -> pd.DataFrame:
    """Load job_summary.csv (large file — may take a moment)."""
    df = pd.read_csv(DATA_DIR / "job_summary.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates(subset=["job_link"])
    return df


def load_merged(include_summary: bool = False) -> pd.DataFrame:
    """
    Merge all three datasets on job_link.

    Parameters
    ----------
    include_summary : bool
        If True, also joins the (large) job_summary table.

    Returns
    -------
    pd.DataFrame
    """
    postings = load_postings()
    skills   = load_skills()
    df = postings.merge(skills, on="job_link", how="left")
    if include_summary:
        summary = load_summary()
        df = df.merge(summary, on="job_link", how="left")
    return df


# ─── Skill Parsing ────────────────────────────────────────────────────────────

def parse_skills(skills_series: pd.Series) -> pd.Series:
    """
    Split comma-separated skill strings into sorted lists.

    Parameters
    ----------
    skills_series : pd.Series
        Column from job_skills containing raw comma-separated strings.

    Returns
    -------
    pd.Series of list[str]
    """
    def _clean(raw: str) -> List[str]:
        if pd.isna(raw):
            return []
        items = [s.strip() for s in raw.split(",")]
        return [s for s in items if s]

    return skills_series.apply(_clean)


def explode_skills(df: pd.DataFrame, skill_col: str = "job_skills") -> pd.DataFrame:
    """
    Explode a DataFrame so every row is (job_link, single_skill).

    Parameters
    ----------
    df         : merged DataFrame containing job_link + skill_col
    skill_col  : name of the column holding comma-separated skill strings

    Returns
    -------
    pd.DataFrame with columns [job_link, skill]
    """
    df = df.copy()
    df["skill_list"] = parse_skills(df[skill_col])
    exploded = df.explode("skill_list").rename(columns={"skill_list": "skill"})
    exploded["skill"] = exploded["skill"].str.strip()
    exploded = exploded[exploded["skill"].notna() & (exploded["skill"] != "")]
    if not exploded.empty:
        casing_map = exploded["skill"].groupby(exploded["skill"].str.lower()).agg(lambda x: x.value_counts().idxmax())
        exploded["skill"] = exploded["skill"].str.lower().map(casing_map)
    return exploded[["job_link", "skill"]]



# ─── Role Categorisation ──────────────────────────────────────────────────────

ROLE_MAP = {
    "Data Scientist"       : ["data scientist", "data science"],
    "Data Engineer"        : ["data engineer", "etl", "data pipeline"],
    "ML Engineer"          : ["machine learning engineer", "ml engineer", "mlops", "ml ops"],
    "Data Analyst"         : ["data analyst", "business analyst", "bi analyst",
                               "business intelligence analyst"],
    "Data Architect"       : ["data architect", "solutions architect"],
    "Database Admin"       : ["database administrator", "dba", "database admin"],
    "AI/ML Researcher"     : ["research scientist", "ai researcher", "applied scientist"],
    "Cloud/Data Ops"       : ["data ops", "dataops", "cloud engineer", "devops"],
    "Healthcare/Bio"       : ["medical", "clinical", "laboratory", "biologist"],
    "Security/Compliance"  : ["dlp", "cybersecurity", "data loss prevention",
                               "data security", "compliance"],
    "Manager/Director"     : ["manager", "director", "head of", "vp", "vice president",
                               "lead data"],
}


def categorise_role(title: str) -> str:
    """Map a job title string to a canonical role category."""
    t = str(title).lower()
    for category, keywords in ROLE_MAP.items():
        if any(kw in t for kw in keywords):
            return category
    return "Other"


def add_role_category(df: pd.DataFrame, title_col: str = "job_title") -> pd.DataFrame:
    """Add a 'role_category' column to the DataFrame."""
    df = df.copy()
    df["role_category"] = df[title_col].apply(categorise_role)
    return df


# ─── Time Series Helpers ──────────────────────────────────────────────────────

def build_daily_series(df: pd.DataFrame,
                        date_col: str = "first_seen",
                        value_col: Optional[str] = None) -> pd.Series:
    """
    Aggregate postings by day.

    Parameters
    ----------
    df        : DataFrame with date column
    date_col  : name of datetime column
    value_col : if None, count rows; otherwise sum this column

    Returns
    -------
    pd.Series indexed by date, filled with 0 for missing days
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    if value_col is None:
        series = df.resample("D").size()
    else:
        series = df[value_col].resample("D").sum()

    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_idx, fill_value=0)
    series.name = value_col or "posting_count"
    return series


def train_test_split_ts(series: pd.Series,
                         test_frac: float = 0.20) -> Tuple[pd.Series, pd.Series]:
    """Split a time series keeping chronological order."""
    split = int(len(series) * (1 - test_frac))
    return series.iloc[:split], series.iloc[split:]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray,
         epsilon: float = 1e-8) -> float:
    return float(np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100)


def metrics_table(actual: np.ndarray,
                  predicted: np.ndarray,
                  model_name: str = "Model") -> pd.DataFrame:
    """Return a one-row DataFrame with MAE, RMSE, MAPE for a model."""
    return pd.DataFrame([{
        "Model" : model_name,
        "MAE"   : round(mae(actual, predicted), 3),
        "RMSE"  : round(rmse(actual, predicted), 3),
        "MAPE%" : round(mape(actual, predicted), 2),
    }])


# ─── Save Helpers ─────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure to outputs/figures/."""
    path = OUT_FIG / f"{name}.png"
    fig.savefig(path)
    print(f"✅ Saved → {path}")


def save_plotly(fig: go.Figure, name: str) -> None:
    """Save a Plotly figure as interactive HTML to outputs/figures/."""
    path = OUT_FIG / f"{name}.html"
    fig.write_html(str(path))
    print(f"✅ Saved → {path}")


# ─── Plotly Convenience ───────────────────────────────────────────────────────

def plotly_bar(df: pd.DataFrame,
               x: str,
               y: str,
               title: str,
               color: str = PALETTE["primary"],
               top_n: Optional[int] = None) -> go.Figure:
    """Quick horizontal bar chart."""
    if top_n:
        df = df.nlargest(top_n, y)
    fig = px.bar(df, x=y, y=x, orientation="h",
                 title=title, template=PLOTLY_TEMPLATE,
                 color_discrete_sequence=[color])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def plotly_line(series_dict: dict,
                title: str,
                y_label: str = "Count") -> go.Figure:
    """
    Multi-line Plotly chart.

    Parameters
    ----------
    series_dict : {label: pd.Series} mapping
    """
    fig = go.Figure()
    colors = list(PALETTE.values())
    for i, (label, s) in enumerate(series_dict.items()):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=label,
            mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
