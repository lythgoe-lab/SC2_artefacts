import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Rectangle, Ellipse
from dataclasses import dataclass
from typing import Sequence


base_dir = "../../"
high_conf_pairs_path = os.path.join(
    base_dir, "processed_data", "transmission_pair_examples",
    "high_confidence_pairs_all_variant_types_and_freqs.csv"
)
random_pairs_path = os.path.join(
    base_dir, "processed_data", "transmission_pair_examples",
    "random_pairs_all_variant_types_and_freqs.csv"
)
maf_table_path = os.path.join(
    base_dir, "processed_data", "10x", "final_maf_thresholds.csv"
)
training_set_path = os.path.join(
    base_dir, "processed_data", "transmission_pair_examples", "training_set.csv"
)
bottleneck_examples_path = os.path.join(
    base_dir, "processed_data", "transmission_pair_examples", "possible_examples_alt.csv"
)

figures_dir = os.path.join(
    base_dir, "figures"
)

depth_thresh = "1000x"
selected_random_pairs = [10, 78]
selected_high_conf_pairs = [109, 145, 30]

RANDOM_COL_TITLES = [
    "Depth ≥ 100x, MAF≥3%\nUnmasked",
    "Depth ≥ 1000x, MAF≥3%\nUnmasked",
    "Depth ≥ 1000x, adaptive MAF\nUnmasked",
    "Depth ≥ 1000x, adaptive MAF\nMasked",
]

COL_SHARED = "#489B42"
COL_UNSHARED = "#7D7C7C"
COL_CONSCHANGE = "#ca3e3e"
DASH_COLOR = "#000000"
X2, Y2 = 0.50, 0.50

BN_COL_UNMASKED = "#B8860B"
BN_COL_MASKED = "#264653"

plt.style.use("plot_style_settings.mplstyle")

@dataclass
class ROI:
    x0: float
    x1: float
    y0: float
    y1: float
    which: str = "all"
    shape: str = "box"
    label_loc: str = "tr"
    pad_pts: tuple[int, int] = (3, 3)
    lw: float = 0.3
    alpha: float = 0.8
    panels: str = "both"
    pair_ns: Sequence[int] | None = None
    rows: Sequence[int] | None = None


ROIS = [
    ROI(
        x0=-0.01, x1=0.03,
        y0=0.97, y1=1.01,
        which="all",
        shape="box",
        panels="both",
        rows=[0],
        pad_pts=(0.01, 0.01),
    ),
]


def _norm_site(protocol_str: str) -> str:
    s = str(protocol_str)
    if s.endswith("_ILLUMINA"):
        s = s[:-len("_ILLUMINA")]
    for prefix in ("QEUH", "LSPA", "BRBR", "MILK"):
        if s.startswith(prefix):
            parts = s.split("_", 1)
            if len(parts) > 1:
                return "SANG_" + parts[1]
            return "SANG"
    return s


def pct_no_symbol(v, _pos):
    return f"{int(round(v * 100))}"


def resolve_thresholds_for_panel(sub_df: pd.DataFrame, maf_map: dict, default=0.03):
    if sub_df.empty:
        return default, default
    src_site = _norm_site(sub_df["site_src"].iloc[0])
    rec_site = _norm_site(sub_df["site_rec"].iloc[0])
    x_thr = maf_map.get(src_site, default)
    y_thr = maf_map.get(rec_site, default)
    if src_site not in maf_map:
        print(f"[WARNING] No MAF threshold found for source site {src_site!r}. Using default={default}.")
    if rec_site not in maf_map:
        print(f"[WARNING] No MAF threshold found for recipient site {rec_site!r}. Using default={default}.")
    return x_thr, y_thr


def axis_cosmetics_tv(ax, show_ylabel=False, show_yticklabels=True):
    ax.set_xlim(-0.01, 0.51)
    ax.set_ylim(-0.01, 1.01)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_box_aspect((y1 - y0) / (x1 - x0))

    ax.xaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(pct_no_symbol))
    ax.yaxis.set_major_formatter(FuncFormatter(pct_no_symbol))

    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.15)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2)

    ax.tick_params(which="both", labelbottom=True, labelleft=show_yticklabels)
    ax.minorticks_on()

    if show_ylabel:
        ax.set_ylabel("Allele frequency sample 2 (%)")
    else:
        ax.set_ylabel("")


def draw_background_gradient(ax, x_thr, y_thr, res=220, w=0.04):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    def band(v, lo, hi):
        left = 1.0 / (1.0 + np.exp(-(v - lo) / max(w, 1e-6)))
        right = 1.0 / (1.0 + np.exp(-(hi - v) / max(w, 1e-6)))
        return np.clip(left * right, -0.01, 1.01)

    xx = np.linspace(x0, x1, res)
    yy = np.linspace(y0, y1, res)
    X, Y = np.meshgrid(xx, yy)

    below_x = band(X, x0, x_thr)
    above_x = band(X, x_thr, x1)

    below_y_low = band(Y, y0, y_thr)
    below_y_high = band(Y, 1.0 - y_thr, y1)
    below_y = np.maximum(below_y_low, below_y_high)
    above_y = band(Y, y_thr, 1.0 - y_thr)

    wG = (above_x * above_y) * 0.6
    wY = (below_x * above_y) + (above_x * below_y)
    wR = below_x * below_y

    S = wG + wY + wR
    S[S == 0] = 1.0
    G = (wG / S)[..., None]
    Yw = (wY / S)[..., None]
    R = (wR / S)[..., None]

    colG = np.array([0xBF, 0xE7, 0xBD]) / 255.0
    colY = np.array([0xF4, 0xE7, 0xAA]) / 255.0
    colR = np.array([0xF5, 0xB5, 0xB5]) / 255.0
    rgb = G * colG + Yw * colY + R * colR

    ax.imshow(
        rgb,
        origin="lower",
        extent=(x0, x1, y0, y1),
        interpolation="bilinear",
        zorder=-5,
    )


def dashed_guides(ax, x_thr, y_thr):
    ax.axvline(x_thr, color=DASH_COLOR, linestyle="--", alpha=0.50, zorder=2)
    ax.axhline(y_thr, color=DASH_COLOR, linestyle="--", alpha=0.50, zorder=2)
    ax.axhline(1.0 - y_thr, color=DASH_COLOR, linestyle=":", alpha=0.60, zorder=2)
    ax.axvline(X2, color=DASH_COLOR, linestyle="--", alpha=0.25, zorder=2)
    ax.axhline(Y2, color=DASH_COLOR, linestyle="--", alpha=0.25, zorder=2)


def assign_isnv_allele_freqs(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in ["minor_freq_src", "minor_freq_rec", "maj_freq_src", "maj_freq_rec"]:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0.0)

    def _row_rule(row):
        fs_minor = float(row.get("minor_freq_src", 0.0))
        fr_minor = float(row.get("minor_freq_rec", 0.0))
        fr_major = float(row.get("maj_freq_rec", 0.0))
        maj_r = str(row.get("maj_rec", ""))
        min_s = str(row.get("min_src", ""))
        min_r = str(row.get("min_rec", ""))
        if fs_minor > 0 and min_s not in ("", "nan", "None"):
            allele = min_s
            freq_src = fs_minor
            if allele == maj_r:
                freq_rec = fr_major
            elif allele == min_r:
                freq_rec = fr_minor
            else:
                freq_rec = 0.0
        else:
            freq_src = 0.0
            freq_rec = fr_minor
        return pd.Series({"freq_src": freq_src, "freq_rec": freq_rec})

    rule_freqs = df2.apply(_row_rule, axis=1)
    df2["freq_src"] = rule_freqs["freq_src"]
    df2["freq_rec"] = rule_freqs["freq_rec"]
    return df2


def _label_anchor(roi: ROI):
    if roi.label_loc == "tr":
        return (roi.x1, roi.y1), ("left", "bottom")
    if roi.label_loc == "tl":
        return (roi.x0, roi.y1), ("right", "bottom")
    if roi.label_loc == "br":
        return (roi.x1, roi.y0), ("left", "top")
    if roi.label_loc == "bl":
        return (roi.x0, roi.y0), ("right", "top")
    return (roi.x1, roi.y1), ("left", "bottom")


def draw_rois_with_counts(ax, fx, fy, masks_by_name: dict[str, np.ndarray], rois: list[ROI],
                         panel_name: str, pair_n: int, row_idx: int):
    for roi in rois:
        if roi.panels not in ("both", panel_name):
            continue
        if roi.pair_ns is not None and pair_n not in set(roi.pair_ns):
            continue
        if roi.rows is not None and row_idx not in set(roi.rows):
            continue

        base_mask = masks_by_name.get(roi.which, masks_by_name["all"])
        in_roi = (fx >= roi.x0) & (fx <= roi.x1) & (fy >= roi.y0) & (fy <= roi.y1) & base_mask
        c = int(np.count_nonzero(in_roi))
        if c <= 1:
            continue

        if roi.shape == "box":
            ax.add_patch(
                Rectangle(
                    (roi.x0, roi.y0),
                    roi.x1 - roi.x0,
                    roi.y1 - roi.y0,
                    fill=False,
                    linewidth=roi.lw,
                    alpha=roi.alpha,
                    edgecolor="black",
                    zorder=20,
                )
            )
        else:
            ax.add_patch(
                Ellipse(
                    ((roi.x0 + roi.x1) / 2.0, (roi.y0 + roi.y1) / 2.0),
                    width=(roi.x1 - roi.x0),
                    height=(roi.y1 - roi.y0),
                    fill=False,
                    linewidth=roi.lw,
                    alpha=roi.alpha,
                    edgecolor="black",
                    zorder=20,
                )
            )

        (xa, ya), (ha, va) = _label_anchor(roi)
        ax.annotate(
            str(c),
            xy=(xa, ya),
            xytext=roi.pad_pts,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=4.5,
            color="black",
            zorder=21,
        )


def scatter_points(ax, sub, x_thr, y_thr, panel_name: str, pair_n: int, row_idx: int):
    keep = sub["freq_src"].to_numpy() < 0.5
    sub_f = sub.loc[keep].copy()

    fx = sub_f["freq_src"].to_numpy()
    fy = sub_f["freq_rec"].to_numpy()

    maj_src = sub_f["maj_src"].astype(str).to_numpy()
    min_src = sub_f["min_src"].astype(str).to_numpy()
    maj_rec = sub_f["maj_rec"].astype(str).to_numpy()
    min_rec = sub_f["min_rec"].astype(str).to_numpy()

    minor_x = (fx >= x_thr) & (fx < 0.5)
    minor_y = (fy >= y_thr) & (fy < 0.5)
    major_x = fx >= 0.5
    major_y = fy >= 0.5
    below_x = fx < x_thr
    below_y = fy < y_thr

    green = minor_x & minor_y & (min_src == min_rec)
    red = (
        ((major_x & (fy < 0.5)) & (maj_src == min_rec))
        | (((fx < 0.5) & major_y) & (min_src == maj_rec))
    )

    grey1 = below_x & (((fy < 0.5) & (min_src == min_rec)) | (major_y & (min_src == maj_rec)))
    grey2 = below_y & (((fx < 0.5) & (min_src == min_rec)) | (major_x & (min_src == maj_rec)))
    grey = (grey1 | grey2) & ~red & ~green

    ax.scatter(fx[grey], fy[grey], s=3.5, alpha=0.8, c=COL_UNSHARED, edgecolor="none", zorder=6)
    ax.scatter(fx[green], fy[green], s=3.5, alpha=0.9, c=COL_SHARED, edgecolor="none", zorder=8)
    ax.scatter(fx[red], fy[red], s=3.5, alpha=0.9, c=COL_CONSCHANGE, edgecolor="none", zorder=7)

    masks_by_name = {"all": np.ones_like(fx, dtype=bool), "green": green, "red": red, "grey": grey}
    draw_rois_with_counts(ax, fx, fy, masks_by_name, ROIS, panel_name=panel_name, pair_n=pair_n, row_idx=row_idx)
    return sub_f


def draw_tv(ax, df_sub: pd.DataFrame, maf_map: dict, panel_name: str, pair_n: int, row_idx: int,
            show_ylabel=False, show_yticklabels=True):
    axis_cosmetics_tv(ax, show_ylabel=show_ylabel, show_yticklabels=show_yticklabels)
    x_thr, y_thr = resolve_thresholds_for_panel(df_sub, maf_map)
    draw_background_gradient(ax, x_thr, y_thr)
    dashed_guides(ax, x_thr, y_thr)
    scatter_points(ax, df_sub, x_thr=x_thr, y_thr=y_thr, panel_name=panel_name, pair_n=pair_n, row_idx=row_idx)
    return x_thr, y_thr


def build_training_pair_map(train_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    required = ["sample.source", "sample.recipient"]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training set: {missing}")
    train_df = train_df.copy()
    train_df["training_pair_n"] = train_df.index + 1
    m = {}
    for _, row in train_df.iterrows():
        key = (str(row["sample.source"]).strip(), str(row["sample.recipient"]).strip())
        m[key] = int(row["training_pair_n"])
    return m


def map_pairs_to_training_set(sub_all: pd.DataFrame, training_map: dict[tuple[str, str], int]) -> int | None:
    required = ["ID_src", "ID_rec"]
    missing = [c for c in required if c not in sub_all.columns]
    if missing:
        raise ValueError(f"Missing required columns in pair data: {missing}")
    src = str(sub_all["ID_src"].iloc[0]).strip()
    rec = str(sub_all["ID_rec"].iloc[0]).strip()
    return training_map.get((src, rec))


def collect_pair_rows(df, bn_df, pair_ids, depth_thresh, training_map) -> tuple[list[dict], float]:
    pair_rows = []
    bn_vals = []

    for pn in pair_ids:
        sub_all = df[(df["pair_n"].astype(str) == str(pn)) & (df["depth_thresh"] == depth_thresh)].copy()
        if sub_all.empty:
            print(f"[WARNING] No rows found for pair_n={pn} with depth_thresh={depth_thresh}. Skipping.")
            continue

        tv_unmasked = assign_isnv_allele_freqs(sub_all)
        masked_any = sub_all["masked_src"] | sub_all["masked_rec"]
        sub_masked = sub_all[~masked_any].copy()
        tv_masked = assign_isnv_allele_freqs(sub_masked) if not sub_masked.empty else sub_masked

        training_pair_id = map_pairs_to_training_set(sub_all, training_map)
        bn_info = None

        if training_pair_id is None:
            print(f"[WARNING] Could not map TV pair_n={pn} (ID_src/ID_rec) to training_set row. Bottleneck will be NA.")
        else:
            bn_row = bn_df[bn_df["pair"].astype(int) == int(training_pair_id)]
            if bn_row.empty:
                print(f"[WARNING] No bottleneck row found for training pair={training_pair_id} (from TV pair_n={pn}).")
            else:
                r = bn_row.iloc[0]
                m0 = float(r.get("mean_0", np.nan))
                m1 = float(r.get("mean_1", np.nan))
                lo0 = float(r.get("hpd.lower_0", np.nan))
                hi0 = float(r.get("hpd.upper_0", np.nan))
                lo1 = float(r.get("hpd.lower_1", np.nan))
                hi1 = float(r.get("hpd.upper_1", np.nan))

                yerr0 = None
                yerr1 = None

                if np.isfinite(m0) and np.isfinite(lo0) and np.isfinite(hi0):
                    if lo0 > hi0:
                        lo0, hi0 = hi0, lo0
                    yerr0 = [[abs(m0 - lo0)], [abs(hi0 - m0)]]
                    bn_vals.append(hi0)
                elif np.isfinite(m0):
                    bn_vals.append(m0)

                if np.isfinite(m1) and np.isfinite(lo1) and np.isfinite(hi1):
                    if lo1 > hi1:
                        lo1, hi1 = hi1, lo1
                    yerr1 = [[abs(m1 - lo1)], [abs(hi1 - m1)]]
                    bn_vals.append(hi1)
                elif np.isfinite(m1):
                    bn_vals.append(m1)

                bn_info = dict(mean_0=m0, mean_1=m1, yerr0=yerr0, yerr1=yerr1)

        pair_rows.append(
            dict(
                pair_n=int(pn),
                training_pair_id=training_pair_id,
                tv_unmasked=tv_unmasked,
                tv_masked=tv_masked,
                bn=bn_info,
            )
        )

    bn_ymax = 10
    if bn_vals:
        bn_ymax = max(10, max(bn_vals) * 1.10)

    return pair_rows, bn_ymax


def subset_random_pair(df_random: pd.DataFrame, pair_n: int, depth_thresh: str) -> pd.DataFrame:
    return df_random[
        (df_random["pair_n"].astype(str) == str(pair_n))
        & (df_random["depth_thresh"].astype(str) == str(depth_thresh))
    ].copy()


def resolve_thresholds_random(mode: str, sub_df: pd.DataFrame, maf_map: dict, default=0.03) -> tuple[float, float]:
    if mode == "adaptive":
        return resolve_thresholds_for_panel(sub_df, maf_map, default=default)
    return default, default


def draw_random_tv(ax, sub_all: pd.DataFrame, maf_map: dict, mode: str, masked: bool,
                   title: str, show_ylabel=False, show_yticklabels=True):
    if sub_all.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=6)
        ax.set_axis_off()
        return

    sub_tv = assign_isnv_allele_freqs(sub_all)
    if masked:
        masked_any = sub_tv["masked_src"] | sub_tv["masked_rec"]
        sub_tv = sub_tv[~masked_any].copy()

    if sub_tv.empty:
        ax.text(0.5, 0.5, "No data\nafter masking", ha="center", va="center", fontsize=6)
        ax.set_axis_off()
        return

    axis_cosmetics_tv(ax, show_ylabel=show_ylabel, show_yticklabels=show_yticklabels)
    x_thr, y_thr = resolve_thresholds_random(mode=mode, sub_df=sub_tv, maf_map=maf_map, default=0.03)
    draw_background_gradient(ax, x_thr, y_thr)
    dashed_guides(ax, x_thr, y_thr)
    scatter_points(
        ax,
        sub_tv,
        x_thr=x_thr,
        y_thr=y_thr,
        panel_name="masked" if masked else "unmasked",
        pair_n=int(sub_tv["pair_n"].iloc[0]) if "pair_n" in sub_tv.columns else -1,
        row_idx=0,
    )
    ax.set_title(title, pad=6)


def draw_bottleneck_panel(ax, pair_rows, bn_ymax):
    n_pairs = len(pair_rows)
    x_centers = np.arange(n_pairs)
    dx = 0.18
    x_unmasked = x_centers - dx
    x_masked = x_centers + dx

    for i, row in enumerate(pair_rows):
        bn = row["bn"]
        if bn is None or not np.isfinite(bn["mean_0"]) or not np.isfinite(bn["mean_1"]):
            continue

        ax.errorbar(
            [x_unmasked[i]], [float(bn["mean_0"])], yerr=bn["yerr0"],
            fmt="o", markersize=3, capsize=1, capthick=0.4,
            color=BN_COL_UNMASKED, ecolor="black", linewidth=0.4, zorder=3
        )
        ax.errorbar(
            [x_masked[i]], [float(bn["mean_1"])], yerr=bn["yerr1"],
            fmt="o", markersize=3, capsize=1, capthick=0.4,
            color=BN_COL_MASKED, ecolor="black", linewidth=0.4, zorder=3
        )

    ax.set_xlim(-0.6, n_pairs - 0.4)
    ax.set_ylim(0, bn_ymax)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"Household\npair {i+1}" for i in range(n_pairs)])
    ax.set_ylabel("Mean bottleneck size")
    ax.grid(axis="y", which="major", color=DASH_COLOR, linestyle="-", linewidth=0.2, alpha=0.25)
    ax.grid(axis="y", which="minor", color=DASH_COLOR, linestyle=":", linewidth=0.2, alpha=0.2)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    handles_bn = [
        Line2D([], [], marker="o", linestyle="", color=BN_COL_UNMASKED, markersize=3, label="Unmasked"),
        Line2D([], [], marker="o", linestyle="", color=BN_COL_MASKED, markersize=3, label="Masked"),
    ]
    ax.legend(handles=handles_bn, frameon=False, ncols=1, loc="upper right")



def main():
    df = pd.read_csv(high_conf_pairs_path)
    df_random = pd.read_csv(random_pairs_path)

    train_df = pd.read_csv(training_set_path)
    training_map = build_training_pair_map(train_df)

    bn_df = pd.read_csv(bottleneck_examples_path)

    maf_df = pd.read_csv(maf_table_path)
    maf_df["Site_norm"] = maf_df["Site"].map(_norm_site)
    maf_map = {str(r["Site_norm"]): float(r["MAF_analysis"]) / 100.0 for _, r in maf_df.iterrows()}

    pair_rows, bn_ymax = collect_pair_rows(
        df=df,
        bn_df=bn_df,
        pair_ids=selected_high_conf_pairs,
        depth_thresh=depth_thresh,
        training_map=training_map,
    )
    if not pair_rows:
        print("[ERROR] No valid pair rows found for plotting. Exiting.")
        return

    random_pairs = [int(x) for x in selected_random_pairs]
    nrows_top = len(random_pairs)

    # set these to easily turn on/off titles and xlabels for panel a
    SHOW_PANELA_TITLES_ON_ROW2 = False
    SHOW_PANELA_XLABEL_TOPROW = False
    SHOW_PANELA_XLABEL_ROW2 = True

    # Create figure and layout
    fig = plt.figure(figsize=(7, 8), constrained_layout=False)

    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[0.8, 1],
        hspace=0.3,
        left=0.12, right=0.9, bottom=0.05, top=0.97
    )

    # panel a: random pairs TV plots
    gs_top = outer[0].subgridspec(
        nrows=nrows_top, ncols=4,
        wspace=0.005,
        hspace=0.2
    )
    top_axes = [[fig.add_subplot(gs_top[r, c]) for c in range(4)] for r in range(nrows_top)]

    for r, pn in enumerate(random_pairs):
        sub100 = subset_random_pair(df_random, pair_n=pn, depth_thresh="100x")
        sub1000 = subset_random_pair(df_random, pair_n=pn, depth_thresh="1000x")

        draw_random_tv(
            top_axes[r][0], sub100, maf_map,
            mode="fixed", masked=False,
            title=RANDOM_COL_TITLES[0] if (r == 0 or SHOW_PANELA_TITLES_ON_ROW2) else "",
            show_ylabel=True, show_yticklabels=True
        )
        draw_random_tv(
            top_axes[r][1], sub1000, maf_map,
            mode="fixed", masked=False,
            title=RANDOM_COL_TITLES[1] if (r == 0 or SHOW_PANELA_TITLES_ON_ROW2) else "",
            show_ylabel=False, show_yticklabels=True
        )
        draw_random_tv(
            top_axes[r][2], sub1000, maf_map,
            mode="adaptive", masked=False,
            title=RANDOM_COL_TITLES[2] if (r == 0 or SHOW_PANELA_TITLES_ON_ROW2) else "",
            show_ylabel=False, show_yticklabels=True
        )
        draw_random_tv(
            top_axes[r][3], sub1000, maf_map,
            mode="adaptive", masked=True,
            title=RANDOM_COL_TITLES[3] if (r == 0 or SHOW_PANELA_TITLES_ON_ROW2) else "",
            show_ylabel=False, show_yticklabels=True
        )

        top_axes[r][0].text(
            -0.7, 0.5, f"Random pair {r+1}",
            transform=top_axes[r][0].transAxes,
            ha="right", va="center",
        )

    # shared xlabel for panel a for each row
    if SHOW_PANELA_XLABEL_TOPROW:
        fig.text(0.51, 0.77, "Allele frequency sample 1 (%)", ha="center")
    if SHOW_PANELA_XLABEL_ROW2:
        fig.text(0.51, 0.57, "Allele frequency sample 1 (%)", ha="center")

    # legend for TV points (used in both panels a and b)
    handles_tv = [
        Line2D([], [], marker="o", linestyle="", color=COL_SHARED, markersize=2, label="Shared"),
        Line2D([], [], marker="o", linestyle="", color=COL_UNSHARED, markersize=2, label="Unshared"),
        Line2D([], [], marker="o", linestyle="", color=COL_CONSCHANGE, markersize=2, label="Consensus change"),
    ]

    # panel a legend
    ax_leg_a = top_axes[0][2]
    ax_leg_a.legend(
        handles=handles_tv,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(-0.8, 1.39),
        borderaxespad=0.0,
    )

    # panel b and c: high-confidence pairs TV plots and bottleneck
    gs_bottom = outer[1].subgridspec(
        nrows=2, ncols=5,
        height_ratios=[1.0, 1.0],
        width_ratios=[0.08, 1.0, 0.3, 1.0, 0.15],
        hspace=0.33,
        wspace=0.005
    )

    # function to create a pair of TV axes (unmasked + masked)
    def make_pair_block(gs_cell):
        gs_pair = gs_cell.subgridspec(nrows=1, ncols=2, wspace=0.001)
        axU = fig.add_subplot(gs_pair[0, 0])
        axM = fig.add_subplot(gs_pair[0, 1])
        return axU, axM
    
    # create three pair blocks for household pairs
    ax_p1_u, ax_p1_m = make_pair_block(gs_bottom[0, 1])
    ax_p2_u, ax_p2_m = make_pair_block(gs_bottom[0, 3])
    ax_p3_u, ax_p3_m = make_pair_block(gs_bottom[1, 1])
    # bottleneck axis in the bottom row of the right column
    ax_bn = fig.add_subplot(gs_bottom[1, 3])

    axes_b = [(ax_p1_u, ax_p1_m), (ax_p2_u, ax_p2_m), (ax_p3_u, ax_p3_m)]

    # draw TV plots for each high-confidence pair
    for i, row in enumerate(pair_rows):
        pair_n = row["pair_n"]
        axU, axM = axes_b[i]

        draw_tv(
            axU, row["tv_unmasked"], maf_map,
            panel_name="unmasked", pair_n=pair_n, row_idx=i,
            show_ylabel=True, show_yticklabels=True
        )
        # Add pair label above both plots
        posU = axU.get_position()
        posM = axM.get_position() if axM.axison else posU
        x_center = 0.5 * (posU.x0 + posM.x1)
        y_top = max(posU.y1, posM.y1)

        fig.text(
            x_center - 0.01, y_top + 0.022,
            f"Household pair {i+1}",
            ha="center", va="bottom"
        )

        if row["tv_masked"].empty:
            axM.text(0.5, 0.5, "No data\nafter masking", ha="center", va="center")
            axM.set_axis_off()
        else:
            draw_tv(
                axM, row["tv_masked"], maf_map,
                panel_name="masked", pair_n=pair_n, row_idx=i,
                show_ylabel=False, show_yticklabels=True
            )

        # small headers
        axU.text(0.5, 1.02, "Unmasked", transform=axU.transAxes, ha="center", va="bottom")
        if axM.axison:
            axM.text(0.5, 1.02, "Masked", transform=axM.transAxes, ha="center", va="bottom")

    xlabel = "Allele frequency sample 1 (%)"
    ypad = 0.027

    # Pair3 (bottom-left)
    axU, axM = axes_b[2]
    posU, posM = axU.get_position(), axM.get_position()
    fig.text(
        0.5 * (posU.x0 + posM.x1),
        min(posU.y0, posM.y0) - ypad,
        xlabel,
        ha="center",
        va="top"
    )

    # Pair2 (top-right)
    axU, axM = axes_b[1]
    posU, posM = axU.get_position(), axM.get_position()
    fig.text(
        0.5 * (posU.x0 + posM.x1),
        min(posU.y0, posM.y0) - ypad,
        xlabel,
        ha="center",
        va="top"
    )


    # bottleneck plot (panel c) in bottom-right, same row height as pair3 block
    draw_bottleneck_panel(ax_bn, pair_rows, bn_ymax)

    # panel b legend
    ax_leg_b = axes_b[0][1]
    ax_leg_b.legend(
        handles=handles_tv,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(1.5, 1.3),
        borderaxespad=0.0,
    )


    # panel letters
    fig.text(0.01, 1.03, "a", fontweight="bold", fontsize=8, va="top")
    fig.text(0.01, 0.56, "b", fontweight="bold", fontsize=8, va="top")
    fig.text(0.50, 0.26, "c", fontweight="bold", fontsize=8, va="top")

    plot_filename = 'figure4_transmission_and_random_pairs.pdf'
    plot_path = os.path.join(figures_dir, plot_filename)
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Figure saved as '{plot_path}'")

if __name__ == "__main__":
    main()