"""RL training-log viewer.

Three views (tabs): line plots (with CI bands across seeds), a heatmap of a
metric over two run parameters (e.g. corridor_length x cell type), and box
plots of a per-seed summary metric across runs.
"""
import io
import itertools
import json
from pathlib import Path

import numpy as np
import streamlit as st

from stream_rl.visual_utils.base import LogLoader, CurveAggregator, PlotStyle, AxisStyle
from stream_rl.visual_utils.plotter import CurvePlotter
from stream_rl.visual_utils.sweeps import RunConfigLoader, get_by_path, normalize_structure_config

st.set_page_config(page_title='RL Log Viewer', layout='wide')
DEFAULT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# --------------------------------------------------------------------------
# Directory browser — streamlit has no native folder picker, so this is a
# small in-app navigator: shows the current path, an "up" control, and a
# selectbox of subdirectories to descend into. State lives in
# st.session_state[f'{key}_path'] so it survives reruns.
# --------------------------------------------------------------------------
def dir_browser(key: str, default: str) -> str:
    state_key = f'{key}_path'
    if state_key not in st.session_state:
        start = Path(default)
        st.session_state[state_key] = str(start.resolve()) if start.exists() else str(Path('.').resolve())

    current = Path(st.session_state[state_key])
    st.caption(f'📁 `{current}`')

    manual = st.text_input('Or type/paste a path directly', value='', key=f'{key}_manual', placeholder=str(current))
    if manual:
        candidate = Path(manual)
        if candidate.exists() and candidate.is_dir():
            st.session_state[state_key] = str(candidate.resolve())
            current = candidate.resolve()
        else:
            st.warning(f"'{manual}' is not a directory.")

    cols = st.columns([1, 4])
    if cols[0].button('⬆ Up', key=f'{key}_up', disabled=current.parent == current):
        st.session_state[state_key] = str(current.parent)
        st.rerun()

    try:
        subdirs = sorted(p.name for p in current.iterdir() if p.is_dir())
    except (PermissionError, FileNotFoundError):
        subdirs = []

    if subdirs:
        chosen = cols[1].selectbox('Descend into', options=['—'] + subdirs, key=f'{key}_sel')
        if chosen != '—' and st.button(f"Enter '{chosen}'", key=f'{key}_enter'):
            st.session_state[state_key] = str(current / chosen)
            st.rerun()
    else:
        cols[1].caption('(no subdirectories here)')

    return str(current)


@st.cache_data(show_spinner=False)
def discover_runs(log_root: str):
    root = Path(log_root)
    runs = []
    if not root.exists():
        return runs
    for series_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in series_dir.iterdir() if p.is_dir()):
            seed_files = sorted(run_dir.glob('monitor_seed_*.csv')) or sorted(run_dir.glob('metrics_seed_*.csv'))
            if not seed_files:
                continue
            manifest_path = run_dir / 'manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            runs.append({
                'run_root': str(run_dir),
                'series_name': series_dir.name,
                'run_id': run_dir.name,
                'variant_name': manifest.get('variant_name', run_dir.name),
                'n_seeds': len(seed_files),
            })
    return runs


@st.cache_data(show_spinner=False)
def list_columns_cached(run_root: str):
    return LogLoader.list_columns(Path(run_root))


@st.cache_data(show_spinner=False)
def load_curves_cached(run_root: str, source: str, y_col: str, smoothing: str, window_steps: float, sigma_steps: float):
    loader = LogLoader(Path(run_root), source=source, y_col=y_col, smoothing=smoothing,
                        window_steps=window_steps, sigma_steps=sigma_steps)
    return loader.load_seed_curves()


@st.cache_data(show_spinner=False)
def load_summary_cached(run_root: str, source: str, y_col: str, reduction: str, last_k: int):
    loader = LogLoader(Path(run_root), source=source, y_col=y_col)
    return loader.load_seed_summary(reduction=reduction, last_k=last_k)


@st.cache_data(show_spinner=False)
def load_run_config_cached(run_root: str):
    try:
        return normalize_structure_config(RunConfigLoader(Path(run_root)).load())
    except FileNotFoundError:
        return {}


def run_label(r: dict) -> str:
    return f"{r['series_name']} / {r['variant_name']}  (n={r['n_seeds']} seeds)"


# ===========================================================================
if 'curves' not in st.session_state:
    st.session_state.curves = []
if '_next_curve_id' not in st.session_state:
    st.session_state._next_curve_id = 0

st.title('RL Training Log Viewer')

with st.sidebar:
    st.header('Data source')
    log_root = dir_browser('log_root', default='logs')

    runs = discover_runs(log_root)
    if not runs:
        st.warning(f"No runs found under '{log_root}'. Expected layout: <log_root>/<series>/<run_id>/monitor_seed_<seed>.csv")

    run_options = {run_label(r): r['run_root'] for r in runs}

tab_line, tab_heatmap, tab_box = st.tabs(['📈 Line plot', '🟦 Heatmap', '📦 Box plot'])

# ===========================================================================
# Line plot tab
# ===========================================================================
with tab_line:
    left, right = st.columns([1, 2])

    with left:
        st.subheader('Add a curve')
        selected_label = st.selectbox('Run', options=list(run_options.keys()) if run_options else ['<none found>'], key='line_run_select')
        selected_root = run_options.get(selected_label)

        available_cols = list_columns_cached(selected_root) if selected_root else []
        col_labels = [f'{src}: {col}' for src, col in available_cols]
        default_idx = next((i for i, (src, col) in enumerate(available_cols)
                             if col in ('info.returned_episode_returns', 'returned_episode_returns')), 0)
        col_choice = st.selectbox('Metric', options=col_labels if col_labels else ['<no columns found>'],
                                   index=default_idx if col_labels else 0, key='line_col_select')

        smoothing_mode = st.radio('Smoothing', options=['window', 'gaussian', 'none'], horizontal=True, key='line_smoothing')
        window_steps = st.number_input('Window (env steps)', min_value=0, value=5000, step=500, key='line_window',
                                        disabled=smoothing_mode != 'window')
        sigma_steps = st.number_input('Gaussian sigma (env steps)', min_value=0, value=2000, step=250, key='line_sigma',
                                       disabled=smoothing_mode != 'gaussian')
        confidence = st.slider('Confidence level for CI', min_value=0.50, max_value=0.99, value=0.95, step=0.01, key='line_ci')

        color_idx = len(st.session_state.curves) % len(DEFAULT_COLORS)
        curve_color = st.color_picker('Color', value=DEFAULT_COLORS[color_idx], key='line_color')
        curve_linestyle = st.selectbox('Line style', options=['-', '--', '-.', ':'], index=0, key='line_style')
        curve_linewidth = st.slider('Line width', 0.5, 5.0, 2.0, 0.5, key='line_width')
        curve_alpha_ci = st.slider('CI band opacity', 0.0, 1.0, 0.2, 0.05, key='line_alpha')

        if st.button('Add curve to plot', disabled=not (run_options and col_labels), type='primary'):
            src, col = available_cols[col_labels.index(col_choice)]
            cid = st.session_state._next_curve_id
            st.session_state._next_curve_id += 1
            st.session_state.curves.append({
                'id': cid,
                'run_root': selected_root,
                'source': src,
                'y_col': col,
                'legend': selected_label.split('  (n=')[0],
                'color': curve_color,
                'linestyle': curve_linestyle,
                'linewidth': curve_linewidth,
                'alpha_ci': curve_alpha_ci,
            })

        st.divider()
        st.subheader('Plot customization')
        xlabel = st.text_input('X-axis label', value='Environment steps', key='line_xlabel')
        ylabel = st.text_input('Y-axis label', value=(col_choice.split(': ', 1)[-1] if col_labels else 'Value'), key='line_ylabel')
        title = st.text_input('Title', value='', key='line_title')
        c1, c2 = st.columns(2)
        xscale = c1.selectbox('X scale', options=['linear', 'log'], index=0, key='line_xscale')
        yscale = c2.selectbox('Y scale', options=['linear', 'log'], index=0, key='line_yscale')
        legend_loc = st.selectbox('Legend position', options=['best', 'upper left', 'upper right', 'lower left', 'lower right'], index=0, key='line_legend_loc')
        figsize_w = st.slider('Figure width', 4.0, 16.0, 9.0, 0.5, key='line_figw')
        figsize_h = st.slider('Figure height', 3.0, 12.0, 5.0, 0.5, key='line_figh')

    with right:
        st.subheader('Curves in current plot — rename legends here')
        if not st.session_state.curves:
            st.info('Add at least one curve from the left panel.')
        else:
            for i, cfg in enumerate(st.session_state.curves):
                row = st.columns([3, 3, 1, 1])
                row[0].markdown(
                    f"<div style='width:16px;height:16px;background:{cfg['color']};border-radius:3px;display:inline-block;margin-right:6px;'></div>"
                    f"`{cfg['source']}:{cfg['y_col']}` — {Path(cfg['run_root']).name}",
                    unsafe_allow_html=True,
                )
                cfg['legend'] = row[1].text_input('Legend', value=cfg['legend'], key=f"legend_{cfg['id']}", label_visibility='collapsed')
                cfg['linestyle'] = row[2].selectbox('style', options=['-', '--', '-.', ':'],
                                                     index=['-', '--', '-.', ':'].index(cfg['linestyle']),
                                                     key=f"ls_{cfg['id']}", label_visibility='collapsed')
                if row[3].button('✕', key=f"remove_{cfg['id']}"):
                    st.session_state.curves.pop(i)
                    st.rerun()

        render = st.button('Render plot', type='primary', disabled=not st.session_state.curves)

        if render:
            axis_style = AxisStyle(xlabel=xlabel, ylabel=ylabel, title=title, xscale=xscale, yscale=yscale,
                                    legend_loc=legend_loc, figsize=(figsize_w, figsize_h))
            plotter = CurvePlotter(axis_style=axis_style)
            errors = []
            for cfg in st.session_state.curves:
                try:
                    curves = load_curves_cached(cfg['run_root'], cfg['source'], cfg['y_col'],
                                                 smoothing_mode, float(window_steps), float(sigma_steps))
                except (FileNotFoundError, ValueError) as e:
                    errors.append(f"{cfg['legend']}: {e}")
                    continue
                style = PlotStyle(label=f"{cfg['legend']} (n={len(curves)})", color=cfg['color'],
                                   linestyle=cfg['linestyle'], linewidth=cfg['linewidth'], alpha_ci=cfg['alpha_ci'])
                if len(curves) == 1:
                    x, y = curves[0]
                    plotter.add_ci_curve(x, y, y, y, style)
                else:
                    grid, mean, lo, hi = CurveAggregator(curves, confidence=confidence).aggregate()
                    plotter.add_ci_curve(grid, mean, lo, hi, style)
            for err in errors:
                st.error(err)
            fig = plotter.finalize()
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=180)
            plotter.close()
            st.download_button('Download PNG', data=buf.getvalue(), file_name='curves_comparison.png', mime='image/png')

# ===========================================================================
# Heatmap tab
# ===========================================================================
with tab_heatmap:
    st.subheader('Heatmap: metric over two run parameters')
    st.caption(
        "Select every run to include (usually all seeds/variants of one sweep), then pick two "
        "dotted-path parameters read from each run's structure.json (e.g. env.kwargs.corridor_length, "
        "agent.cell) to use as the grid axes. Cells average the metric over every seed that shares "
        "the same (param1, param2) pair."
    )
    heat_runs = st.multiselect('Runs to include', options=list(run_options.keys()), key='heat_runs')

    if heat_runs:
        sample_root = run_options[heat_runs[0]]
        sample_cols = list_columns_cached(sample_root)
        sample_cfg = load_run_config_cached(sample_root)

        c1, c2 = st.columns(2)
        param1_path = c1.text_input('Row parameter (dotted path)', value='env.kwargs.corridor_length', key='heat_p1')
        param2_path = c2.text_input('Column parameter (dotted path)', value='agent.name', key='heat_p2')

        col_labels = [f'{src}: {col}' for src, col in sample_cols]
        metric_choice = st.selectbox('Metric', options=col_labels if col_labels else ['<none>'], key='heat_metric')
        reduction = st.selectbox('Per-seed reduction', options=['final', 'mean_last_k', 'max', 'mean'], key='heat_reduction')
        last_k = st.number_input('last_k (for mean_last_k)', min_value=1, value=20, key='heat_last_k')

        with st.expander('Sample config for the first selected run (for finding dotted paths)'):
            st.json(sample_cfg)

        if st.button('Build heatmap', type='primary', disabled=not col_labels):
            src, col = sample_cols[col_labels.index(metric_choice)]
            rows_by_key = {}
            errors = []
            for label in heat_runs:
                root = run_options[label]
                cfg = load_run_config_cached(root)
                try:
                    p1 = get_by_path(cfg, param1_path)
                    p2 = get_by_path(cfg, param2_path)
                except KeyError as e:
                    errors.append(f'{label}: {e}')
                    continue
                try:
                    values = load_summary_cached(root, src, col, reduction, int(last_k))
                except (FileNotFoundError, ValueError) as e:
                    errors.append(f'{label}: {e}')
                    continue
                rows_by_key.setdefault((p1, p2), []).extend(values)

            for err in errors:
                st.warning(err)

            if rows_by_key:
                row_vals = sorted({k[0] for k in rows_by_key}, key=lambda v: (0, v) if isinstance(v, (int, float)) else (1, str(v)))
                col_vals = sorted({k[1] for k in rows_by_key}, key=lambda v: (0, v) if isinstance(v, (int, float)) else (1, str(v)))
                matrix = np.full((len(row_vals), len(col_vals)), np.nan)
                for (p1, p2), values in rows_by_key.items():
                    matrix[row_vals.index(p1), col_vals.index(p2)] = float(np.mean(values))

                axis_style = AxisStyle(xlabel=param2_path, ylabel=param1_path,
                                        title=f'{metric_choice} ({reduction})', figsize=(max(6, len(col_vals) * 1.2), max(4, len(row_vals) * 0.8)))
                plotter = CurvePlotter(axis_style=axis_style)
                plotter.add_heatmap(matrix, [str(v) for v in row_vals], [str(v) for v in col_vals],
                                     colorbar_label=metric_choice)
                fig = plotter.finalize()
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=180)
                plotter.close()
                st.download_button('Download PNG', data=buf.getvalue(), file_name='heatmap.png', mime='image/png', key='heat_dl')
            else:
                st.error('No data matched — check the parameter paths against the sample config above.')
    else:
        st.info('Select at least one run above.')

# ===========================================================================
# Box plot tab
# ===========================================================================
with tab_box:
    st.subheader('Box plot: per-seed metric distribution across runs')
    box_runs = st.multiselect('Runs to include', options=list(run_options.keys()), key='box_runs')

    if box_runs:
        sample_root = run_options[box_runs[0]]
        sample_cols = list_columns_cached(sample_root)
        col_labels = [f'{src}: {col}' for src, col in sample_cols]
        metric_choice = st.selectbox('Metric', options=col_labels if col_labels else ['<none>'], key='box_metric')
        reduction = st.selectbox('Per-seed reduction', options=['final', 'mean_last_k', 'max', 'mean'], key='box_reduction')
        last_k = st.number_input('last_k (for mean_last_k)', min_value=1, value=20, key='box_last_k')

        if 'box_legends' not in st.session_state:
            st.session_state.box_legends = {}
        st.caption('Rename box labels:')
        legend_cols = st.columns(min(len(box_runs), 4)) if box_runs else []
        for i, label in enumerate(box_runs):
            default_name = run_options[label] and Path(run_options[label]).name
            key = f'box_legend_{label}'
            st.session_state.box_legends[label] = legend_cols[i % len(legend_cols)].text_input(
                label.split('  (n=')[0][:24], value=st.session_state.box_legends.get(label, default_name), key=key,
            )

        if st.button('Build box plot', type='primary', disabled=not col_labels):
            src, col = sample_cols[col_labels.index(metric_choice)]
            data, labels, colors, errors = [], [], [], []
            for i, label in enumerate(box_runs):
                root = run_options[label]
                try:
                    values = load_summary_cached(root, src, col, reduction, int(last_k))
                except (FileNotFoundError, ValueError) as e:
                    errors.append(f'{label}: {e}')
                    continue
                if not values:
                    errors.append(f'{label}: no data')
                    continue
                data.append(values)
                labels.append(st.session_state.box_legends.get(label, label))
                colors.append(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

            for err in errors:
                st.warning(err)

            if data:
                axis_style = AxisStyle(xlabel='', ylabel=metric_choice, title=f'{metric_choice} ({reduction})',
                                        figsize=(max(6, len(data) * 1.5), 5))
                plotter = CurvePlotter(axis_style=axis_style)
                plotter.add_boxplot(data, labels, colors)
                fig = plotter.finalize()
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=180)
                plotter.close()
                st.download_button('Download PNG', data=buf.getvalue(), file_name='boxplot.png', mime='image/png', key='box_dl')
    else:
        st.info('Select at least one run above.')
