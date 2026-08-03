"""Create publication-style Plotly figures for the streaming-memory study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "plots" / "streaming_memory_academic"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6B7280"
LIGHT_GRID = "#D9D9D9"


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text())


def rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def academic_layout(fig: go.Figure, *, width: int, height: int) -> None:
    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        font={"family": "Times New Roman, Times, serif", "size": 19, "color": "#111111"},
        margin={"l": 90, "r": 35, "t": 90, "b": 95},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 18},
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel={"font": {"family": "Arial, sans-serif", "size": 14}},
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1.3,
        linecolor="#222222",
        mirror=True,
        ticks="outside",
        tickwidth=1.2,
        tickcolor="#222222",
        gridcolor=LIGHT_GRID,
        zeroline=False,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.3,
        linecolor="#222222",
        mirror=True,
        ticks="outside",
        tickwidth=1.2,
        tickcolor="#222222",
        gridcolor=LIGHT_GRID,
        zerolinecolor="#888888",
        zerolinewidth=1,
    )


def export(fig: go.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUTPUT / f"{stem}.html",
        include_plotlyjs="cdn",
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    fig.write_image(OUTPUT / f"{stem}.png", width=fig.layout.width, height=fig.layout.height, scale=2)
    fig.write_image(
        OUTPUT / f"{stem}.jpeg",
        format="jpeg",
        width=fig.layout.width,
        height=fig.layout.height,
        scale=2,
    )


def eval_matrix(summary: dict) -> tuple[list[int], np.ndarray]:
    lengths = [int(length) for length in summary["eval_lengths"]]
    matrix = np.asarray(
        [
            [seed["eval_by_length"][str(length)]["success"] for length in lengths]
            for seed in summary["per_seed"]
        ],
        dtype=float,
    )
    return lengths, matrix


def add_seed_line(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    x: list[int],
    values: np.ndarray,
    name: str,
    color: str,
    showlegend: bool,
) -> None:
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            mode="lines+markers",
            line={"color": color, "width": 3},
            marker={"color": color, "size": 10},
            error_y={"type": "data", "array": sd, "visible": True, "thickness": 1.4},
            hovertemplate=f"{name}<br>Length=%{{x}}<br>Mean=%{{y:.3f}}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    jitter_scale = max(x) * 0.008
    for seed_index, seed_values in enumerate(values):
        offset = (seed_index - (len(values) - 1) / 2) * jitter_scale
        fig.add_trace(
            go.Scatter(
                x=[value + offset for value in x],
                y=seed_values,
                mode="markers",
                showlegend=False,
                legendgroup=name,
                marker={
                    "symbol": "circle-open",
                    "color": color,
                    "size": 8,
                    "line": {"width": 1.5},
                },
                hovertemplate=f"{name}, seed {seed_index}<br>Success=%{{y:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def frozen_generalization_figure() -> go.Figure:
    passive_memory = load_json(
        "stream_rl/logs/autoresearch_streaming_q/"
        "passive_train_L25_100k_eval_25_50_100/summary.json"
    )
    passive_control = load_json(
        "stream_rl/logs/autoresearch_streaming_q/"
        "passive_train_L25_no_memory_100k_eval_25_50_100/summary.json"
    )
    active_memory = load_json(
        "stream_rl/logs/autoresearch_streaming_q/"
        "active_train_L5_200k_qtrace_eval_5_10_25/summary.json"
    )
    active_control = load_json(
        "stream_rl/logs/autoresearch_streaming_q/"
        "active_train_L5_no_memory_200k_qtrace_eval_5_10_25/summary.json"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("(a) Passive T-maze: trained at L=25", "(b) Active T-maze: trained at L=5"),
        horizontal_spacing=0.12,
    )
    for col, memory, control in (
        (1, passive_memory, passive_control),
        (2, active_memory, active_control),
    ):
        lengths, memory_values = eval_matrix(memory)
        control_lengths, control_values = eval_matrix(control)
        assert lengths == control_lengths
        add_seed_line(
            fig,
            row=1,
            col=col,
            x=lengths,
            values=memory_values,
            name="Frozen memory",
            color=BLUE,
            showlegend=col == 1,
        )
        add_seed_line(
            fig,
            row=1,
            col=col,
            x=lengths,
            values=control_values,
            name="No memory",
            color=ORANGE,
            showlegend=col == 1,
        )
        fig.update_xaxes(title_text="Evaluation corridor length", tickvals=lengths, row=1, col=col)
        fig.update_yaxes(range=[0, 1.07], tick0=0, dtick=0.2, row=1, col=col)
    fig.update_yaxes(title_text="Success rate", row=1, col=1)
    academic_layout(fig, width=1450, height=620)
    fig.add_annotation(
        text="Mean ± 1 SD; open circles are individual seeds (n=3); 1,000 frozen greedy episodes per seed and length.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.22,
        showarrow=False,
        font={"size": 16},
    )
    return fig


def direct_training_figure() -> go.Figure:
    panels = [
        (
            "(a) Passive T-maze",
            ["L=25<br>100k", "L=50<br>100k", "L=50<br>300k"],
            [[1, 1, 1], [0.521, 0, 0.499], [0, 0.501, 0]],
            BLUE,
        ),
        (
            "(b) Active T-maze",
            ["L=5<br>200k", "L=10<br>500k", "L=25<br>500k"],
            [[1, 1, 1], [0.493, 1, 1], [0, 0, 0]],
            GREEN,
        ),
    ]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[panel[0] for panel in panels])
    for col, (_, labels, groups, color) in enumerate(panels, start=1):
        means = [float(np.mean(group)) for group in groups]
        sd = [float(np.std(group, ddof=1)) for group in groups]
        fig.add_trace(
            go.Bar(
                x=list(range(len(labels))),
                y=means,
                error_y={"type": "data", "array": sd, "visible": True, "thickness": 1.4},
                marker={"color": color, "opacity": 0.42, "line": {"color": color, "width": 1.5}},
                showlegend=False,
                hovertemplate="%{x}<br>Mean success=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        for group_index, group in enumerate(groups):
            for seed_index, value in enumerate(group):
                fig.add_trace(
                    go.Scatter(
                        x=[group_index + (seed_index - 1) * 0.08],
                        y=[value],
                        mode="markers",
                        marker={"color": color, "size": 10, "symbol": "circle-open"},
                        showlegend=False,
                        hovertemplate=f"Seed {seed_index}: %{{y:.3f}}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
        fig.update_yaxes(range=[0, 1.08], tick0=0, dtick=0.2, row=1, col=col)
        fig.update_xaxes(tickvals=list(range(len(labels))), ticktext=labels, row=1, col=col)
    fig.update_yaxes(title_text="Frozen-evaluation success", row=1, col=1)
    academic_layout(fig, width=1450, height=700)
    fig.update_layout(margin={"l": 90, "r": 35, "t": 90, "b": 150})
    fig.add_annotation(
        text="Training length / transitions",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.13,
        showarrow=False,
        font={"size": 20},
    )
    fig.add_annotation(
        text="Bars show means; error bars show 1 SD; open circles show individual seeds (n=3).",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.27,
        showarrow=False,
        font={"size": 16},
    )
    return fig


def popgym_data() -> list[dict]:
    random = load_json("stream_rl/logs/autoresearch_popgym_native/random_baselines.json")
    environments = [
        (
            "AutoencodeEasy",
            "popgym-AutoencodeEasy-v0",
            "stream_rl/logs/autoresearch_popgym_native/autoencode_easy_memory_200k/summary.json",
            "logs/autoresearch_popgym_native/autoencode_easy_stack_100k/summary.json",
            "LIFO stack",
        ),
        (
            "ConcentrationEasy",
            "popgym-ConcentrationEasy-v0",
            "stream_rl/logs/autoresearch_popgym_native/concentration_easy_memory_200k/summary.json",
            "logs/action_memory_confirm/concentration_shared_200k/summary.json",
            "Assoc. scorer",
        ),
        (
            "RepeatFirstEasy",
            "popgym-RepeatFirstEasy-v0",
            "stream_rl/logs/autoresearch_popgym_native/repeatfirst_easy_memory_200k/summary.json",
            "stream_rl/logs/autoresearch_popgym_native/repeatfirst_easy_anchor_only_bandit_300k/summary.json",
            "Anchor slot",
        ),
        (
            "BattleshipEasy",
            "popgym-BattleshipEasy-v0",
            "stream_rl/logs/autoresearch_popgym_native/battleship_easy_memory_200k/summary.json",
            "logs/action_memory_confirm/battleship_minimal_shared_200k/summary.json",
            "Spatial scorer",
        ),
    ]
    rows = []
    for title, env_id, ssm_path, structured_path, structured_label in environments:
        ssm = load_json(ssm_path)
        structured = load_json(structured_path)
        rows.append(
            {
                "title": title,
                "random": random[env_id]["per_seed"],
                "ssm": [seed["eval_return"] for seed in ssm["per_seed"]],
                "structured": [seed["eval_return"] for seed in structured["per_seed"]],
                "structured_label": structured_label,
            }
        )
    return rows


def popgym_figure() -> go.Figure:
    rows = popgym_data()
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"({chr(97 + index)}) {row['title']}" for index, row in enumerate(rows)],
        vertical_spacing=0.16,
        horizontal_spacing=0.11,
    )
    colors = [GRAY, BLUE, GREEN]
    base_labels = ["Random", "Frozen SSM", "Task-aligned"]
    for index, data in enumerate(rows):
        subplot_row, subplot_col = index // 2 + 1, index % 2 + 1
        values = [data["random"], data["ssm"], data["structured"]]
        means = [float(np.mean(group)) for group in values]
        sd = [float(np.std(group, ddof=1)) if len(group) > 1 else 0.0 for group in values]
        labels = ["Random", "Frozen SSM", data["structured_label"]]
        for method_index, (label, group, method_mean, method_sd, color) in enumerate(
            zip(labels, values, means, sd, colors)
        ):
            fig.add_trace(
                go.Bar(
                    x=[method_index],
                    y=[method_mean],
                    name=base_labels[method_index],
                    legendgroup=base_labels[method_index],
                    showlegend=index == 0,
                    error_y={"type": "data", "array": [method_sd], "visible": len(group) > 1},
                    marker={"color": color, "opacity": 0.42, "line": {"color": color, "width": 1.4}},
                    hovertemplate=f"{label}<br>Mean return=%{{y:.3f}}<extra></extra>",
                ),
                row=subplot_row,
                col=subplot_col,
            )
            for seed_index, value in enumerate(group):
                fig.add_trace(
                    go.Scatter(
                        x=[method_index + (seed_index - (len(group) - 1) / 2) * 0.07],
                        y=[value],
                        mode="markers",
                        marker={"color": color, "size": 9, "symbol": "circle-open"},
                        showlegend=False,
                        hovertemplate=f"{label}, seed {seed_index}<br>Return=%{{y:.3f}}<extra></extra>",
                    ),
                    row=subplot_row,
                    col=subplot_col,
                )
        fig.update_xaxes(tickvals=[0, 1, 2], ticktext=labels, row=subplot_row, col=subplot_col)
        fig.update_yaxes(range=[-1.32, 1.1], tick0=-1, dtick=0.5, row=subplot_row, col=subplot_col)
    fig.update_yaxes(title_text="Frozen greedy return", row=1, col=1)
    fig.update_yaxes(title_text="Frozen greedy return", row=2, col=1)
    academic_layout(fig, width=1500, height=1050)
    fig.update_layout(barmode="overlay")
    fig.add_annotation(
        text="Mean ± 1 SD with seed-level observations (n=3 for every method).",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
        font={"size": 16},
    )
    return fig


def action_memory_comparison_figure() -> go.Figure:
    random = load_json("stream_rl/logs/autoresearch_popgym_native/random_baselines.json")
    panels = [
        {
            "title": "(a) ConcentrationEasy",
            "env": "popgym-ConcentrationEasy-v0",
            "generic": "stream_rl/logs/autoresearch_popgym_native/concentration_easy_memory_200k/summary.json",
            "mask": "logs/action_memory_confirm/concentration_mask_only_100k/summary.json",
            "shared": "logs/action_memory_confirm/concentration_shared_200k/summary.json",
        },
        {
            "title": "(b) BattleshipEasy",
            "env": "popgym-BattleshipEasy-v0",
            "generic": "stream_rl/logs/autoresearch_popgym_native/battleship_easy_memory_200k/summary.json",
            "mask": "logs/action_memory_confirm/battleship_mask_only_100k/summary.json",
            "shared": "logs/action_memory_confirm/battleship_minimal_shared_200k/summary.json",
        },
    ]
    labels = ["Random", "Frozen SSM", "Mask only", "Shared scorer"]
    colors = [GRAY, BLUE, ORANGE, GREEN]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[panel["title"] for panel in panels])
    for col, panel in enumerate(panels, start=1):
        generic = load_json(panel["generic"])
        mask = load_json(panel["mask"])
        shared = load_json(panel["shared"])
        groups = [
            random[panel["env"]]["per_seed"],
            [seed["eval_return"] for seed in generic["per_seed"]],
            [seed["eval_return"] for seed in mask["per_seed"]],
            [seed["eval_return"] for seed in shared["per_seed"]],
        ]
        for method_index, (label, color, values) in enumerate(zip(labels, colors, groups)):
            method_mean = float(np.mean(values))
            method_sd = float(np.std(values, ddof=1))
            fig.add_trace(
                go.Bar(
                    x=[method_index],
                    y=[method_mean],
                    name=label,
                    legendgroup=label,
                    showlegend=col == 1,
                    error_y={"type": "data", "array": [method_sd], "visible": True},
                    marker={"color": color, "opacity": 0.42, "line": {"color": color, "width": 1.4}},
                    hovertemplate=f"{label}<br>Mean return=%{{y:.3f}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
            for seed_index, value in enumerate(values):
                fig.add_trace(
                    go.Scatter(
                        x=[method_index + (seed_index - 1) * 0.07],
                        y=[value],
                        mode="markers",
                        marker={"color": color, "size": 10, "symbol": "circle-open"},
                        showlegend=False,
                        hovertemplate=f"{label}, seed {seed_index}<br>Return=%{{y:.3f}}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
        fig.update_xaxes(tickvals=list(range(4)), ticktext=labels, row=1, col=col)
        fig.update_yaxes(range=[-1.3, 1.02], tick0=-1, dtick=0.5, row=1, col=col)
    fig.update_yaxes(title_text="Frozen greedy return", row=1, col=1)
    academic_layout(fig, width=1500, height=760)
    fig.update_layout(margin={"l": 90, "r": 35, "t": 90, "b": 135})
    fig.add_annotation(
        text="Mean ± 1 SD; open circles show individual seeds (n=3); 1,000 frozen episodes per seed.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.23,
        showarrow=False,
        font={"size": 16},
    )
    return fig


def binned_monitor(relative_directory: str, metric: str, bins: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = ROOT / relative_directory
    seed_curves = []
    max_steps = []
    for seed in (0, 1, 2):
        with (directory / f"monitor_seed_{seed}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        steps = np.asarray([int(row["step"]) for row in rows], dtype=float)
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        max_steps.append(float(steps.max()))
        seed_curves.append((steps, values))
    shared_max = min(max_steps)
    edges = np.linspace(0, shared_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    curves = []
    for steps, values in seed_curves:
        curve = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = values[(steps > lower) & (steps <= upper)]
            curve.append(float(selected.mean()))
        curves.append(curve)
    matrix = np.asarray(curves, dtype=float)
    return centers, matrix.mean(axis=0), matrix.std(axis=0, ddof=1)


def add_learning_curve(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    directory: str,
    metric: str,
    name: str,
    color: str,
    showlegend: bool,
) -> None:
    x, mean, sd = binned_monitor(directory, metric)
    lower = mean - sd
    upper = mean + sd
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line={"width": 0, "color": color},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line={"width": 0, "color": color},
            fill="tonexty",
            fillcolor=rgba(color, 0.2),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line={"color": color, "width": 3},
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            hovertemplate=f"{name}<br>Step=%{{x:.0f}}<br>Mean=%{{y:.3f}}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def learning_curves_figure() -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "(a) Passive T-maze, L=25",
            "(b) Active T-maze, L=5",
            "(c) AutoencodeEasy",
            "(d) RepeatFirstEasy",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.11,
    )
    panels = [
        (
            1,
            1,
            "success",
            [
                ("stream_rl/logs/autoresearch_streaming_q/passive_train_L25_100k_eval_25_50_100", "Frozen memory", BLUE),
                ("stream_rl/logs/autoresearch_streaming_q/passive_train_L25_no_memory_100k_eval_25_50_100", "No memory", ORANGE),
            ],
        ),
        (
            1,
            2,
            "success",
            [
                ("stream_rl/logs/autoresearch_streaming_q/active_train_L5_200k_qtrace_eval_5_10_25", "Frozen memory", BLUE),
                ("stream_rl/logs/autoresearch_streaming_q/active_train_L5_no_memory_200k_qtrace_eval_5_10_25", "No memory", ORANGE),
            ],
        ),
        (
            2,
            1,
            "return",
            [
                ("logs/autoresearch_popgym_native/autoencode_easy_stack_100k", "LIFO stack", GREEN),
                ("stream_rl/logs/autoresearch_popgym_native/autoencode_easy_memory_200k", "Frozen SSM", BLUE),
            ],
        ),
        (
            2,
            2,
            "return",
            [
                ("stream_rl/logs/autoresearch_popgym_native/repeatfirst_easy_anchor_only_bandit_300k", "Anchor slot", GREEN),
                ("stream_rl/logs/autoresearch_popgym_native/repeatfirst_easy_memory_200k", "Frozen SSM", BLUE),
            ],
        ),
    ]
    shown = set()
    for row, col, metric, methods in panels:
        for directory, name, color in methods:
            add_learning_curve(
                fig,
                row=row,
                col=col,
                directory=directory,
                metric=metric,
                name=name,
                color=color,
                showlegend=name not in shown,
            )
            shown.add(name)
        fig.update_xaxes(title_text="Environment transitions", exponentformat="power", row=row, col=col)
    fig.update_yaxes(title_text="Online success", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    fig.update_yaxes(title_text="Online return", range=[-0.65, 1.08], row=2, col=1)
    fig.update_yaxes(range=[-0.65, 1.08], row=2, col=2)
    academic_layout(fig, width=1500, height=1050)
    fig.add_annotation(
        text="Lines show 40-bin seed means; ribbons show ±1 SD (n=3). Training includes ε-greedy exploration.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
        font={"size": 16},
    )
    return fig


def action_memory_learning_figure() -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("(a) ConcentrationEasy", "(b) BattleshipEasy"),
        horizontal_spacing=0.12,
    )
    panels = [
        (
            1,
            [
                (
                    "logs/action_memory_confirm/concentration_shared_200k",
                    "Shared scorer",
                    GREEN,
                ),
                (
                    "logs/action_memory_confirm/concentration_mask_only_100k",
                    "Mask only",
                    ORANGE,
                ),
                (
                    "stream_rl/logs/autoresearch_popgym_native/concentration_easy_memory_200k",
                    "Frozen SSM",
                    BLUE,
                ),
            ],
        ),
        (
            2,
            [
                (
                    "logs/action_memory_confirm/battleship_minimal_shared_200k",
                    "Shared scorer",
                    GREEN,
                ),
                (
                    "logs/action_memory_confirm/battleship_mask_only_100k",
                    "Mask only",
                    ORANGE,
                ),
                (
                    "stream_rl/logs/autoresearch_popgym_native/battleship_easy_memory_200k",
                    "Frozen SSM",
                    BLUE,
                ),
            ],
        ),
    ]
    shown = set()
    for col, methods in panels:
        for directory, name, color in methods:
            add_learning_curve(
                fig,
                row=1,
                col=col,
                directory=directory,
                metric="return",
                name=name,
                color=color,
                showlegend=name not in shown,
            )
            shown.add(name)
        fig.update_xaxes(title_text="Environment transitions", exponentformat="power", row=1, col=col)
        fig.update_yaxes(range=[-1.3, 1.02], tick0=-1, dtick=0.5, row=1, col=col)
    fig.update_yaxes(title_text="Online episodic return", row=1, col=1)
    academic_layout(fig, width=1500, height=760)
    fig.update_layout(margin={"l": 90, "r": 35, "t": 90, "b": 135})
    fig.add_annotation(
        text="Lines show 40-bin seed means; ribbons show ±1 SD (n=3). Training includes ε-greedy exploration.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.23,
        showarrow=False,
        font={"size": 16},
    )
    return fig


def main() -> None:
    figures = {
        "figure_1_frozen_length_generalization": frozen_generalization_figure(),
        "figure_2_direct_training_robustness": direct_training_figure(),
        "figure_3_popgym_easy_comparison": popgym_figure(),
        "figure_4_online_learning_curves": learning_curves_figure(),
        "figure_5_action_memory_comparison": action_memory_comparison_figure(),
        "figure_6_action_memory_learning_curves": action_memory_learning_figure(),
    }
    for stem, figure in figures.items():
        export(figure, stem)
        print(OUTPUT / stem)


if __name__ == "__main__":
    main()
