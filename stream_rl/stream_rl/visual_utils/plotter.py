"""Low-level plotting API on top of matplotlib.

Kept separate from LogLoader/CurveAggregator (visual_utils/base.py) so the
same figure can mix curves coming from different sources (e.g. one seed-
averaged curve plus one raw scatter of episode returns) without the plotter
caring where the data came from.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stream_rl.visual_utils.base import PlotStyle, AxisStyle


class CurvePlotter:
    def __init__(self, axis_style: AxisStyle = AxisStyle()):
        self.axis_style = axis_style
        self.fig, self.ax = plt.subplots(figsize=axis_style.figsize)
        self.ax.set_xscale(axis_style.xscale)
        self.ax.set_yscale(axis_style.yscale)

    def add_ci_curve(self, x: np.ndarray, mean: np.ndarray, lo: np.ndarray, hi: np.ndarray, style: PlotStyle):
        line, = self.ax.plot(
            x, mean,
            color=style.color,
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            marker=style.marker,
            label=style.label,
            zorder=style.zorder,
        )
        color = style.color or line.get_color()
        self.ax.fill_between(x, lo, hi, color=color, alpha=style.alpha_ci, zorder=style.zorder - 1)
        return line

    def add_scatter(self, x: np.ndarray, y: np.ndarray, style: PlotStyle, size: float = 8, alpha: float = 0.25):
        self.ax.scatter(x, y, s=size, alpha=alpha, color=style.color, label=style.label, zorder=style.zorder - 1)

    def add_hline(self, y: float, style: PlotStyle):
        self.ax.axhline(y, color=style.color or "gray", linestyle=style.linestyle,
                         linewidth=style.linewidth, label=style.label)

    def add_boxplot(self, data: list, labels: list, colors: list = None):
        bp = self.ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        if colors:
            for patch, color in zip(bp['boxes'], colors):
                if color:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
        self.ax.tick_params(axis='x', rotation=30)
        return bp

    def add_heatmap(self, matrix: "np.ndarray", row_labels: list, col_labels: list,
                     cmap: str = 'viridis', annotate: bool = True, fmt: str = '.3g',
                     colorbar_label: str = ''):
        im = self.ax.imshow(matrix, cmap=cmap, aspect='auto', origin='lower')
        self.ax.set_xticks(range(len(col_labels)))
        self.ax.set_xticklabels(col_labels, rotation=30, ha='right')
        self.ax.set_yticks(range(len(row_labels)))
        self.ax.set_yticklabels(row_labels)
        cbar = self.fig.colorbar(im, ax=self.ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        if annotate:
            finite = matrix[np.isfinite(matrix)]
            mid = (np.nanmin(finite) + np.nanmax(finite)) / 2 if finite.size else 0.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    if not np.isfinite(val):
                        continue
                    color = 'white' if val < mid else 'black'
                    self.ax.text(j, i, format(val, fmt), ha='center', va='center', color=color, fontsize=8)
        return im

    def finalize(self):
        self.ax.set_xlabel(self.axis_style.xlabel)
        self.ax.set_ylabel(self.axis_style.ylabel)
        self.ax.set_title(self.axis_style.title)
        self.ax.grid(alpha=self.axis_style.grid_alpha)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc=self.axis_style.legend_loc)
        self.fig.tight_layout()
        return self.fig

    def save(self, path, dpi: int = 180):
        self.finalize()
        self.fig.savefig(path, dpi=dpi)

    def close(self):
        plt.close(self.fig)
