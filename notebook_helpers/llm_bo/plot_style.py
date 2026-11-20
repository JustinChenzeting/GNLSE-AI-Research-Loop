"""Shared Matplotlib settings for spectral-mask reports."""
import matplotlib.pyplot as plt


def configure():
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "pdf.fonttype": 42})


def subplots(*args, width_mm, height_mm):
    return plt.subplots(*args, figsize=(width_mm / 25.4, height_mm / 25.4), layout="constrained", squeeze=False)


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)


def panel_label(ax, letter):
    ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, va="bottom")
