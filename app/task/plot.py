from random import randint
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd


def pie_chart(fractions: pd.Series, colors: list[tuple], hatches: list[str], type_names: list[str], file_path: Path):
    """ Draw a pie chart """
    plt.close()
    fig, ax = plt.subplots()
    if not fractions.any():
        plt.text(0.5, 0.5, 'No Error Type', horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    wedges, texts, autotexts = ax.pie(fractions, autopct='%1.1f%%', colors=colors,
                                      wedgeprops=dict(width=0.6), hatch=hatches)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        p.set_edgecolor('grey')
        ang = (p.theta2 - p.theta1) / 2. + p.theta1 + 1e-2  # 1e-2 is the offset for the case(100% proportion)
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontal_alignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connection_style = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connection_style})
        if fractions.iloc[i] < 0.01:
            print(f'[Ignore] Type {type_names[i]} has a proportion({fractions.iloc[i]}) less than 1%')
            continue
        ax.annotate(type_names[i], xy=(x, y), xytext=(1.05 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontal_alignment, **kw)
    ax.set_title('Error Type Distribution')
    plt.savefig(file_path)


def histogram(x: list, x_max: Union[int, float], file_path: Path):
    """ Draw a histogram """
    plt.close()
    plt.hist(x=x, bins=range(0, int(x_max) + 2), histtype='bar', color='skyblue',
             align='left', rwidth=0.5)
    if x_max >= 10:
        plt.xticks(range(0, int(x_max) + 2, 5))
    plt.xlabel('Number of Errors per Image')
    plt.ylabel('Frequency')
    plt.title('Error Frequency Distribution')
    plt.grid(True)
    if not x:
        plt.text(0.5, 0.5, 'No Error Type', horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    plt.savefig(file_path)


def line_chart(data: dict, file_path: Path):
    """ Draw a line chart """
    plt.close()
    for class_id, (r, p, r_interp, p_interp) in data.items():
        plt.plot(r, p, color='black')
        if len(p) < 11:  # For the sparse data, annotate the data points
            plt.scatter(r, p, color='black', s=5)
        plt.scatter(r_interp, p_interp, color='red', edgecolor='none', alpha=0.75, s=5)

    plt.ylim(0, 1.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(['Precision', '101-point interpolated precision'], loc='upper right')
    plt.grid(True)
    plt.savefig(file_path)


def bar_chart(data: dict, file_path: Path):
    plt.close()
    plt.bar(list(data.keys()), list(data.values()), color='grey', hatch='/')
    plt.xticks(rotation=15)
    plt.xlabel('Error Types')
    plt.ylabel('Delta-AP@0.5')
    # Add value annotation above the bar
    for x, y in enumerate(data.values()):
        plt.text(x, y, f'{y:.4f}', ha='center', va='bottom')
    plt.title('Error Type Impact on mAP')
    plt.tight_layout()
    plt.savefig(file_path)


def annotate_polygon(img, data: list[tuple[np.array, tuple[int, int, int], str]], dst_file_name: str):
    """ A general polygon patch. """
    plt.close()
    fig, ax = plt.subplots()
    ax.imshow(img)
    for points, color, name in data:
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        poly = Polygon(points, edgecolor=color, facecolor=color, alpha=0.6)
        ax.add_patch(poly)
        text_position = (randint(points[0][0], points[1][0]), randint(points[0][1], points[1][1]))
        ax.text(text_position[0], text_position[1], name, fontsize=12, color='black', weight='bold',
                path_effects=[patheffects.withStroke(linewidth=6, foreground=color, capstyle="round")])
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(dst_file_name, bbox_inches='tight', pad_inches=0)


def plot_color_legend(colors: list[tuple[int, int, int]], names: list[str], dst_file_name: str):
    """ Visualize the color legend of the error types. """
    plt.close()
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    height, width = 0.1, 0.4
    for i, (color, name) in enumerate(zip(colors, names)):
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        rect = Rectangle((0, i * height), width, height, color=color)
        ax.add_patch(rect)
        ax.text(width + 0.05, (i+0.5) * height, name, fontsize=22, color='black', weight='regular',
                verticalalignment='top')
    ax.set_ylim(0, (len(names) + 0.5) * height)
    plt.axis('off')
    plt.title('Color Legend of the Error Types', fontsize=22, fontweight='bold')
    plt.tight_layout(pad=0)
    plt.savefig(dst_file_name)
