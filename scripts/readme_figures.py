#!/usr/bin/env python3
"""Render README figures from bundled goals and route maps, without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vnav_benchmark import mapview, topo  # noqa: E402
from vnav_benchmark.paths import data_root  # noqa: E402
from vnav_benchmark.report import style  # noqa: E402

SCENES = (
    ("env0_test_v3", "ENV0 / Single-floor office"),
    ("env1_test_v1", "ENV1 / Garden and outdoor buildings"),
    ("env3_test_v1", "ENV3 / Two-floor office"),
)


def goal_sheet(output: Path) -> List[Path]:
    """Compose all frozen goal photographs in route order, without cropping.

    Args:
        output: Destination PNG path.

    Returns:
        The source episode files and photographs used in the composition.
    """
    width, margin, gap = 1680, 30, 15
    tile_width = (width - 2 * margin - 4 * gap) // 5
    tile_height = tile_width * 3 // 4
    row_height = tile_height + 95
    canvas = Image.new("RGB", (width, 80 + 3 * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 16),
        "15 routes. Three environments.",
        font=style.pil_font(34, bold=True),
        fill="#202020",
    )
    inputs = []
    for row, (episode_set, title) in enumerate(SCENES):
        root = REPO / "configs/episodes" / episode_set
        episode_path = root / "episodes.json"
        inputs.append(episode_path)
        episodes = json.loads(episode_path.read_text(encoding="utf-8"))["episodes"]
        episodes.sort(key=lambda episode: int(episode["label"][1:]))
        top = 80 + row * row_height
        draw.text(
            (margin, top), title, font=style.pil_font(26, bold=True), fill=style.color_hex(row)
        )
        for column, episode in enumerate(episodes):
            source = root / episode["goal_rgb"]
            inputs.append(source)
            left = margin + column * (tile_width + gap)
            with Image.open(source) as image:
                image = image.convert("RGB").resize((tile_width, tile_height), Image.LANCZOS)
                canvas.paste(image, (left, top + 39))
            label = "{} / {}".format(episode["label"], episode["goal_room"].replace("_", " "))
            draw.text(
                (left, top + 44 + tile_height), label, font=style.pil_font(21), fill="#202020"
            )
    canvas.save(output)
    return inputs


def map_figure(output: Path) -> List[Path]:
    """Plot R5 traversal graphs and four real dense-map keyframe photographs.

    Args:
        output: Destination PNG path.

    Returns:
        The registry, map files and evaluator geometry used for the figure.
    """
    config = mapview.load_maps_config()
    episode_path = REPO / "configs/episodes/env0_test_v3/episodes.json"
    episodes = json.loads(episode_path.read_text(encoding="utf-8"))["episodes"]
    route = next(episode["route"] for episode in episodes if episode["label"] == "R5")
    entries = {entry.variant: entry for entry in config.maps.values() if entry.route == route}
    scene_dir = REPO / "data/scenes/env0_office_single_floor"
    scene_path, truth_path = scene_dir / "scene.json", scene_dir / "truth.npz"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    floor = scene["floors"]["f1"]
    with np.load(truth_path) as truth:
        # Evaluator grids index [x, y]; image arrays index [row=y, column=x].
        blocked = truth["f1_raw_blocked"].T.copy()
    extent = (
        floor["x0"],
        floor["x0"] + blocked.shape[1] * scene["cell_m"],
        floor["y0"],
        floor["y0"] + blocked.shape[0] * scene["cell_m"],
    )
    figure = plt.figure(figsize=(13, 8), facecolor="white")
    grid = figure.add_gridspec(2, 4, height_ratios=(2.1, 1), hspace=0.35, wspace=0.2)
    inputs = [REPO / "configs/maps.yaml", episode_path, scene_path, truth_path]
    for index, variant in enumerate(("dense", "sparse")):
        entry = entries[variant]
        graph = topo.load_topo_graph(entry.source, config.camera_forward_m, entry.edge_type)
        transform = np.asarray(entry.map_to_world)
        points = np.array(
            [(transform @ np.array([pose.x, pose.y, 0, 1]))[:2] for pose in graph.poses.values()]
        )
        edges = {(min(a, b), max(a, b)) for a in graph.poses for b, _ in graph.neighbors(a)}
        axis = figure.add_subplot(grid[0, index * 2 : index * 2 + 2])
        axis.imshow(blocked, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=2.5)
        color = style.color_hex(index)
        for a, b in sorted(edges):
            axis.plot(points[[a, b], 0], points[[a, b], 1], color=color, linewidth=2.5)
        axis.scatter(points[:, 0], points[:, 1], s=68, c=color, edgecolors="white", zorder=3)
        for node, (x, y) in enumerate(points):
            axis.annotate(
                str(node),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=11,
                color="#202020",
            )
        axis.set(xlim=(3.5, 14), ylim=(1, 15), xlabel="World x (m)", ylabel="World y (m)")
        axis.set_aspect("equal")
        axis.set_title(
            f"{variant.capitalize()} / {len(graph.poses)} keyframes, {len(edges)} traversal edges\n"
            f"{len(graph.components())} connected component",
            color=color,
            fontsize=16,
        )
        inputs.extend(entry.source / name for name in ("poses.txt", "edges_trav.txt"))
    dense = entries["dense"].source
    image_names = [
        line.split()[0]
        for line in (dense / "poses.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for column, node in enumerate((0, 3, 6, 10)):
        source = dense / image_names[node]
        inputs.append(source)
        axis = figure.add_subplot(grid[1, column])
        with Image.open(source) as image:
            axis.imshow(image)
        axis.set_title(f"Dense keyframe {node}", fontsize=14)
        axis.axis("off")
    figure.suptitle("R5 / Same route, different map density", fontsize=22, y=0.98)
    figure.subplots_adjust(top=0.85, bottom=0.035)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)
    return inputs


def source_name(path: Path) -> str:
    """Return a logical source key without exposing a relocated data root."""
    try:
        return "data/" + path.relative_to(data_root()).as_posix()
    except ValueError:
        return path.relative_to(REPO).as_posix()


def main() -> int:
    """Generate the figures and an input checksum record for provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs/visuals")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style.setting_font(fontsize=13, titlesize=16, legend_fontsize=12)
    font_manager.fontManager.addfont(style.font_file())
    family = font_manager.FontProperties(fname=style.font_file()).get_name()
    matplotlib.rcParams.update({"font.serif": [family], "text.usetex": False})
    inputs = goal_sheet(args.output_dir / "goals.png")
    inputs.extend(map_figure(args.output_dir / "maps.png"))
    inputs.extend((Path(__file__).resolve(), Path(style.__file__), Path(topo.__file__)))
    sources: Dict[str, str] = {
        source_name(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(inputs))
    }
    document = {"schema": "vnav-readme-figures/1", "sources_sha256": sources}
    (args.output_dir / "sources.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote goals.png, maps.png and sources.json to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
