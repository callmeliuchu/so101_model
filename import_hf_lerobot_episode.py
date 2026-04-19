#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

DATASET_PRESETS = {
    "samuelcombey/so101_data": {
        "info_url": "https://huggingface.co/datasets/samuelcombey/so101_data/resolve/main/meta/info.json",
        "episode_url": "https://huggingface.co/datasets/samuelcombey/so101_data/resolve/main/data/chunk-000/episode_{episode_index:06d}.parquet",
    },
    "BobChang/lerobot-so101": {
        "info_url": "https://huggingface.co/datasets/BobChang/lerobot-so101/resolve/main/meta/info.json",
        "episode_url": "https://huggingface.co/datasets/BobChang/lerobot-so101/resolve/main/data/chunk-000/file-000.parquet",
    },
}
ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download one public SO101 LeRobot episode from Hugging Face and convert it to a local playback sequence."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PRESETS),
        default="samuelcombey/so101_data",
        help="Public Hugging Face dataset to pull from.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to extract. Ignored for BobChang/lerobot-so101 because its single parquet contains all episodes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults under ./external_data.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Keep one frame every N dataset frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Maximum number of sequence entries to export after filtering.",
    )
    parser.add_argument(
        "--merge-duplicates",
        action="store_true",
        help="Skip consecutive duplicate actions after stride filtering.",
    )
    return parser


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        output_path.write_bytes(response.read())


def default_output_path(dataset: str, episode_index: int) -> Path:
    dataset_slug = dataset.replace("/", "__")
    return Path(__file__).resolve().parent / "external_data" / f"{dataset_slug}_episode_{episode_index:06d}_sequence.json"


def load_actions_from_parquet(dataset: str, parquet_path: Path, episode_index: int) -> tuple[list[list[float]], int]:
    table = pq.read_table(parquet_path, columns=["action", "episode_index"])
    actions = table.column("action").to_pylist()
    fps = fetch_json(DATASET_PRESETS[dataset]["info_url"])["fps"]

    if dataset == "BobChang/lerobot-so101":
        episode_indices = [int(value) for value in table.column("episode_index").to_pylist()]
        selected = [action for action, idx in zip(actions, episode_indices) if idx == episode_index]
        if not selected:
            raise ValueError(f"Episode {episode_index} not found in {parquet_path.name}.")
        return selected, int(fps)

    return actions, int(fps)


def action_dict_from_row(row: list[float]) -> dict[str, float]:
    return {key: float(value) for key, value in zip(ACTION_KEYS, row)}


def sequence_from_actions(
    actions: list[list[float]],
    stride: int,
    max_frames: int,
    merge_duplicates: bool,
    steps_per_frame: int,
) -> list[dict]:
    sequence: list[dict] = []
    previous_action: dict[str, float] | None = None

    for raw_action in actions[:: max(1, stride)]:
        action = action_dict_from_row(raw_action)
        if merge_duplicates and previous_action == action:
            continue
        sequence.append({"action": action, "steps": steps_per_frame})
        previous_action = action
        if len(sequence) >= max_frames:
            break

    if not sequence:
        raise ValueError("No sequence entries produced. Check stride/max-frames arguments.")
    return sequence


def main() -> None:
    args = build_parser().parse_args()
    dataset = args.dataset
    preset = DATASET_PRESETS[dataset]
    output_path = args.output or default_output_path(dataset, args.episode_index)

    parquet_path = output_path.with_suffix(".parquet")
    episode_url = preset["episode_url"].format(episode_index=args.episode_index)
    download_file(episode_url, parquet_path)

    actions, fps = load_actions_from_parquet(dataset, parquet_path, args.episode_index)
    steps_per_frame = max(1, round((1.0 / fps) / 0.002))
    sequence = sequence_from_actions(
        actions=actions,
        stride=args.stride,
        max_frames=args.max_frames,
        merge_duplicates=args.merge_duplicates,
        steps_per_frame=steps_per_frame,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sequence, indent=2))

    print(f"dataset={dataset}")
    print(f"episode_index={args.episode_index}")
    print(f"fps={fps}")
    print(f"original_frames={len(actions)}")
    print(f"exported_frames={len(sequence)}")
    print(f"steps_per_frame={steps_per_frame}")
    print(f"parquet={parquet_path}")
    print(f"sequence={output_path}")


if __name__ == "__main__":
    main()
