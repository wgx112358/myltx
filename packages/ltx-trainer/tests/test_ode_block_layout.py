from __future__ import annotations

import pytest

from ltx_trainer.ode_block_layout import (
    build_audio_block_ends,
    build_video_block_ranges,
    expand_block_values_by_ends,
    expand_video_block_values,
)


def test_build_video_block_ranges_with_independent_first_frame() -> None:
    assert build_video_block_ranges(7, block_size=3, independent_first_frame=True) == [
        (0, 0),
        (1, 3),
        (4, 6),
    ]


def test_build_video_block_ranges_without_independent_first_frame() -> None:
    assert build_video_block_ranges(7, block_size=3, independent_first_frame=False) == [
        (0, 2),
        (3, 5),
        (6, 6),
    ]


@pytest.mark.parametrize(
    ("audio_boundary_mode", "expected"),
    [
        ("left", [2, 4, 6]),
        ("center", [1, 4, 6]),
        ("right", [1, 3, 6]),
    ],
)
def test_build_audio_block_ends(audio_boundary_mode: str, expected: list[int]) -> None:
    block_ranges = [(0, 0), (1, 2), (3, 4)]
    video_frame_end_times = [0.5, 1.0, 1.5, 2.0, 2.5]
    audio_starts = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    audio_ends = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]

    assert build_audio_block_ends(
        block_ranges=block_ranges,
        video_frame_end_times=video_frame_end_times,
        audio_starts=audio_starts,
        audio_ends=audio_ends,
        audio_boundary_mode=audio_boundary_mode,
    ) == expected


def test_expand_video_block_values() -> None:
    block_ranges = [(0, 0), (1, 2), (3, 4)]
    assert expand_video_block_values(block_ranges, [10, 20, 30]) == [10, 20, 20, 30, 30]


def test_expand_block_values_by_ends() -> None:
    assert expand_block_values_by_ends(block_ends=[2, 4, 6], block_values=[10, 20, 30], total_len=6) == [
        10,
        10,
        20,
        20,
        30,
        30,
    ]
