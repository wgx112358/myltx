from __future__ import annotations

from bisect import bisect_left, bisect_right


def build_video_block_ranges(
    num_frames: int,
    *,
    block_size: int,
    independent_first_frame: bool,
) -> list[tuple[int, int]]:
    if num_frames <= 0:
        raise ValueError(f"Expected num_frames > 0, got {num_frames}")
    if block_size <= 0:
        raise ValueError(f"Expected block_size > 0, got {block_size}")

    block_ranges: list[tuple[int, int]] = []
    next_start = 0
    if independent_first_frame:
        block_ranges.append((0, 0))
        next_start = 1

    while next_start < num_frames:
        end = min(next_start + block_size - 1, num_frames - 1)
        block_ranges.append((next_start, end))
        next_start = end + 1

    return block_ranges


def build_audio_block_ends(
    *,
    block_ranges: list[tuple[int, int]],
    video_frame_end_times: list[float],
    audio_starts: list[float],
    audio_ends: list[float],
    audio_boundary_mode: str,
) -> list[int]:
    if len(audio_starts) != len(audio_ends):
        raise ValueError("audio_starts and audio_ends must have the same length")

    if audio_boundary_mode not in {"left", "center", "right"}:
        raise ValueError(f"Unsupported audio_boundary_mode: {audio_boundary_mode}")

    audio_centers = [(start + end) / 2.0 for start, end in zip(audio_starts, audio_ends, strict=True)]
    audio_seq_len = len(audio_starts)
    boundary_times = [video_frame_end_times[frame_end] for _, frame_end in block_ranges]

    block_ends: list[int] = []
    for boundary_time in boundary_times:
        if audio_boundary_mode == "left":
            block_end = bisect_left(audio_starts, boundary_time)
        elif audio_boundary_mode == "center":
            block_end = bisect_right(audio_centers, boundary_time)
        else:
            block_end = bisect_right(audio_ends, boundary_time)
        block_ends.append(min(max(block_end, 0), audio_seq_len))

    if block_ends:
        block_ends[-1] = audio_seq_len
    return block_ends


def expand_video_block_values(
    block_ranges: list[tuple[int, int]],
    block_values: list[float],
) -> list[float]:
    if len(block_ranges) != len(block_values):
        raise ValueError("block_ranges and block_values must have the same length")

    expanded: list[float] = []
    for (start, end), value in zip(block_ranges, block_values, strict=True):
        expanded.extend([value] * (end - start + 1))
    return expanded


def expand_block_values_by_ends(
    *,
    block_ends: list[int],
    block_values: list[float],
    total_len: int,
) -> list[float]:
    if len(block_ends) != len(block_values):
        raise ValueError("block_ends and block_values must have the same length")
    if total_len < 0:
        raise ValueError(f"Expected total_len >= 0, got {total_len}")

    expanded: list[float] = []
    prev_end = 0
    for block_end, value in zip(block_ends, block_values, strict=True):
        if block_end < prev_end:
            raise ValueError("block_ends must be non-decreasing")
        expanded.extend([value] * (block_end - prev_end))
        prev_end = block_end

    if prev_end < total_len and block_values:
        expanded.extend([block_values[-1]] * (total_len - prev_end))

    if len(expanded) != total_len:
        raise ValueError(f"Expanded length {len(expanded)} does not match total_len {total_len}")
    return expanded
