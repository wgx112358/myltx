from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VMC_ROOT = REPO_ROOT / "visual_multi_compare"
LOG_ROOT = "/mnt/petrelfs/wanggongxuan/workspace/ode_infer_log"


def test_submit_all_spot_creates_log_root_and_task_directories() -> None:
    content = (VMC_ROOT / "submit_all_spot.sh").read_text(encoding="utf-8")

    assert 'LOG_ROOT="/mnt/petrelfs/wanggongxuan/workspace/ode_infer_log"' in content
    assert 'mkdir -p "$LOG_ROOT"' in content

    for task_dir in (
        "exact_replay",
        "official_distilled",
        "ode_pipe_distilled",
        "base_step250",
        "latest_4node_step1000",
        "run1_step1000",
    ):
        assert f'mkdir -p "$LOG_ROOT/{task_dir}"' in content


def test_each_sbatch_writes_logs_to_task_specific_directory() -> None:
    expected = {
        "exact_replay_spot.sbatch": "exact_replay",
        "official_distilled_spot.sbatch": "official_distilled",
        "ode_pipe_distilled_spot.sbatch": "ode_pipe_distilled",
        "ode_base_step250_spot.sbatch": "base_step250",
        "ode_latest_step1000_spot.sbatch": "latest_4node_step1000",
        "ode_run1_step1000_spot.sbatch": "run1_step1000",
    }

    for script_name, task_dir in expected.items():
        content = (VMC_ROOT / script_name).read_text(encoding="utf-8")
        assert f"#SBATCH -o {LOG_ROOT}/{task_dir}/{task_dir}-%j.out" in content
        assert f"#SBATCH -e {LOG_ROOT}/{task_dir}/{task_dir}-%j.err" in content
