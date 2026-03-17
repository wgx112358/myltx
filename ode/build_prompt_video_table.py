import argparse
import csv
import html
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a structured table mapping selected videos to prompts.")
    parser.add_argument(
        "--selected-latents",
        type=Path,
        default=Path("ode/distilled_example_videos/selected_latents.txt"),
        help="Text file containing selected latent filenames, one per line.",
    )
    parser.add_argument(
        "--prompts-csv",
        type=Path,
        default=Path("datagen/ltx_prompts_12000.csv"),
        help="CSV file containing prompts.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("ode/distilled_example_videos"),
        help="Directory containing decoded videos.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ode/distilled_example_videos/prompt_video_table.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("ode/distilled_example_videos/prompt_video_table.md"),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("ode/distilled_example_videos/index.html"),
        help="Output shareable HTML preview path.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_selected_latents(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_rows(selected_latents: list[str], prompts: list[dict[str, str]], video_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for latent_name in selected_latents:
        sample_id = int(Path(latent_name).stem)
        if sample_id >= len(prompts):
            raise IndexError(f"Sample id {sample_id} is out of range for prompt CSV")
        prompt = prompts[sample_id]["text_prompt"]
        video_name = f"{sample_id:05d}.mp4"
        rows.append(
            {
                "sample_id": f"{sample_id:05d}",
                "latent_file": latent_name,
                "video_file": video_name,
                "video_rel_path": f"./{video_name}",
                "prompt": prompt,
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "latent_file", "video_file", "video_rel_path", "prompt"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Distilled Example Videos And Prompts",
        "",
        f"Total samples: {len(rows)}",
        "",
        "| sample_id | video_file | prompt |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        prompt = row["prompt"].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {row['sample_id']} | {row['video_file']} | {prompt} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        cards.append(
            f"""
            <section class="card">
              <div class="meta">
                <div class="id">Sample {html.escape(row["sample_id"])}</div>
                <div class="file">{html.escape(row["video_file"])}</div>
              </div>
              <video controls preload="metadata" src="{html.escape(row["video_rel_path"])}"></video>
              <p class="prompt">{html.escape(row["prompt"])}</p>
            </section>
            """.strip()
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Distilled Example Videos</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #5b6472;
      --line: #d8cfbf;
      --accent: #8b5e3c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8ea 0%, transparent 30%),
        linear-gradient(180deg, #efe8db 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 40px;
      line-height: 1.1;
    }}
    .intro {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: color-mix(in srgb, var(--panel) 92%, white 8%);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(69, 50, 28, 0.08);
    }}
    .meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      font-size: 14px;
      color: var(--muted);
    }}
    .id {{
      color: var(--accent);
      font-weight: 700;
    }}
    video {{
      width: 100%;
      border-radius: 12px;
      background: #000;
      margin-bottom: 12px;
    }}
    .prompt {{
      margin: 0;
      font-size: 15px;
      line-height: 1.6;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Distilled Example Videos</h1>
    <p class="intro">Share this folder as-is. The recipient can open <code>index.html</code> locally and watch the videos with their prompts.</p>
    <div class="grid">
      {' '.join(cards)}
    </div>
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_csv.resolve())
    selected_latents = load_selected_latents(args.selected_latents.resolve())
    rows = build_rows(selected_latents, prompts, args.video_dir.resolve())
    write_csv(rows, args.output_csv.resolve())
    write_markdown(rows, args.output_md.resolve())
    write_html(rows, args.output_html.resolve())
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote {len(rows)} rows to {args.output_md}")
    print(f"Wrote {len(rows)} rows to {args.output_html}")


if __name__ == "__main__":
    main()
