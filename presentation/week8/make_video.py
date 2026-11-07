"""
Combine the slide images with your own per-slide voice recordings into one MP4.

Put recordings in presentation/week8/recordings/ named slide01.<ext> ... slide11.<ext>
(m4a, mp3, wav, aac, ogg or opus). Each slide stays on screen for the length of its recording.
Needs ffmpeg/ffprobe on PATH.

Or record one continuous take and give the time each slide starts (mm:ss), slides 2..11:
  python presentation/week8/make_video.py --one-take take.m4a --starts 0:15,0:50,1:25,...

Usage: python presentation/week8/make_video.py [--out FILE.mp4] [--one-take FILE --starts T2,T3,...]
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIDES, RECS = HERE / "slides", HERE / "recordings"
EXTS = (".m4a", ".mp3", ".wav", ".aac", ".ogg", ".opus")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "HealthConnect_Week8_MLE_Presentation_Ramsey.mp4"))
    ap.add_argument("--one-take", help="single recording covering all slides")
    ap.add_argument("--starts", help="comma-separated start times (mm:ss) of slides 2..N in the one-take recording")
    args = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    slides = sorted(SLIDES.glob("slide*.png"))
    if not slides:
        sys.exit("no slides found; run build_week8_slides.py first")
    if args.one_take:
        return one_take(slides, Path(args.one_take), args.starts, args.out)
    pairs, missing = [], []
    for png in slides:
        audio = next((RECS / f"{png.stem}{e}" for e in EXTS if (RECS / f"{png.stem}{e}").exists()), None)
        (pairs.append((png, audio)) if audio else missing.append(png.stem))
    if missing:
        sys.exit(f"missing recordings in {RECS}: {', '.join(missing)}")

    total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = []
        for png, audio in pairs:
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                        "-of", "default=nw=1:nk=1", str(audio)],
                                       capture_output=True, text=True, check=True).stdout)
            total += dur + 0.6
            part = tmp / f"{png.stem}.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30", "-i", str(png),
                            "-i", str(audio), "-af", "apad=pad_dur=0.6", "-t", f"{dur + 0.6:.2f}",
                            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                            "-vf", "scale=1920:1080", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                            str(part)], check=True)
            parts.append(part)
            print(f"{png.stem}: {dur:5.1f}s")
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", args.out], check=True)
    m, s = divmod(round(total), 60)
    print(f"\n{args.out}\nlength {m}:{s:02d}" + ("  (brief asks for 5-10 minutes)" if not 300 <= total <= 600 else ""))


def seconds(t):
    m, _, s = t.strip().rpartition(":")
    return int(m or 0) * 60 + float(s)


def one_take(slides, audio, starts, out):
    if not audio.exists():
        sys.exit(f"recording not found: {audio}")
    if not starts:
        sys.exit("--one-take needs --starts with the start time of every slide after the first")
    cuts = [0.0] + [seconds(t) for t in starts.split(",")]
    if len(cuts) != len(slides):
        sys.exit(f"need {len(slides) - 1} start times (slides 2..{len(slides)}), got {len(cuts) - 1}")
    if cuts != sorted(cuts):
        sys.exit("start times must be in increasing order")
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nw=1:nk=1", str(audio)],
                                 capture_output=True, text=True, check=True).stdout)
    if cuts[-1] >= total:
        sys.exit(f"last start time {cuts[-1]:.0f}s is past the end of the recording ({total:.0f}s)")
    durations = [b - a for a, b in zip(cuts, cuts[1:] + [total])]
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "slides.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\nduration {d:.3f}\n" for p, d in zip(slides, durations))
                           + f"file '{slides[-1].as_posix()}'\n")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-i", str(audio), "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                        "-vf", "scale=1920:1080,fps=30", "-c:a", "aac", "-b:a", "160k", "-shortest", out], check=True)
    for p, d in zip(slides, durations):
        print(f"{p.stem}: {d:5.1f}s")
    m, s = divmod(round(total), 60)
    print(f"\n{out}\nlength {m}:{s:02d}" + ("  (brief asks for 5-10 minutes)" if not 300 <= total <= 600 else ""))


if __name__ == "__main__":
    main()
