#!/usr/bin/env python3
"""Sangeet mashup builder.

Two steps, so section choices can be reviewed before anything is rendered:

    python3 mashup.py analyze [SONG_DIR]     # match files, pick sections, write plan.json
    python3 mashup.py render  [SONG_DIR]     # build the MP3 + rehearsal MP4 from plan.json

To override a section, edit plan.json ("start" in seconds) or run:
    python3 mashup.py analyze --set 4=72.5 --set 7=1:58
"""
import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "plan.json")
OUT_MP3 = "sangeetha_sangeet_mashup.mp3"
OUT_MP4 = "sangeetha_sangeet_mashup.mp4"
AUDIO_EXT = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".webm", ".wma", ".mp4"}

XFADE = 2.5          # seconds of crossfade between songs
FADE_IN = 3.0        # fade in at the very start
FADE_OUT = 3.0       # fade out at the very end
TARGET_LUFS = -14.0  # loudness every segment is normalised to
TRUE_PEAK = -1.5
SR = 22050           # analysis sample rate
HOP = 0.1            # analysis frame hop (s)

# (title, duration, mode, match keywords, note)
SONGS = [
    ("Maye Ni Maye",           45, "calm",  ["maye ni maye", "maye", "mayeni"], "slow emotional part"),
    ("Le Gayi Le Gayi",        45, "peak",  ["le gayi", "legayi", "gayi"], "most upbeat playful section"),
    ("Roop Suhana Lagta Hai",  45, "chorus", ["roop suhana", "suhana", "roop"], "chorus"),
    ("Mehndi Laga Ke Rakhna",  45, "chorus", ["mehndi laga", "mehendi", "mehndi", "doli saja"], "'doli saja ke rakhna' chorus"),
    ("Ainvayi Ainvayi",        45, "peak",  ["ainvayi", "ainvayi ainvayi", "band baaja"], "high energy chorus"),
    ("London Thumakda",        45, "peak",  ["london thumakda", "thumakda", "london"], "chorus with dhol beats"),
    ("Gallan Goodiyaan",       60, "chorus", ["gallan goodiyaan", "goodiyaan", "gallan", "gooddiyan"], "chorus, 3 s fade out"),
]


# ---------------------------------------------------------------- helpers
def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def ensure_ffmpeg():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    print("ffmpeg not found - installing...")
    if shutil.which("apt-get"):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
    elif shutil.which("brew"):
        subprocess.run(["brew", "install", "ffmpeg"], check=True)
    else:
        sys.exit("Please install ffmpeg manually.")


def fmt(t):
    m, s = divmod(max(t, 0), 60)
    return f"{int(m)}:{s:05.2f}"


def parse_time(s):
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path]).stdout
    return float(out.strip())


def norm_name(s):
    s = os.path.splitext(os.path.basename(s))[0].lower()
    s = re.sub(r"[_\-.()\[\]]+", " ", s)
    s = re.sub(r"\b(official|video|audio|full|song|lyrical|lyrics|hd|4k|mp3|320kbps|kbps|feat|ft)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- matching
def match_files(song_dir):
    files = sorted(f for f in os.listdir(song_dir)
                   if os.path.splitext(f)[1].lower() in AUDIO_EXT
                   and not f.startswith("sangeetha_sangeet_mashup"))
    scores = {}
    for i, (_, _, _, keys, _) in enumerate(SONGS):
        for f in files:
            n = norm_name(f)
            best = 0.0
            for k in keys:
                if k in n or k.replace(" ", "") in n.replace(" ", ""):
                    best = max(best, 0.6 + 0.4 * len(k) / max(len(n), 1))
                best = max(best, difflib.SequenceMatcher(None, k, n).ratio() * 0.8)
            scores[(i, f)] = best

    mapping, problems = {}, []
    used = set()
    # greedy by best score so the strongest matches are fixed first
    for (i, f), sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if i in mapping or f in used or sc < 0.6:
            continue
        mapping[i], _ = f, used.add(f)
    for i, (title, *_rest) in enumerate(SONGS):
        if i not in mapping:
            cands = sorted(((scores[(i, f)], f) for f in files if f not in used), reverse=True)[:3]
            problems.append(f"  - {title}: no confident match"
                            + (f" (closest: {', '.join(c[1] for c in cands)})" if cands else ""))
        else:
            # ambiguous if another file scores almost as well for this song
            rivals = [f for f in files if f != mapping[i] and scores[(i, f)] >= scores[(i, mapping[i])] - 0.05
                      and scores[(i, f)] >= 0.6]
            if rivals:
                problems.append(f"  - {title}: '{mapping[i]}' vs {rivals} - which one?")
    leftovers = [f for f in files if f not in used]
    return files, mapping, problems, leftovers


# ---------------------------------------------------------------- analysis
def load_mono(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def features(y):
    hop = int(SR * HOP)
    win = 2048
    n = max(1 + (len(y) - win) // hop, 1)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = np.pad(y, (0, max(0, idx.max() + 1 - len(y))))[idx] * np.hanning(win)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(win, 1 / SR)

    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    loud = 20 * np.log10(rms + 1e-9)                       # dB per frame
    logspec = np.log1p(spec)
    flux = np.maximum(np.diff(logspec, axis=0, prepend=logspec[:1]), 0).sum(axis=1)
    low = (freqs > 40) & (freqs < 200)                     # dhol / kick band
    low_flux = np.maximum(np.diff(logspec[:, low], axis=0, prepend=logspec[:1, low]), 0).sum(axis=1)

    # chroma for repetition (chorus) detection
    band = (freqs > 80) & (freqs < 5000)
    pitch = np.round(12 * np.log2(freqs[band] / 440.0)).astype(int) % 12
    chroma = np.zeros((n, 12))
    for c in range(12):
        chroma[:, c] = spec[:, band][:, pitch == c].sum(axis=1)
    chroma /= np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9
    return loud, flux, low_flux, chroma


def tempo_strength(onset):
    """How strong/regular the beat is in this slice (autocorrelation peak 60-180 BPM)."""
    o = onset - onset.mean()
    if o.std() < 1e-9:
        return 0.0
    ac = np.correlate(o, o, "full")[len(o) - 1:]
    ac /= ac[0]
    lo, hi = int(60 / 180 / HOP), int(60 / 60 / HOP)
    return float(ac[lo:hi].max()) if hi < len(ac) else 0.0


def z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def analyze_song(path, dur, mode, total):
    y = load_mono(path)
    loud, flux, low_flux, chroma = features(y)
    nfr = len(loud)
    W = int((dur + XFADE) / HOP)                          # segment incl. crossfade tail
    margin = int(8 / HOP)                                 # avoid intros / outros
    step = int(0.5 / HOP)
    starts = np.arange(margin, max(margin + 1, nfr - W - margin), step)
    if len(starts) == 0 or nfr < W:
        starts = np.array([0])

    # repetition score: how much each 4 s block recurs elsewhere in the song (chorus cue)
    blk = int(4 / HOP)
    nb = nfr // blk
    bc = np.array([chroma[i * blk:(i + 1) * blk].mean(0) for i in range(nb)]) if nb else np.zeros((1, 12))
    bc /= np.linalg.norm(bc, axis=1, keepdims=True) + 1e-9
    sim = bc @ bc.T
    for d in range(-1, 2):                                 # ignore self / neighbours
        np.fill_diagonal(sim[max(0, d):, max(0, -d):], 0)
    rep_blk = np.sort(sim, axis=1)[:, -3:].mean(1) if nb > 3 else np.zeros(max(nb, 1))
    rep = np.repeat(rep_blk, blk)[:nfr]
    rep = np.pad(rep, (0, nfr - len(rep)), mode="edge")

    rows = []
    for s in starts:
        e = min(s + W, nfr)
        rows.append((np.mean(loud[s:e]), np.mean(flux[s:e]), np.mean(low_flux[s:e]),
                     np.mean(rep[s:e]), tempo_strength(flux[s:e])))
    L, F, B, R, T = map(np.array, zip(*rows))

    if mode == "calm":        # emotional: quieter beat, sustained voice, still clearly audible
        score = -1.0 * z(F) - 0.5 * z(B) + 0.6 * z(L) * (L > np.median(L)) + 0.3 * z(R)
    elif mode == "peak":      # most energetic: loud, dense onsets, strong dhol, steady groove
        score = 1.0 * z(L) + 1.0 * z(F) + 0.8 * z(B) + 0.5 * z(T) + 0.6 * z(R)
    else:                     # chorus: the repeated, loud hook
        score = 0.8 * z(L) + 0.6 * z(F) + 0.4 * z(B) + 1.2 * z(R)
    best = int(np.argmax(score))
    s = starts[best]

    # snap start to the nearest strong onset (downbeat-ish) within +/-1 s so cuts land on a beat
    lo, hi = max(0, s - 10), min(nfr - W, s + 10)
    if hi > lo:
        s = lo + int(np.argmax(flux[lo:hi + 1]))
    start = round(s * HOP, 2)
    return {
        "start": start,
        "loudness_db": round(float(L[best]), 1),
        "song_avg_db": round(float(np.mean(loud)), 1),
        "beat_energy": round(float(F[best] / (np.mean(flux) + 1e-9)), 2),   # 1.0 = song average
        "dhol_energy": round(float(B[best] / (np.mean(low_flux) + 1e-9)), 2),
        "repetition": round(float(R[best]), 2),
    }


def cmd_analyze(a):
    ensure_ffmpeg()
    song_dir = os.path.abspath(a.dir)
    files, mapping, problems, leftovers = match_files(song_dir)
    print(f"Found {len(files)} audio file(s) in {song_dir}")
    if problems or len(files) < len(SONGS):
        print("\nI couldn't match every song confidently:")
        print("\n".join(problems) if problems else "  - fewer than 7 audio files")
        if leftovers:
            print(f"  Unmatched files: {leftovers}")
        print("\nRename the files or pass --map N=filename (e.g. --map 3='track03.mp3').")
    for m in a.map or []:
        n, f = m.split("=", 1)
        mapping[int(n) - 1] = f
    if len(mapping) < len(SONGS):
        sys.exit(2)
    overrides = {int(k) - 1: parse_time(v) for k, v in (s.split("=", 1) for s in (a.set or []))}

    plan = []
    print(f"\n{'#':<2} {'Song':<24} {'File':<34} {'Start':>8} {'End':>8}  Why")
    for i, (title, dur, mode, _, note) in enumerate(SONGS):
        path = os.path.join(song_dir, mapping[i])
        total = duration(path)
        info = analyze_song(path, dur, mode, total)
        if i in overrides:
            info["start"], info["override"] = overrides[i], True
        tail = XFADE if i < len(SONGS) - 1 else 0
        info["start"] = min(info["start"], max(0, total - dur - tail))
        end = info["start"] + dur
        plan.append({"n": i + 1, "title": title, "file": mapping[i], "duration": dur,
                     "song_length": round(total, 2), "end": round(end, 2), **info})
        why = ("manual override" if info.get("override") else
               f"{note}; level {info['loudness_db']} dB (song avg {info['song_avg_db']}), "
               f"beat x{info['beat_energy']}, dhol x{info['dhol_energy']}, repeat {info['repetition']}")
        print(f"{i+1:<2} {title:<24} {mapping[i][:34]:<34} {fmt(info['start']):>8} {fmt(end):>8}  {why}")
    with open(PLAN, "w") as fh:
        json.dump({"song_dir": song_dir, "segments": plan}, fh, indent=2)
    print(f"\n(Crossfade tail of {XFADE}s is taken after each 'End' except the last.)")
    print(f"Plan saved to {PLAN}. Review it, then run:  python3 mashup.py render")


# ---------------------------------------------------------------- render
def loudnorm_segment(src, start, length, dst, fade_in=0.0, fade_out=0.0):
    pre = f"atrim=0:{length},asetpts=PTS-STARTPTS"
    target = f"I={TARGET_LUFS}:TP={TRUE_PEAK}:LRA=11"
    meas = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(length),
                           "-i", src, "-af", f"{pre},loudnorm={target}:print_format=json",
                           "-f", "null", "-"], capture_output=True, text=True, check=True).stderr
    m = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
    af = (f"{pre},loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
          f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
          f":offset={m['target_offset']}:linear=true,aresample=44100")
    if fade_in:
        af += f",afade=t=in:st=0:d={fade_in}"
    if fade_out:
        af += f",afade=t=out:st={length - fade_out}:d={fade_out}"
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(length), "-i", src,
         "-af", af, "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", dst])
    return float(m["input_i"])


def cmd_render(a):
    ensure_ffmpeg()
    with open(PLAN) as fh:
        plan = json.load(fh)
    song_dir = os.path.abspath(a.dir or plan["song_dir"])
    segs = plan["segments"]
    out_dir = os.path.abspath(a.out or song_dir)
    tmp = tempfile.mkdtemp(prefix="mashup_")

    print("Normalising segments to", TARGET_LUFS, "LUFS ...")
    wavs = []
    for i, s in enumerate(segs):
        last = i == len(segs) - 1
        length = s["duration"] + (0 if last else XFADE)
        dst = os.path.join(tmp, f"seg{i+1}.wav")
        before = loudnorm_segment(os.path.join(song_dir, s["file"]), s["start"], length, dst,
                                  fade_in=FADE_IN if i == 0 else 0, fade_out=FADE_OUT if last else 0)
        print(f"  {s['n']}. {s['title']:<24} {before:6.1f} LUFS -> {TARGET_LUFS}")
        wavs.append(dst)

    # crossfade chain: [0][1]acrossfade[a1]; [a1][2]acrossfade[a2] ...
    inputs = sum((["-i", w] for w in wavs), [])
    chain, prev = [], "0:a"
    for i in range(1, len(wavs)):
        lab = f"x{i}"
        chain.append(f"[{prev}][{i}:a]acrossfade=d={XFADE}:c1=tri:c2=tri[{lab}]")
        prev = lab
    master = os.path.join(tmp, "master.wav")
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(chain),
         "-map", f"[{prev}]", "-c:a", "pcm_s16le", master])

    mp3 = os.path.join(out_dir, OUT_MP3)
    run(["ffmpeg", "-y", "-v", "error", "-i", master, "-c:a", "libmp3lame", "-b:a", "320k",
         "-id3v2_version", "3", "-metadata", "title=Sangeetha Sangeet Mashup", mp3])

    # cue times: each song starts where the previous one's crossfade begins
    cues, t = [], 0.0
    for s in segs:
        cues.append(t)
        t += s["duration"]
    total = duration(mp3)

    # rehearsal video: black background, current song name, next song, clock
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_r = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    vf = []
    for i, s in enumerate(segs):
        # switch the name at the middle of each crossfade
        a0 = 0 if i == 0 else cues[i] + XFADE / 2
        a1 = total if i == len(segs) - 1 else cues[i + 1] + XFADE / 2
        name_f = os.path.join(tmp, f"name{i}.txt")
        with open(name_f, "w") as fh:
            fh.write(s["title"])
        nxt_f = os.path.join(tmp, f"next{i}.txt")
        with open(nxt_f, "w") as fh:
            fh.write(f"Next: {segs[i+1]['title']}" if i + 1 < len(segs) else "Final song")
        cnt_f = os.path.join(tmp, f"count{i}.txt")
        with open(cnt_f, "w") as fh:
            fh.write(f"Song {i+1} of {len(segs)}")
        en = f"enable='between(t,{a0:.2f},{a1:.2f})'"
        vf += [
            f"drawtext=fontfile={font}:textfile={name_f}:fontcolor=white:fontsize=84:"
            f"x=(w-text_w)/2:y=(h-text_h)/2-30:{en}",
            f"drawtext=fontfile={font_r}:textfile={cnt_f}:fontcolor=0xBBBBBB:fontsize=34:"
            f"x=(w-text_w)/2:y=(h/2)-150:{en}",
            f"drawtext=fontfile={font_r}:textfile={nxt_f}:fontcolor=0x888888:fontsize=36:"
            f"x=(w-text_w)/2:y=h-120:{en}",
        ]
    vf.append(f"drawtext=fontfile={font_r}:text='%{{pts\\:gmtime\\:0\\:%M\\\\\\:%S}}':"
              f"fontcolor=0x888888:fontsize=36:x=w-text_w-40:y=40")
    mp4 = os.path.join(out_dir, OUT_MP4)
    run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s=1280x720:r=10:d={total}",
         "-i", master, "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", mp4])

    lines = ["SANGEETHA SANGEET MASHUP - CUE SHEET", f"Total length: {fmt(total)}", "",
             f"{'#':<3}{'Mashup time':<13}{'Song':<26}{'Source section':<20}"]
    for i, s in enumerate(segs):
        lines.append(f"{i+1:<3}{fmt(cues[i]):<13}{s['title']:<26}{fmt(s['start'])} - {fmt(s['end'])}")
    lines += ["", f"Crossfades: {XFADE}s (each song starts as the previous one begins fading out).",
              f"Fade in {FADE_IN}s at start, fade out {FADE_OUT}s at end. All segments at {TARGET_LUFS} LUFS."]
    sheet = "\n".join(lines)
    with open(os.path.join(out_dir, "cue_sheet.txt"), "w") as fh:
        fh.write(sheet + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nWrote {mp3}\nWrote {mp4}\n\n{sheet}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("dir", nargs="?", default=".")
    an.add_argument("--set", action="append", help="override start, e.g. 4=1:12.5")
    an.add_argument("--map", action="append", help="force file for song N, e.g. 3=track03.mp3")
    rd = sub.add_parser("render")
    rd.add_argument("dir", nargs="?")
    rd.add_argument("--out", help="output folder (default: song folder)")
    a = p.parse_args()
    cmd_analyze(a) if a.cmd == "analyze" else cmd_render(a)


if __name__ == "__main__":
    main()
