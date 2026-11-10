"""
Learn which of your songs "go together" from your own listening sessions,
test it properly, and make playlists from it.

Usage (in your vibe-search folder):
    pip install gensim
    python3 sessions.py my_spotify_data_cleaned.zip

Optional:
    --cutoff 2025-07-01      train on sessions before this date, test on sessions after
    --seed "sara smile"      make a playlist from any song (repeat --seed for more)

What it does:
  1. Splits your history into sessions (a 30+ minute gap starts a new one).
  2. Trains on sessions before the cutoff date only.
  3. Tests on sessions after it: for each song, can each method guess the song
     you actually played next? A guess counts if it's in the method's top 10.
  4. Compares four methods:
       Popular        your 10 most played songs, every time (the floor)
       Played-next    songs most often played right next to this one before
       Embedding      AI model trained on your sessions (word2vec)
       Mix            Played-next and Embedding rankings merged
  5. Prints a few playlists from the Embedding model.

Saves results.json next to this script.
"""

import argparse
import collections
import json
import os
import random
import sys
import zipfile
from datetime import datetime, timedelta

try:
    from gensim.models import Word2Vec
except ImportError:
    sys.exit("Install gensim first:  pip install gensim")

HERE = os.path.dirname(os.path.abspath(__file__))
GAP = timedelta(minutes=30)      # silence that starts a new session
MIN_LISTEN_MS = 30_000           # under 30 seconds doesn't count as a listen
MIN_SONG_COUNT = 3               # songs seen fewer times than this in training are ignored
K = 10                           # a guess counts if it's in the top 10


def load(zip_path):
    plays = []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            base = os.path.basename(n).lower()
            if base.startswith("streaming_history_audio") and base.endswith(".json"):
                plays.extend(json.loads(z.read(n)))
    if not plays:
        sys.exit("No Streaming_History_Audio files in that zip.")
    return plays


def build_sessions(plays):
    names = {}
    rows = []
    for p in plays:
        t, a = p.get("master_metadata_track_name"), p.get("master_metadata_album_artist_name")
        if not t or not a or (p.get("ms_played") or 0) < MIN_LISTEN_MS:
            continue
        key = f"{t.strip().lower()} || {a.strip().lower()}"   # album and single versions count as one song
        names.setdefault(key, (t.strip(), a.strip()))
        rows.append((p["ts"], key))
    rows.sort()

    sessions, cur, prev = [], [], None
    for ts, key in rows:
        when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if prev and when - prev > GAP:
            sessions.append(cur)
            cur = []
        if not cur or cur[-1][1] != key:          # collapse back-to-back repeats
            cur.append((when, key))
        prev = when
    if cur:
        sessions.append(cur)
    sessions = [s for s in sessions if len(s) >= 3]
    return sessions, names


def label(names, key):
    t, a = names[key]
    return f"{t} – {a}"


def rrf(*rankings, k=60):
    # Reciprocal rank fusion: a standard way to merge ranked lists
    score = collections.Counter()
    for r in rankings:
        for i, item in enumerate(r):
            score[item] += 1 / (k + i + 1)
    return [x for x, _ in score.most_common()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--cutoff", default="2025-07-01")
    ap.add_argument("--seed", action="append", default=[])
    args = ap.parse_args()
    random.seed(42)

    plays = load(os.path.expanduser(args.zip_path))
    sessions, names = build_sessions(plays)
    cutoff = datetime.fromisoformat(args.cutoff + "T00:00:00+00:00")
    train = [[k for _, k in s] for s in sessions if s[0][0] < cutoff]
    test = [[k for _, k in s] for s in sessions if s[0][0] >= cutoff]
    if not train or not test:
        sys.exit("No sessions on one side of the cutoff. Try a different --cutoff date.")
    print(f"{len(sessions):,} sessions of 3+ songs. Training on {len(train):,} before {args.cutoff}, testing on {len(test):,} after.")

    # --- train ---
    counts = collections.Counter(k for s in train for k in s)
    vocab = {k for k, c in counts.items() if c >= MIN_SONG_COUNT}
    popular = [k for k, _ in counts.most_common(K + 1)]

    neighbours = collections.defaultdict(collections.Counter)
    for s in train:
        for i, k in enumerate(s):
            for j in range(max(0, i - 2), min(len(s), i + 3)):   # songs within 2 places either side
                if j != i:
                    neighbours[k][s[j]] += 1 if abs(i - j) == 2 else 2

    print("Training the embedding model...")
    model = Word2Vec(
        [[k for k in s if k in vocab] for s in train],
        vector_size=64, window=5, min_count=MIN_SONG_COUNT, sg=1,
        negative=10, epochs=30, seed=42, workers=1,
    )

    def guess_popular(cur):
        return [k for k in popular if k != cur][:K]

    def guess_next(cur):
        ranked = [k for k, _ in neighbours[cur].most_common(K)]
        return (ranked + [k for k in popular if k != cur and k not in ranked])[:K]

    def guess_embed(cur):
        return [k for k, _ in model.wv.most_similar(cur, topn=K)]

    def guess_mix(cur):
        a = [k for k, _ in neighbours[cur].most_common(30)]
        b = [k for k, _ in model.wv.most_similar(cur, topn=30)]
        return rrf(a, b)[:K]

    methods = {"Popular": guess_popular, "Played-next": guess_next, "Embedding": guess_embed, "Mix": guess_mix}

    # --- test: guess the next song ---
    pairs, skipped = [], 0
    for s in test:
        for cur, nxt in zip(s, s[1:]):
            if cur in model.wv and nxt in model.wv:
                pairs.append((cur, nxt))
            else:
                skipped += 1
    print(f"Test: {len(pairs):,} song-to-next-song moments ({skipped:,} skipped because a song was new after {args.cutoff}).\n")

    def bucket(k):
        c = counts[k]
        return "rare (3–9 plays)" if c < 10 else "medium (10–49)" if c < 50 else "favourite (50+)"

    hits = {m: 0 for m in methods}
    by_bucket = {m: collections.Counter() for m in methods}
    bucket_n = collections.Counter()
    for cur, nxt in pairs:
        b = bucket(cur)
        bucket_n[b] += 1
        for m, f in methods.items():
            if nxt in f(cur):
                hits[m] += 1
                by_bucket[m][b] += 1

    print(f"How often the real next song was in each method's top {K}:")
    for m in methods:
        print(f"  {m:<12} {hits[m] / len(pairs):6.1%}")
    order = ["rare (3–9 plays)", "medium (10–49)", "favourite (50+)"]
    print("\nSplit by how often the current song was played before the cutoff:")
    print("  " + " " * 12 + "".join(f"{b:>20}" for b in order))
    for m in methods:
        print(f"  {m:<12}" + "".join(f"{by_bucket[m][b] / max(1, bucket_n[b]):>20.1%}" for b in order))
    print("  " + "moments".ljust(12) + "".join(f"{bucket_n[b]:>20,}" for b in order))

    # --- playlists ---
    def playlist(seed_key, n=15, per_artist=2):
        out, per = [], collections.Counter()
        for k, _ in model.wv.most_similar(seed_key, topn=200):
            artist = names[k][1]
            if per[artist] < per_artist:
                out.append(k)
                per[artist] += 1
            if len(out) == n:
                break
        return out

    def find(text):
        text = text.lower()
        hits_ = [k for k in model.wv.index_to_key if text in k]
        return max(hits_, key=lambda k: counts[k]) if hits_ else None

    seeds = [find(s) for s in args.seed] if args.seed else []
    if args.seed and not all(seeds):
        missing = [s for s, k in zip(args.seed, seeds) if not k]
        print(f"\nCouldn't find: {', '.join(missing)}")
    seeds = [k for k in seeds if k]
    if not seeds:
        top = [k for k, _ in counts.most_common(200) if k in model.wv]
        seeds = top[:2] + random.sample(top[2:], 3)

    lists = {}
    for sk in seeds:
        pl = playlist(sk)
        lists[label(names, sk)] = [label(names, k) for k in pl]
        print(f"\nPlaylist from {label(names, sk)}:")
        for i, k in enumerate(pl, 1):
            print(f"  {i:2}. {label(names, k)}")

    json.dump({
        "cutoff": args.cutoff, "train_sessions": len(train), "test_sessions": len(test),
        "test_moments": len(pairs), "top_k": K,
        "hit_rate": {m: hits[m] / len(pairs) for m in methods},
        "hit_rate_by_bucket": {m: {b: by_bucket[m][b] / max(1, bucket_n[b]) for b in order} for m in methods},
        "playlists": lists,
    }, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
    print("\nSaved results.json")


if __name__ == "__main__":
    main()
