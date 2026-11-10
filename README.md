# Spotify playlist model

I trained a small AI model on my own Spotify listening history to learn which songs go together, then used it to make playlists. I also tested it fairly against simpler methods.

## How it works

1. **Sessions.** I split about 255,000 plays (2015–2026) into listening sessions. A gap of 30+ minutes starts a new session, plays under 30 seconds count as skips, and only sessions with 3+ different songs are kept. That leaves about 7,900 sessions.
2. **Model.** I treat each session like a sentence and each song like a word, then train word2vec (gensim; 64 dimensions, window 5, skip-gram, negative sampling). Songs played in the same sessions end up close together, with no genre labels given.
3. **Playlists.** I pick a seed song and take its 15 nearest neighbours by cosine similarity, with at most 2 songs per artist.

## Testing

- **Time split:** every method learns from sessions before 1 July 2025 and is tested on sessions after that date, so there's no leakage.
- **Task:** for each song in a test session, is the real next song in the method's top 10? (hit rate @ 10)
- **Baselines:** most-played songs, and a "played-next" count from my history.

| Method | Hit rate @ 10 |
|---|---|
| Most played | 1.5% |
| Played-next (counting) | **7.0%** |
| Embedding (word2vec) | 4.1% |
| Mix (reciprocal rank fusion) | 6.3% |

The simple counting method wins at guessing the exact next song, because it memorises sequences. The embeddings capture songs with the same *feel*, which is what a playlist needs (a 2Pac seed gives Biggie, Nas, Gang Starr and A Tribe Called Quest). On rarely played songs, the mix of both methods does best (11.0%). The main lesson is that the metric you choose decides which method looks best.

## Limitations

- It's one listener's data.
- About two-thirds of test moments involved songs first played after the cutoff (cold start), so they couldn't be scored.
- The model's settings haven't been tuned yet.

## Run it

```
pip install gensim numpy
python sessions.py my_spotify_data.zip
```

Use your own Spotify "Extended streaming history" export. My data isn't included in this repo.
