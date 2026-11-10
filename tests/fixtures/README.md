# Test fixtures

`speech_en.m4a` — 6 s English speech clip generated with macOS text-to-speech, so it carries no
third-party licensing:

```sh
say -v Samantha --file-format=m4af --data-format=aac -o speech_en.m4a \
  "The quick brown fox jumps over the lazy dog. Speech recognition turns spoken words into written text."
```

All other media used by the tests (tones, stereo audio, video with and without audio, corrupt
files) is generated with ffmpeg at test time — see `tests/conftest.py`.
