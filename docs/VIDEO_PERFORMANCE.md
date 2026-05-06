# VIDEO PERFORMANCE NOTES

SKYWATCH can run inference on GPU and still feel visually slow if video decode/input streams are heavy.

## Why This Happens

- 4K files (for example 3840x2160) are expensive to decode with OpenCV.
- Even when display and inference loops are optimized, decode can become the bottleneck.
- The UI may keep refreshing while stream frames update slowly (stale frame effect).

## Recommended Practice

- Prefer 720p/30fps proxy videos for test and development.
- Keep inference and display decoupled (already in place).
- Use `video_sources.py` to point cameras to lighter proxy assets.

## FFmpeg Example (4K -> 720p30)

```bash
ffmpeg -i Kamera4.mp4 -vf scale=1280:-2 -r 30 -c:v libx264 -preset veryfast -crf 23 Kamera4_720p30.mp4
```

## Integration Tip

After generating proxies, update `src/video_sources.py` paths so active camera IDs use the `*_720p30` files.
