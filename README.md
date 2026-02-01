# Smart Timelapse Editor 🎬

Create **Devon Crawford-style montages** from hours of study/work footage. This script intelligently detects moments with actual activity (typing, scrolling, transitions) rather than just speeding up everything uniformly.

## Features

- **Smart Motion Detection**: Finds moments with real activity, not just static screens
- **Typing Pattern Recognition**: Prioritizes coding/writing activity
- **Scene Change Detection**: Captures transitions and important moments
- **Configurable Everything**: Duration, quality, speed variation, clip lengths
- **Social Media Ready**: Optimized defaults for 60-second reels/shorts

## Requirements

Create a virtual environment first

```bash
python -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# activate python virtual environment
source .venv/bin/activate

# Basic usage - analyze folder and create 75-second timelapse
python timelapse_editor.py -c config.json

# Show all options
python timelapse_editor.py --help

# Show tips for creating great timelapses
python timelapse_editor.py --tips
```

## Usage Examples

### 1. Basic Timelapse (60 seconds)

```bash
python timelapse_editor.py -i ./footage -o ./output.mp4
```

### 2. Longer YouTube Short (90 seconds, higher quality)

```bash
python timelapse_editor.py -i ./footage -o ./youtube_short.mp4 -d 90 -q 18
```

### 3. Quick TikTok/Reel (30 seconds, more selective)

```bash
python timelapse_editor.py -i ./footage -o ./tiktok.mp4 -d 30 --activity-percentile 90
```

### 4. Vertical Format for Stories

```bash
python timelapse_editor.py -i ./footage -o ./story.mp4 -d 15 --resolution 1080x1920
```

### 5. Generate & Customize Config

```bash
# Generate config file
python timelapse_editor.py --generate-config config.json

# Edit my_settings.json with your preferences, then:
python timelapse_editor.py -c config.json
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input folder with videos | `./raw_footage` |
| `-o, --output` | Output video path | `./timelapse_output.mp4` |
| `-d, --duration` | Target duration (seconds) | `60` |
| `-q, --quality` | Quality (CRF, lower=better) | `23` |
| `-c, --config` | Use config JSON file | - |
| `--motion-threshold` | Motion sensitivity (0-1) | `0.02` |
| `--activity-percentile` | Keep top X% active | `80` |
| `--min-clip` | Min clip length (seconds) | `0.3` |
| `--max-clip` | Max clip length (seconds) | `2.0` |
| `--no-speed-variation` | Disable speed variation | - |
| `--keep-audio` | Keep original audio | - |
| `--randomize` | Random clip order | - |
| `--resolution` | Output resolution (WxH) | Original |

## Configuration Deep Dive

### Key Settings for Study/Research Timelapses

```json
{
  "target_duration": 60,
  "min_clip_duration": 0.3,
  "max_clip_duration": 1.5,
  "motion_threshold": 0.015,
  "activity_percentile": 85,
  "prefer_typing_motion": true,
  "speed_variation": true,
  "speed_range": [1.0, 1.5]
}
```

### For Coding Sessions (More Selective)

```json
{
  "motion_threshold": 0.01,
  "activity_percentile": 90,
  "prefer_typing_motion": true,
  "min_clip_duration": 0.2,
  "max_clip_duration": 1.0
}
```

### For Art/Design Work (Capture More)

```json
{
  "motion_threshold": 0.03,
  "activity_percentile": 70,
  "prefer_typing_motion": false,
  "max_clip_duration": 3.0
}
```

---

## 🎯 Tips for Entertaining Study Timelapses

### Recording Setup

- **Camera angle**: Slightly above and to the side (45° works great)
- **Lighting**: Soft, consistent lighting reduces flickering artifacts
- **Multi-angle**: Consider recording both yourself AND your screen
- **Clean desk**: Minimal distractions in frame

### Making It Watchable

- **60-90 seconds** is the sweet spot for social media
- **Vary clip lengths**: Mix quick 0.3s cuts with longer 1-2s moments
- **Show progress**: Before/after shots of your work
- **Include milestones**: Finishing chapters, taking breaks, celebrations

### Post-Production

- **Add music**: Lofi hip-hop, electronic, or upbeat tracks (royalty-free!)
- **Cut to the beat**: Sync clip changes with music beats
- **Text overlays**: "Hour 3...", "Finally understood it!", timestamps
- **Progress indicator**: Add a timer or progress bar overlay

### Devon Crawford Style Specifics

1. **Quick cuts** (0.3-0.5 seconds average)
2. **Energetic music** that builds throughout
3. **Text overlays** explaining what's happening
4. **Mix of angles**: Screen recordings + camera footage
5. **Show the grind AND the victories**
6. **End with the result/achievement**

### Audio Ideas

- Keep subtle keyboard sounds (ASMR appeal)
- Add background music in editing software
- Consider a voiceover intro/outro
- Match cuts to beat drops

---

## Workflow Recommendation

1. **Record** your study session (OBS, webcam, screen recorder)
2. **Organize** footage in a single folder
3. **Run script** to create initial timelapse
4. **Import** into video editor (DaVinci Resolve, Premiere, etc.)
5. **Add** music, text overlays, and polish
6. **Export** for your platform (TikTok, YouTube, Instagram)

---

## Troubleshooting

**"No interesting clips found"**

- Lower `motion_threshold` (try 0.01)
- Lower `activity_percentile` (try 70)

**Output too long/short**

- Adjust `target_duration`
- Modify `min_clip_duration` and `max_clip_duration`

**Missing active moments**

- Increase `sample_interval` for more candidates
- Lower `activity_percentile`

**Processing too slow**

- Lower `analysis_fps` (try 1)
- Process videos in smaller batches

---

## License

MIT - Feel free to modify and share!

---

*Created for capturing flow state study sessions. Good luck with your research! 📚*

*Disclaimer*: This entire repository was implemented by Claude Opus 4.5 with a single prompt and it worked perfectly first try! That is impressive and scary at the same time.
