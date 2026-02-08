#!/usr/bin/env python3
"""
Smart Timelapse Editor - Devon Crawford Style
=============================================

Creates watchable, entertaining timelapse montages from hours of footage
by detecting moments with actual activity/motion rather than just speeding up.

Author: BooleanCube's AI Assistant
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random

import cv2
import numpy as np


@dataclass
class Config:
    """Configuration for the timelapse editor."""
    
    # Input/Output
    input_folder: str = "./raw_footage"
    output_file: str = "./timelapse_output.mp4"
    
    # Timing
    target_duration: int = 60  # Target output duration in seconds
    min_clip_duration: float = 0.3  # Minimum clip length in seconds
    max_clip_duration: float = 2.0  # Maximum clip length in seconds
    
    # Detection thresholds
    motion_threshold: float = 0.02  # Motion detection sensitivity (0-1, lower = more sensitive)
    scene_change_threshold: float = 0.3  # Scene change detection threshold
    activity_percentile: int = 80  # Only keep top X% most active moments
    
    # Sampling
    analysis_fps: int = 2  # FPS for motion analysis (lower = faster processing)
    sample_interval: int = 5  # Seconds between potential clip candidates
    
    # Output settings
    output_fps: int = 30
    output_resolution: Optional[Tuple[int, int]] = None  # None = keep original
    output_codec: str = "libx264"
    output_quality: int = 23  # CRF value (lower = better quality, 18-28 recommended)
    
    # Style options
    speed_variation: bool = True  # Vary playback speed for effect
    speed_range: Tuple[float, float] = (1.0, 2.0)  # Speed multiplier range
    add_transitions: bool = False  # Add crossfade transitions
    transition_duration: float = 0.1  # Transition duration in seconds
    
    # Audio
    keep_audio: bool = False  # Keep original audio (usually False for timelapse)
    
    # Advanced
    prefer_typing_motion: bool = True  # Prioritize keyboard/mouse movement patterns
    avoid_static: bool = True  # Skip completely static sections
    randomize_order: bool = False  # Randomize clip order (False = chronological)
    
    def to_dict(self) -> dict:
        return {
            "input_folder": self.input_folder,
            "output_file": self.output_file,
            "target_duration": self.target_duration,
            "min_clip_duration": self.min_clip_duration,
            "max_clip_duration": self.max_clip_duration,
            "motion_threshold": self.motion_threshold,
            "scene_change_threshold": self.scene_change_threshold,
            "activity_percentile": self.activity_percentile,
            "analysis_fps": self.analysis_fps,
            "sample_interval": self.sample_interval,
            "output_fps": self.output_fps,
            "output_resolution": self.output_resolution,
            "output_codec": self.output_codec,
            "output_quality": self.output_quality,
            "speed_variation": self.speed_variation,
            "speed_range": list(self.speed_range),
            "add_transitions": self.add_transitions,
            "transition_duration": self.transition_duration,
            "keep_audio": self.keep_audio,
            "prefer_typing_motion": self.prefer_typing_motion,
            "avoid_static": self.avoid_static,
            "randomize_order": self.randomize_order,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        if "speed_range" in d and isinstance(d["speed_range"], list):
            d["speed_range"] = tuple(d["speed_range"])
        if "output_resolution" in d and isinstance(d["output_resolution"], list):
            d["output_resolution"] = tuple(d["output_resolution"])
        return cls(**d)
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class ClipCandidate:
    """Represents a potential clip to include in the timelapse."""
    video_path: str
    start_time: float  # seconds
    end_time: float  # seconds
    motion_score: float  # 0-1, higher = more motion
    scene_change_score: float  # 0-1
    activity_type: str  # "typing", "mouse", "general", "transition"
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def combined_score(self) -> float:
        """Combined score for ranking clips."""
        return self.motion_score * 0.7 + self.scene_change_score * 0.3


class VideoAnalyzer:
    """Analyzes video content to find interesting moments."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to probe video: {video_path}")
        return json.loads(result.stdout)
    
    def analyze_motion(self, video_path: str) -> List[Tuple[float, float, float]]:
        """
        Analyze video for motion/activity.
        Returns list of (timestamp, motion_score, scene_change_score).
        """
        print(f"  Analyzing motion in: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame skip for analysis FPS
        frame_skip = max(1, int(fps / self.config.analysis_fps))
        
        results = []
        prev_frame = None
        prev_hist = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                timestamp = frame_idx / fps
                
                # Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                # Calculate histogram for scene change detection
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                motion_score = 0.0
                scene_change_score = 0.0
                
                if prev_frame is not None:
                    # Motion detection using frame difference
                    frame_diff = cv2.absdiff(prev_frame, gray)
                    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
                    motion_score = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
                    
                    # Scene change detection using histogram comparison
                    if prev_hist is not None:
                        scene_change_score = 1.0 - cv2.compareHist(
                            prev_hist, hist, cv2.HISTCMP_CORREL
                        )
                
                results.append((timestamp, motion_score, scene_change_score))
                prev_frame = gray.copy()
                prev_hist = hist.copy()
            
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % (frame_skip * 100) == 0:
                progress = frame_idx / total_frames * 100
                print(f"    Progress: {progress:.1f}%", end="\r")
        
        cap.release()
        print(f"    Analyzed {len(results)} frames from {duration:.1f}s video")
        return results
    
    def detect_typing_patterns(self, motion_data: List[Tuple[float, float, float]]) -> List[float]:
        """
        Detect timestamps that look like typing/coding activity.
        Typing has consistent small motions (screen updates).
        """
        typing_timestamps = []
        window_size = 10  # frames to analyze
        
        for i in range(window_size, len(motion_data) - window_size):
            window = motion_data[i-window_size:i+window_size]
            motions = [m[1] for m in window]
            
            # Typing pattern: consistent small-medium motion, low variance
            mean_motion = np.mean(motions)
            std_motion = np.std(motions)
            
            if (self.config.motion_threshold < mean_motion < 0.15 and 
                std_motion < 0.05):
                typing_timestamps.append(motion_data[i][0])
        
        return typing_timestamps
    
    def find_clip_candidates(self, video_path: str) -> List[ClipCandidate]:
        """Find all potential clips in a video."""
        motion_data = self.analyze_motion(video_path)
        
        if not motion_data:
            return []
        
        # Calculate activity threshold
        all_motion_scores = [m[1] for m in motion_data]
        motion_threshold = np.percentile(
            all_motion_scores, 
            100 - self.config.activity_percentile
        )
        
        # Detect typing patterns if enabled
        typing_timestamps = set()
        if self.config.prefer_typing_motion:
            typing_timestamps = set(self.detect_typing_patterns(motion_data))
        
        candidates = []
        
        # Sample at regular intervals
        sample_timestamps = np.arange(
            0, 
            motion_data[-1][0], 
            self.config.sample_interval
        )
        
        for sample_time in sample_timestamps:
            # Find motion data near this timestamp
            nearby_data = [
                m for m in motion_data 
                if abs(m[0] - sample_time) < self.config.max_clip_duration
            ]
            
            if not nearby_data:
                continue
            
            # Calculate scores for this region
            avg_motion = np.mean([m[1] for m in nearby_data])
            max_scene_change = max([m[2] for m in nearby_data])
            
            # Skip if below activity threshold
            if self.config.avoid_static and avg_motion < self.config.motion_threshold:
                continue
            
            # Determine activity type
            if sample_time in typing_timestamps:
                activity_type = "typing"
            elif max_scene_change > self.config.scene_change_threshold:
                activity_type = "transition"
            elif avg_motion > 0.1:
                activity_type = "mouse"
            else:
                activity_type = "general"
            
            # Only keep if above motion threshold percentile
            if avg_motion >= motion_threshold or activity_type == "transition":
                # Determine clip boundaries
                clip_duration = random.uniform(
                    self.config.min_clip_duration,
                    self.config.max_clip_duration
                )
                
                start_time = max(0, sample_time - clip_duration / 2)
                end_time = start_time + clip_duration
                
                candidates.append(ClipCandidate(
                    video_path=video_path,
                    start_time=start_time,
                    end_time=end_time,
                    motion_score=avg_motion,
                    scene_change_score=max_scene_change,
                    activity_type=activity_type
                ))
        
        print(f"    Found {len(candidates)} clip candidates")
        return candidates


class TimelapseEditor:
    """Main editor that creates the timelapse montage."""
    
    def __init__(self, config: Config):
        self.config = config
        self.analyzer = VideoAnalyzer(config)
        
    def get_video_files(self) -> List[str]:
        """Get all video files from input folder."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        input_path = Path(self.config.input_folder)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_path}")
        
        videos = []
        for ext in video_extensions:
            videos.extend(input_path.glob(f"*{ext}"))
            videos.extend(input_path.glob(f"*{ext.upper()}"))
        
        # Sort by filename (usually gives chronological order)
        videos = sorted(videos, key=lambda x: x.name)
        
        print(f"Found {len(videos)} video files")
        return [str(v) for v in videos]
    
    def select_clips(self, all_candidates: List[ClipCandidate]) -> List[ClipCandidate]:
        """Select the best clips to meet target duration."""
        if not all_candidates:
            return []
        
        # Sort by combined score
        sorted_candidates = sorted(
            all_candidates, 
            key=lambda c: c.combined_score, 
            reverse=True
        )
        
        # Calculate how many clips we need
        avg_clip_duration = (
            self.config.min_clip_duration + self.config.max_clip_duration
        ) / 2
        
        # Account for speed variation
        if self.config.speed_variation:
            avg_speed = sum(self.config.speed_range) / 2
            effective_duration = avg_clip_duration / avg_speed
        else:
            effective_duration = avg_clip_duration
        
        num_clips_needed = int(self.config.target_duration / effective_duration)
        
        # Select top clips
        selected = sorted_candidates[:num_clips_needed]
        
        # Sort by video path and timestamp for chronological order
        if not self.config.randomize_order:
            selected = sorted(
                selected, 
                key=lambda c: (c.video_path, c.start_time)
            )
        else:
            random.shuffle(selected)
        
        print(f"Selected {len(selected)} clips for ~{self.config.target_duration}s output")
        return selected
    
    def extract_clip(self, clip: ClipCandidate, output_path: str, speed: float = 1.0):
        """Extract a single clip using ffmpeg."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(clip.start_time),
            "-i", clip.video_path,
            "-t", str(clip.duration),
        ]
        
        # Video filters
        vfilters = []
        
        # Speed adjustment
        if speed != 1.0:
            vfilters.append(f"setpts={1/speed}*PTS")
        
        # Resolution scaling
        if self.config.output_resolution:
            w, h = self.config.output_resolution
            vfilters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease")
            vfilters.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")
        
        if vfilters:
            cmd.extend(["-vf", ",".join(vfilters)])
        
        # Audio handling
        if self.config.keep_audio:
            if speed != 1.0:
                cmd.extend(["-af", f"atempo={speed}"])
        else:
            cmd.extend(["-an"])
        
        # Output settings
        cmd.extend([
            "-c:v", self.config.output_codec,
            "-crf", str(self.config.output_quality),
            "-preset", "fast",
            "-r", str(self.config.output_fps),
            output_path
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to extract clip: {result.stderr}")
            return False
        return True
    
    def concatenate_clips(self, clip_paths: List[str], output_path: str):
        """Concatenate all clips into final video."""
        # Create concat file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in clip_paths:
                f.write(f"file '{path}'\n")
            concat_file = f.name
        
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c:v", self.config.output_codec,
                "-crf", str(self.config.output_quality),
                "-preset", "medium",
            ]
            
            if not self.config.keep_audio:
                cmd.extend(["-an"])
            
            cmd.append(output_path)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Concatenation failed: {result.stderr}")
                
        finally:
            os.unlink(concat_file)
    
    def create_timelapse(self):
        """Main method to create the timelapse."""
        print("=" * 60)
        print("SMART TIMELAPSE EDITOR")
        print("=" * 60)
        
        # Get all video files
        video_files = self.get_video_files()
        if not video_files:
            print("No video files found in input folder!")
            return
        
        # Analyze all videos and find candidates
        print("\n[1/4] Analyzing videos for interesting moments...")
        all_candidates = []
        for video in video_files:
            candidates = self.analyzer.find_clip_candidates(video)
            all_candidates.extend(candidates)
        
        if not all_candidates:
            print("No interesting clips found! Try lowering the motion threshold.")
            return
        
        # Select best clips
        print("\n[2/4] Selecting best clips...")
        selected_clips = self.select_clips(all_candidates)
        
        if not selected_clips:
            print("No clips selected!")
            return
        
        # Extract clips
        print("\n[3/4] Extracting clips...")
        temp_dir = tempfile.mkdtemp()
        clip_paths = []
        
        try:
            for i, clip in enumerate(selected_clips):
                # Determine speed for this clip
                if self.config.speed_variation:
                    speed = random.uniform(*self.config.speed_range)
                else:
                    speed = 1.0
                
                clip_path = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
                
                print(f"  Extracting clip {i+1}/{len(selected_clips)} "
                      f"({clip.activity_type}, {speed:.1f}x speed)")
                
                if self.extract_clip(clip, clip_path, speed):
                    clip_paths.append(clip_path)
            
            if not clip_paths:
                print("No clips were successfully extracted!")
                return
            
            # Concatenate clips
            print("\n[4/4] Creating final timelapse...")
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.concatenate_clips(clip_paths, str(output_path))
            
            # Get final video info
            final_duration = self._get_duration(str(output_path))
            
            print("\n" + "=" * 60)
            print("TIMELAPSE COMPLETE!")
            print("=" * 60)
            print(f"Output: {output_path}")
            print(f"Duration: {final_duration:.1f} seconds")
            print(f"Clips used: {len(clip_paths)}")
            print("=" * 60)
            
        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir)
    
    def _get_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0


def print_tips():
    """Print tips for creating entertaining study timelapses."""
    tips = """
╔══════════════════════════════════════════════════════════════════════╗
║           TIPS FOR ENTERTAINING STUDY/WORK TIMELAPSES               ║
╚══════════════════════════════════════════════════════════════════════╝

📹 RECORDING SETUP:
  • Use a consistent camera angle - slightly above and to the side works great
  • Good lighting is essential - soft, even lighting reduces flickering
  • Record your screen AND yourself for variety (PiP style)
  • Consider a second angle for B-roll (hands on keyboard, coffee, etc.)

⚡ MAKING IT WATCHABLE:
  • Target 60-90 seconds for social media (this script's default)
  • Vary clip lengths - mix quick 0.3s cuts with longer 1-2s moments
  • Include "milestone moments" - finishing a section, taking a break
  • Show the transformation - before/after of your work

🎵 AUDIO/MUSIC:
  • Add upbeat, royalty-free music (lofi hip-hop, electronic, etc.)
  • Match cuts to the beat for extra polish
  • Consider keeping keyboard sounds (ASMR appeal)

📊 CONTENT VARIETY:
  • Mix typing with scrolling, reading, and note-taking
  • Include screen recordings showing actual progress
  • Add text overlays: "Hour 3...", "Finally got it!", timestamps
  • Show small wins and breakthroughs

🎬 POST-PRODUCTION:
  • Add a progress bar or timer overlay
  • Use subtle zoom effects on key moments
  • Color grade for consistency across all footage
  • Add your logo/watermark

💡 DEVON CRAWFORD STYLE SPECIFICS:
  • Quick cuts (0.3-0.5s average)
  • Energetic music that builds
  • Text overlays explaining what's happening
  • Mix of screen recordings and camera footage
  • Show the grind AND the victories
  • End with the final result/achievement

🔧 SCRIPT RECOMMENDATIONS FOR YOUR USE CASE:
  • For coding/research: enable prefer_typing_motion
  • Increase activity_percentile to 85-90 for more selective clips
  • Use speed_variation for dynamic pacing
  • Set target_duration to 60 for TikTok/Reels, 90 for YouTube Shorts

"""
    print(tips)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Timelapse Editor - Create Devon Crawford style montages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python timelapse_editor.py -i ./footage -o ./output.mp4

  # Create a 90 second timelapse with higher quality
  python timelapse_editor.py -i ./footage -o ./output.mp4 -d 90 -q 18

  # Generate a config file to customize
  python timelapse_editor.py --generate-config my_config.json

  # Use a config file
  python timelapse_editor.py -c my_config.json

  # Show tips for creating great timelapses
  python timelapse_editor.py --tips
        """
    )
    
    parser.add_argument("-i", "--input", help="Input folder with video files")
    parser.add_argument("-o", "--output", help="Output video file path")
    parser.add_argument("-d", "--duration", type=int, help="Target duration in seconds")
    parser.add_argument("-q", "--quality", type=int, help="Output quality (CRF, lower=better, 18-28)")
    parser.add_argument("-c", "--config", help="Path to config JSON file")
    parser.add_argument("--generate-config", metavar="PATH", help="Generate a config file template")
    parser.add_argument("--tips", action="store_true", help="Show tips for creating great timelapses")
    parser.add_argument("--motion-threshold", type=float, help="Motion detection sensitivity (0-1)")
    parser.add_argument("--activity-percentile", type=int, help="Keep top X%% most active moments")
    parser.add_argument("--min-clip", type=float, help="Minimum clip duration in seconds")
    parser.add_argument("--max-clip", type=float, help="Maximum clip duration in seconds")
    parser.add_argument("--no-speed-variation", action="store_true", help="Disable speed variation")
    parser.add_argument("--keep-audio", action="store_true", help="Keep original audio")
    parser.add_argument("--randomize", action="store_true", help="Randomize clip order")
    parser.add_argument("--resolution", help="Output resolution (e.g., 1920x1080)")
    
    args = parser.parse_args()
    
    # Show tips if requested
    if args.tips:
        print_tips()
        return
    
    # Generate config file if requested
    if args.generate_config:
        config = Config()
        config.save(args.generate_config)
        print(f"\nGenerated config file: {args.generate_config}")
        print("Edit this file to customize your timelapse settings.")
        return
    
    # Load config
    if args.config:
        config = Config.load(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.input:
        config.input_folder = args.input
    if args.output:
        config.output_file = args.output
    if args.duration:
        config.target_duration = args.duration
    if args.quality:
        config.output_quality = args.quality
    if args.motion_threshold:
        config.motion_threshold = args.motion_threshold
    if args.activity_percentile:
        config.activity_percentile = args.activity_percentile
    if args.min_clip:
        config.min_clip_duration = args.min_clip
    if args.max_clip:
        config.max_clip_duration = args.max_clip
    if args.no_speed_variation:
        config.speed_variation = False
    if args.keep_audio:
        config.keep_audio = True
    if args.randomize:
        config.randomize_order = True
    if args.resolution:
        w, h = map(int, args.resolution.split("x"))
        config.output_resolution = (w, h)
    
    # Create timelapse
    editor = TimelapseEditor(config)
    editor.create_timelapse()


if __name__ == "__main__":
    main()
