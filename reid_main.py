
import os
import warnings
import cv2
import numpy as np
import torch
import shutil
import time
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Suppress noisy warnings from unused packages (e.g. torchreid Cython)
warnings.filterwarnings("ignore", category=UserWarning)

# Resolve paths relative to this project folder
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

YOLO_PATH = os.path.join(_PROJECT_ROOT, "models_path", "yolo_finetuned_best.pt")

# ---------------- CONFIGURATION ----------------
DETECTION_CONFIDENCE = 0.30  # Lowered to 0.30 to detect small/distant persons
# Similarity threshold to consider a track as a potential match
SIM_THRESHOLD = 0.65  
# How close to the best score other tracks need to be to be included (handling broken tracks)
RELATIVE_SCORE_MARGIN = 0.1 
OUTPUT_FPS = 12
FRAME_SKIP = 1
MAX_FRAMES = float('inf')  # Increased slightly as per user optimization request potential
RESIZE_WIDTH = 640
BATCH_SIZE = 16
MAX_IMAGES_PER_TRACK = 150 # Increased max images per track for better averaging
VIDEO_CODEC = 'mp4v'

# ---------------- CLIP-SPECIFIC CONFIGURATION ----------------
# Absolute similarity floor — track must beat this to be considered
CLIP_SIM_THRESHOLD = 0.20
# Relative margin for broken/re-ID'd track fragments in image mode.
# Widened to 0.10: after occlusion the re-emerging track has fewer frames
# so its avg score is lower — needs a wider margin to still be accepted.
CLIP_RELATIVE_SCORE_MARGIN = 0.10
# For CLIP text mode: very tight margin (scores compressed to ~0.10-0.15)
CLIP_TEXT_RELATIVE_MARGIN = 0.02
# Hard absolute per-frame floor for CLIP TEXT mode.
# Frames below this are dropped even within accepted tracks.
# Prevents wrong people appearing when target exits the frame.
# Tune: raise if wrong person appears, lower if valid frames get dropped.
CLIP_TEXT_ABS_FRAME_THRESHOLD = 0.138
# Hard absolute per-frame floor for CLIP IMAGE mode.
# When bbox drifts to an adjacent wrong person, that frame's score drops.
# ~80% of expected target similarity (if target ~0.25-0.30, floor at 0.18).
# Tune: raise if wrong adjacent person appears; lower if valid frames drop.
CLIP_IMAGE_ABS_FRAME_THRESHOLD = 0.18
# ---------------- HELPER FUNCTIONS ----------------

def prepare_output_dir(output_dir: str):
    """
    Safely cleans and prepares the output directory.
    """
    output_dir = os.path.abspath(output_dir)
    # Django project root (estimated)
    django_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Safety checks
    if not output_dir.startswith(django_root):
        # Fallback for safety if paths are weird, but try to allow valid subdirs
        if "reid_output" not in output_dir:
             print(f"Warning: Output dir {output_dir} might be unsafe. Proceeding with caution.")

    if output_dir == django_root or output_dir == "/":
        raise RuntimeError("Refusing to delete system or project root")

    os.makedirs(output_dir, exist_ok=True)

    # Clean existing content
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    print(f"Output directory cleared and prepared: {output_dir}")

# OSNet/FeatureExtractor removed — this pipeline uses CLIP only.


def _normalize_vector(vec: np.ndarray):
    norm = np.linalg.norm(vec) + 1e-6
    return vec / norm


def _build_query_feature(
    extractor,
    model_type: str,
    clip_query_mode: str,
    input_person_image: str,
    input_person_text: str,
    alpha: float
):
    model_type = (model_type or "clip").lower()
    clip_query_mode = (clip_query_mode or "image").lower()

    if model_type != "clip":
        raise RuntimeError(f"Unsupported model_type: '{model_type}'. This pipeline supports 'clip' only.")

    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise RuntimeError("alpha must be between 0.0 and 1.0")

    image_embedding = None
    text_embedding = None

    if clip_query_mode in ["image", "both"]:
        if not input_person_image:
            raise RuntimeError("CLIP image or both mode requires INPUT_PERSON_IMAGE.")
        query_img = cv2.imread(input_person_image)
        if query_img is None:
            raise RuntimeError(f"Query image not found: {input_person_image}")
        image_feat = extractor.extract_image_embedding(query_img)
        if image_feat is None:
            raise RuntimeError("Could not extract CLIP image query features.")
        image_embedding = image_feat.flatten()

    if clip_query_mode in ["text", "both"]:
        if not input_person_text or not str(input_person_text).strip():
            raise RuntimeError("CLIP text or both mode requires INPUT_PERSON_TEXT.")
        text_feat = extractor.extract_text_embedding(input_person_text)
        if text_feat is None:
            raise RuntimeError("Could not extract CLIP text query features.")
        text_embedding = text_feat.flatten()

    if clip_query_mode == "image":
        return _normalize_vector(image_embedding)
    if clip_query_mode == "text":
        return _normalize_vector(text_embedding)
    if clip_query_mode == "both":
        fused = alpha * image_embedding + (1.0 - alpha) * text_embedding
        return _normalize_vector(fused)

    raise RuntimeError("Invalid clip_query_mode. Use 'image', 'text', or 'both'.")

def compute_track_score(similarities):
    """
    Robust way to score a track based on list of similarities.
    Uses Top-50% Average to discard occlusions/outliers.
    """
    if not similarities:
        return 0.0
    
    # Sort descending
    sims = sorted(similarities, reverse=True)
    
    # Take top 50% (at least 1)
    k = max(1, len(sims) // 2)
    top_k = sims[:k]
    
    # Ensure standard python float
    return float(np.mean(top_k))

def create_video_from_frames(frame_list, output_path, fps=10):
    """
    Create video from a list of image file paths.
    """
    if not frame_list:
        print("No frames to write to video.")
        return False

    print(f"Creating video with {len(frame_list)} frames...")
    
    # Read first image for strict size
    first = cv2.imread(frame_list[0])
    if first is None:
        return False
        
    h, w = first.shape[:2]
    # Ensure even dims
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    size = (w, h)

    codecs = ['avc1', 'H264', 'mp4v', 'XVID']
    writer = None
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            w = cv2.VideoWriter(output_path, fourcc, fps, size)
            if w.isOpened():
                writer = w
                print(f"Crop video codec: {codec}")
                break
            w.release()
        except:
            continue
            
    if not writer or not writer.isOpened():
        print("Failed to open video writer.")
        return False

    count = 0
    for p in frame_list:
        img = cv2.imread(p)
        if img is None: continue
        img_r = cv2.resize(img, size)
        writer.write(img_r)
        count += 1
        
    writer.release()
    print(f"Video saved to {output_path} ({count} frames)")
    return True


def _encode_crop_jpg(crop_img, quality=90):
    """Encode BGR crop as JPEG bytes for in-memory buffering."""
    ok, buf = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return buf.tobytes()


def _decode_crop_jpg(jpg_bytes):
    if jpg_bytes is None:
        return None
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def create_video_from_memory_crops(crop_bytes_list, output_path, fps=10):
    """
    Create video from a list of JPEG-encoded crop bytes.
    Avoids temporary crop files on disk.
    """
    if not crop_bytes_list:
        print("No in-memory crops to write to video.")
        return False

    first = None
    for jpg in crop_bytes_list:
        first = _decode_crop_jpg(jpg)
        if first is not None:
            break

    if first is None:
        print("Could not decode any in-memory crop.")
        return False

    h, w = first.shape[:2]
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    size = (w, h)

    codecs = ['avc1', 'H264', 'mp4v', 'XVID']
    writer = None
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            w = cv2.VideoWriter(output_path, fourcc, fps, size)
            if w.isOpened():
                writer = w
                print(f"Memory crop video codec: {codec}")
                break
            w.release()
        except Exception:
            continue

    if not writer or not writer.isOpened():
        print("Failed to open video writer for in-memory crops.")
        return False

    count = 0
    for jpg in crop_bytes_list:
        img = _decode_crop_jpg(jpg)
        if img is None:
            continue
        img_r = cv2.resize(img, size)
        writer.write(img_r)
        count += 1

    writer.release()
    print(f"Video saved to {output_path} ({count} frames from memory)")
    return count > 0


def dump_selected_crops_to_disk(unique_frames, save_dir):
    """
    Persist only final selected crops to disk for inspection/debugging.
    unique_frames maps timestamp -> {'crop_jpg': bytes, 'bbox': tuple, 'tid': int}
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = 0

    for ts in sorted(unique_frames.keys()):
        info = unique_frames[ts]
        tid = info.get('tid', -1)
        bbox = info.get('bbox')
        jpg = info.get('crop_jpg')
        if jpg is None:
            continue

        tid_dir = os.path.join(save_dir, f"id_{tid}")
        os.makedirs(tid_dir, exist_ok=True)

        if bbox is not None:
            l, t, r, b = bbox
            name = f"{ts:06d}_{l}_{t}_{r}_{b}.jpg"
        else:
            name = f"{ts:06d}.jpg"

        out_path = os.path.join(tid_dir, name)
        with open(out_path, "wb") as f:
            f.write(jpg)
        saved += 1

    return saved

def create_annotated_video(original_video_path, unique_frames, output_path, fps=10):
    """
    Create a full-frame video with bounding boxes for the identified person.
    unique_frames: dict { timestamp: {'bbox': (l,t,r,b), ...} }
    """
    print(f"Creating annotated full-frame video from {original_video_path}...")
    
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print("Could not open original video for annotation.")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # Always use the provided target fps (OUTPUT_FPS = 12) for the output
    write_fps = fps
    
    print(f"Writing annotated video at target FPS: {write_fps} (Input FPS: {input_fps})")
    
    # Ensure even dims
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    size = (width, height)
    
    # Init writer — H264 first for browser compatibility
    codecs = ['avc1', 'H264', 'mp4v', 'XVID']
    writer = None
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            w = cv2.VideoWriter(output_path, fourcc, int(write_fps), size)
            if w.isOpened():
                writer = w
                print(f"Annotated video codec: {codec}")
                break
            w.release()
        except:
            continue

    if not writer or not writer.isOpened():
        cap.release()
        print("Failed to open annotated video writer.")
        return False

    frame_idx = 0
    frames_written = 0
    
    time_per_target_frame = 1.0 / write_fps
    next_process_time = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1 # 1-based index to match extraction
        current_time = frame_idx / input_fps
        
        # Precise framerate skipping
        if current_time < next_process_time:
            continue
            
        next_process_time += time_per_target_frame
        
        # Draw on frame if target matches
        if frame_idx in unique_frames:
            info = unique_frames[frame_idx]
            bbox = info.get('bbox')
            
            if bbox:
                l, t, r, b = bbox
                # Draw Box
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                # Add Label
                cv2.putText(frame, "Target", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Resize if needed to match writer size
        if frame.shape[1] != width or frame.shape[0] != height:
             frame = cv2.resize(frame, size)
             
        writer.write(frame)
        frames_written += 1
            
    cap.release()
    writer.release()
    print(f"Annotated video saved to {output_path} ({frames_written} frames)")
    return True


# ---------------- MAIN PIPELINE ----------------

def run_reid(
    VIDEO_PATH,
    INPUT_PERSON_IMAGE,
    OUTPUT_DIR,
    model_type="osnet",
    clip_query_mode="image",
    INPUT_PERSON_TEXT=None,
    alpha=0.5,
):
    print("=== ReID Pipeline Started (Optimized) ===")
    
    # 1. Setup Directories — clean previous run artifacts
    prepare_output_dir(OUTPUT_DIR)
    
    SELECTED_CROPS_DIR = os.path.join(OUTPUT_DIR, "selected_crops")
    OUTPUT_CLIP_PATH = os.path.join(OUTPUT_DIR, "output.mp4")
    OUTPUT_ANNOTATED_PATH = os.path.join(OUTPUT_DIR, "output_annotated.mp4")

    # 2. Init Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_type = (model_type or "osnet").lower()
    clip_query_mode = (clip_query_mode or "image").lower()

    if model_type == "clip":
        from clip_inference import ClipFeatureExtractor
        extractor = ClipFeatureExtractor(
            device=device,
            image_ckpt_path=os.path.join(_PROJECT_ROOT, "clip_models", "best_model_image.pth"),
            text_ckpt_path=os.path.join(_PROJECT_ROOT, "clip_models", "best_model_text.pth"),
        )
    else:
        extractor = FeatureExtractor(device)
    
    try:
        yolo = YOLO(YOLO_PATH) # Changed to nano for speed
        yolo.overrides["conf"] = DETECTION_CONFIDENCE
        print("YOLOv8n loaded.")
    except Exception as e:
        raise RuntimeError(f"YOLO load failed: {e}")

    tracker = DeepSort(
        max_age=90,    # keep track alive ~7s at 12fps — survives long occlusions
        n_init=2,      # confirm track after 2 detections — faster re-ID after occlusion
        embedder="mobilenet",
        embedder_gpu=False
    )

    # 3. Build Query Feature
    query_feat = _build_query_feature(
        extractor=extractor,
        model_type=model_type,
        clip_query_mode=clip_query_mode,
        input_person_image=INPUT_PERSON_IMAGE,
        input_person_text=INPUT_PERSON_TEXT,
        alpha=alpha,
    )

    # 4. Video Processing Loop
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0: input_fps = 30 # fallback
    
    target_fps = OUTPUT_FPS
    print(f"Input FPS: {input_fps:.2f}, Target FPS: {target_fps}")
    
    # Data structure to hold extensive track info (kept in memory)
    # tracks_db[track_id] = list of { 'timestamp': int, 'crop_jpg': bytes, 'sim': float or None }
    tracks_db = {}
    
    frame_idx = 0
    processed_frames = 0
    
    # Optimization: Extract ReID features only every N frames to save time
    # We still track every frame for smoothness.
    REID_EXTRACTION_INTERVAL = 3 
    
    print(f"Processing video frames (ReID interval: {REID_EXTRACTION_INTERVAL})...")
    
    # Track when the next frame should be processed based on target 12FPS
    time_per_target_frame = 1.0 / target_fps
    next_process_time = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret or processed_frames > MAX_FRAMES:
            break
            
        frame_idx += 1
        current_time = frame_idx / input_fps
        
        # Precise framerate skipping
        if current_time < next_process_time:
            continue
            
        # Update next target time (aligns perfectly)
        next_process_time += time_per_target_frame
        
        # Additional manual frame skip if defined in config
        if frame_idx % FRAME_SKIP != 0:
            continue
            
        processed_frames += 1
        
        # Resize for consistent processing speed
        h, w = frame.shape[:2]
        scale = 1.0
        if w > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w
            frame_s = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            frame_s = frame

        # YOLO Detection
        results = yolo(frame_s, verbose=False, classes=[0]) # class 0 is person
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                # Scale back to original
                if scale != 1.0:
                    x1 /= scale
                    x2 /= scale
                    y1 /= scale
                    y2 /= scale
                
                if conf < DETECTION_CONFIDENCE:
                    continue
                    
                w_b = x2 - x1
                h_b = y2 - y1
                detections.append(([x1, y1, w_b, h_b], conf, 'person'))

        # Tracker Update
        tracks = tracker.update_tracks(detections, frame=frame) # Pass original frame for embedder if needed

        for track in tracks:
            if not track.is_confirmed():
                continue
                
            tid = track.track_id
            ltrb = track.to_ltrb() # left, top, right, bottom
            
            # Crop
            l, t, r, b = [int(x) for x in ltrb]
            l, t = max(0, l), max(0, t)
            r, b = min(w, r), min(h, b)
            
            if r - l < 10 or b - t < 10:
                continue
                
            crop_img = frame[t:b, l:r]
            
            # --- REAL-TIME REID (OPTIMIZED) ---
            # Only run expensive feature extraction periodically
            should_extract = (processed_frames % REID_EXTRACTION_INTERVAL == 0)
            
            sim = None
            if should_extract:
                # Use the correct image encoder based on query mode:
                # - CLIP text/both mode: use IRRA's visual encoder (same embedding space as text)
                # - CLIP image mode / OSNet: use the default image encoder
                if model_type == "clip" and clip_query_mode in ("text", "both"):
                    feat = extractor.extract_image_embedding_for_text_query(crop_img)
                else:
                    feat = extractor.extract_image_embedding(crop_img)
                if feat is not None:
                    feat = feat.flatten()
                    sim_val = np.dot(query_feat, feat)
                    sim = float(sim_val)
            
            # Keep crop in memory; no per-frame disk write.
            crop_jpg = _encode_crop_jpg(crop_img)
            if crop_jpg is None:
                continue
            
            # Record Data
            if tid not in tracks_db:
                tracks_db[tid] = []
                
            tracks_db[tid].append({
                'timestamp': frame_idx,
                'crop_jpg': crop_jpg,
                'sim': sim,
                'bbox': (l, t, r, b)
            })
            
        if processed_frames % 20 == 0:
            print(f"Processed {processed_frames} frames...", end='\r')

    cap.release()
    print("\nVideo processing complete.")
    
    # 5. Global Track Analysis & Matching
    if not tracks_db:
        raise RuntimeError("No tracks found.")
        
    print("Analyzing tracks...")
    
    final_stats = []
    
    for tid, frames in tracks_db.items():
        # diverse_sims: Collect all computed similarities for track scoring
        valid_sims = [f['sim'] for f in frames if f['sim'] is not None]
        
        score = compute_track_score(valid_sims)
        
        final_stats.append({
            'tid': tid, 
            'score': score,
            'count': len(frames),
            'n_sims': len(valid_sims),
            'frames': frames
        })
    
    # Sort tracks by score
    final_stats.sort(key=lambda x: x['score'], reverse=True)
    
    if not final_stats:
        print("No valid tracks.")
        return None, {}

    best_match = final_stats[0]
    best_score = best_match['score']
    
    print(f"Best match: ID {best_match['tid']} with Score {best_score:.3f}")
    
    # --- Diagnostic: show ALL track scores for debugging ---
    print(f"[Debug] All track scores (top 15):")
    for rank, t in enumerate(final_stats[:15], 1):
        sims_list = [f['sim'] for f in t['frames'] if f['sim'] is not None]
        sim_range = f"sim_range=[{min(sims_list):.4f}, {max(sims_list):.4f}]" if sims_list else "no_sims"
        print(f"  #{rank:2d}  Track {t['tid']:>4s}  score={t['score']:.4f}  "
              f"frames={t['count']:>4d}  sims_computed={t['n_sims']:>3d}  {sim_range}")

    # Model-aware selection logic:
    # - OSNet keeps absolute + relative thresholding
    # - CLIP uses its own thresholds calibrated for CLIP similarity distributions
    if model_type == "clip":
        if clip_query_mode == "text":
            # Text mode: accept best track + close fragments of same person
            effective_threshold = best_score - CLIP_TEXT_RELATIVE_MARGIN
            # Hard per-frame floor: drop frames below this even within accepted tracks.
            # Prevents false positives when the actual target is out of frame.
            FRAME_MIN_THRESHOLD = CLIP_TEXT_ABS_FRAME_THRESHOLD
        else:
            clip_abs_threshold = CLIP_SIM_THRESHOLD
            effective_threshold = max(clip_abs_threshold, best_score - CLIP_RELATIVE_SCORE_MARGIN)
            # Hard per-frame floor: drops frames where bbox drifted to wrong adjacent person
            FRAME_MIN_THRESHOLD = CLIP_IMAGE_ABS_FRAME_THRESHOLD
    else:
        effective_threshold = max(SIM_THRESHOLD, best_score - RELATIVE_SCORE_MARGIN)
        # Also define a minimum frame-level threshold to prune bad segments within a good track
        FRAME_MIN_THRESHOLD = effective_threshold * 0.90 
    
    if model_type == "clip":
        print(
            f"Selection criteria (CLIP {clip_query_mode}): Track Score >= {effective_threshold:.3f}, "
            f"Frame Threshold >= {FRAME_MIN_THRESHOLD:.3f}"
        )
    else:
        print(f"Selection criteria (OSNet): Track Score >= {effective_threshold:.3f}, Frame Threshold >= {FRAME_MIN_THRESHOLD:.3f}")
    
    # Store information about the best match to return to the view if needed
    best_track_info = {
        'id': best_match['tid'],
        'score': float(best_score),
        'model_type': model_type,
        'clip_query_mode': clip_query_mode if model_type == 'clip' else None,
        'alpha': float(alpha) if model_type == 'clip' and clip_query_mode == 'both' else None,
    }
    
    # Deduplicate timestamps
    unique_frames = {}   # timestamp -> path (prioritizing higher score tracks)
    
    for t in final_stats:
        if t['score'] >= effective_threshold:
            print(f"-> Accepting Person {t['tid']} (Avg Score: {t['score']:.3f}, Frames: {t['count']})")
            
            # Local cleaning: Iterate frames and check similarities
            # Problem: Sim is only available every N frames.
            # Solution: Fill missing sims with nearest neighbor (forward fill)
            
            frames = t['frames']
            last_valid_sim = 0.0
            
            # Find first valid sim to init
            for f in frames:
                if f['sim'] is not None:
                    last_valid_sim = f['sim']
                    break
            
            accepted_frames_count = 0
            
            # For each frame, we check if the localized similarity is good enough.
            # If a chunk of frames (like 200 frames) has low similarity, it will be discarded.
            for f in frames:
                current_sim = f['sim']
                if current_sim is not None:
                    last_valid_sim = current_sim
                
                # Check if this segment matches the person
                # We use the nearest valid similarity
                if last_valid_sim >= FRAME_MIN_THRESHOLD:
                    ts = f['timestamp']
                    
                    # If duplicate, High Score Track already took it?
                    # final_stats is sorted by score descending.
                    # So if we are first, we take it.
                    if ts not in unique_frames:
                        unique_frames[ts] = {
                            'crop_jpg': f.get('crop_jpg'),
                            'bbox': f.get('bbox'),
                            'tid': t['tid']
                        }
                        accepted_frames_count += 1
            
            print(f"   -> Kept {accepted_frames_count}/{len(frames)} frames after quality filter.")
            
        else:
            print(f"-> Rejecting Person {t['tid']} (Score: {t['score']:.3f})")

    if not unique_frames:
        print("No tracks met the similarity threshold.")
        return None, {}

    # Sort accepted frames by timestamp
    sorted_timestamps = sorted(unique_frames.keys())
    
    # 6. Generate "Cropped" Output Video (Existing Style)
    final_crops = [unique_frames[ts]['crop_jpg'] for ts in sorted_timestamps]
    print(f"Compiling cropped video from {len(final_crops)} in-memory segments...")
    success_crop = create_video_from_memory_crops(final_crops, OUTPUT_CLIP_PATH, fps=OUTPUT_FPS)

    # Persist only final selected crops (no intermediate track crops on disk).
    saved_selected = dump_selected_crops_to_disk(unique_frames, SELECTED_CROPS_DIR)
    print(f"Saved {saved_selected} selected crops to: {SELECTED_CROPS_DIR}")
    
    # 7. Generate "Annotated" Output Video (New Style)
    success_annotated = create_annotated_video(VIDEO_PATH, unique_frames, OUTPUT_ANNOTATED_PATH, fps=OUTPUT_FPS)
    
    print("=== Video generation complete ===")

    # Pack results
    results = {
        'crop': OUTPUT_CLIP_PATH if success_crop else None,
        'annotated': OUTPUT_ANNOTATED_PATH if success_annotated else None
    }
    
    if success_crop or success_annotated:
        print("=== ReID Finished Successfully ===")
        return results, best_track_info
    else:
        print("Video generation failed.")
        return None, {}


# ---------------- MULTI-VIDEO PIPELINE ----------------

def run_reid_multi(
    video_paths: list,
    INPUT_PERSON_IMAGE,
    OUTPUT_DIR,
    model_type="osnet",
    clip_query_mode="image",
    INPUT_PERSON_TEXT=None,
    alpha=0.5,
):
    """
    Run the ReID pipeline on multiple videos sequentially.

    Each video is processed independently with a fresh DeepSort tracker.
    Models (YOLO + feature extractor) are loaded ONCE and reused.
    The query feature is also built ONCE from the provided image/text.

    Args:
        video_paths: list of absolute video file paths to process
        INPUT_PERSON_IMAGE: path to query image (used for OSNet / CLIP image/both)
        OUTPUT_DIR: root output directory; per-video sub-dirs are created here
        model_type: "osnet" or "clip"
        clip_query_mode: "image", "text", or "both"
        INPUT_PERSON_TEXT: text query string (CLIP text/both modes)
        alpha: image/text blend weight for CLIP-both mode

    Returns:
        all_results: list of dicts, one per video:
            {
                'video_name': str,
                'video_path': str,
                'output_dir': str,
                'results': dict or None,   # {'crop': path, 'annotated': path}
                'info': dict,              # best_track_info
                'error': str or None,
            }
    """
    n = len(video_paths)
    sep = "=" * 60

    print(sep)
    print(f"  Multi-Video ReID Pipeline -- {n} video(s) to process")
    print(sep)

    # ── Prepare root output dir ───────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load models ONCE ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Device: {device}")

    model_type = (model_type or "osnet").lower()
    clip_query_mode = (clip_query_mode or "image").lower()

    print(f"[Setup] Loading feature extractor ({model_type})...")
    if model_type == "clip":
        from clip_inference import ClipFeatureExtractor
        extractor = ClipFeatureExtractor(
            device=device,
            image_ckpt_path=os.path.join(_PROJECT_ROOT, "clip_models", "best_model_image.pth"),
            text_ckpt_path=os.path.join(_PROJECT_ROOT, "clip_models", "best_model_text.pth"),
        )
    else:
        extractor = FeatureExtractor(device)
    print(f"[Setup] Feature extractor ready.")

    print(f"[Setup] Loading YOLO...")
    try:
        yolo = YOLO(YOLO_PATH)
        yolo.overrides["conf"] = DETECTION_CONFIDENCE
        print(f"[Setup] YOLO loaded.")
    except Exception as e:
        raise RuntimeError(f"YOLO load failed: {e}")

    # ── Build query feature ONCE ──────────────────────────────────────────────
    print(f"[Setup] Building query feature ({model_type})...")
    query_feat = _build_query_feature(
        extractor=extractor,
        model_type=model_type,
        clip_query_mode=clip_query_mode,
        input_person_image=INPUT_PERSON_IMAGE,
        input_person_text=INPUT_PERSON_TEXT,
        alpha=alpha,
    )
    print(f"[Setup] Query feature ready. Shape: {query_feat.shape}")

    # ── Process each video ────────────────────────────────────────────────────
    all_results = []

    for idx, video_path in enumerate(video_paths, start=1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{sep}")
        print(f"  [{idx}/{n}] Processing: {os.path.basename(video_path)}")
        print(sep)

        # Per-video output directory
        video_output_dir = os.path.join(OUTPUT_DIR, f"video_{idx:02d}_{video_name}")
        os.makedirs(video_output_dir, exist_ok=True)

        SELECTED_CROPS_DIR = os.path.join(video_output_dir, "selected_crops")
        OUTPUT_CLIP_PATH = os.path.join(video_output_dir, "output.mp4")
        OUTPUT_ANNOTATED_PATH = os.path.join(video_output_dir, "output_annotated.mp4")

        # ── Reinitialize DeepSort tracker for each video ──────────────────────
        print(f"  [Tracker] Reinitializing DeepSort for video {idx}...")
        tracker = DeepSort(
            max_age=90,    # keep track alive ~7s at 12fps — survives long occlusions
            n_init=2,      # confirm track after 2 detections — faster re-ID after occlusion
            embedder="mobilenet",
            embedder_gpu=False
        )
        print(f"  [Tracker] DeepSort ready.")

        # ── Open video ────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            err_msg = f"Cannot open video: {video_path}"
            print(f"  [ERROR] {err_msg}")
            all_results.append({
                'video_name': video_name,
                'video_path': video_path,
                'output_dir': video_output_dir,
                'results': None,
                'info': {},
                'error': err_msg,
            })
            continue

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if input_fps <= 0:
            input_fps = 30
        duration_s = total_frames / input_fps if input_fps > 0 else 0

        print(f"  [Video]   FPS: {input_fps:.1f}  |  Total frames: {total_frames}  |  "
              f"Duration: {duration_s:.1f}s")
        print(f"  [Video]   Output target FPS: {OUTPUT_FPS}")
        print(f"  [Video]   Starting frame extraction & tracking...")

        # ── Frame loop ────────────────────────────────────────────────────────
        tracks_db = {}
        frame_idx = 0
        processed_frames = 0
        REID_EXTRACTION_INTERVAL = 3
        target_fps = OUTPUT_FPS
        time_per_target_frame = 1.0 / target_fps
        next_process_time = 0.0

        _t_start = time.time()
        _last_report = 0

        while True:
            ret, frame = cap.read()
            if not ret or processed_frames > MAX_FRAMES:
                break

            frame_idx += 1
            current_time = frame_idx / input_fps

            if current_time < next_process_time:
                continue
            next_process_time += time_per_target_frame

            if frame_idx % FRAME_SKIP != 0:
                continue
            processed_frames += 1

            h, w = frame.shape[:2]
            scale = 1.0
            if w > RESIZE_WIDTH:
                scale = RESIZE_WIDTH / w
                frame_s = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            else:
                frame_s = frame

            # YOLO Detection
            results_yolo = yolo(frame_s, verbose=False, classes=[0])
            detections = []
            for r in results_yolo:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    if scale != 1.0:
                        x1 /= scale; x2 /= scale
                        y1 /= scale; y2 /= scale
                    if conf < DETECTION_CONFIDENCE:
                        continue
                    w_b = x2 - x1
                    h_b = y2 - y1
                    detections.append(([x1, y1, w_b, h_b], conf, 'person'))

            # Tracker update
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                ltrb = track.to_ltrb()
                l, t, r, b = [int(x) for x in ltrb]
                l, t = max(0, l), max(0, t)
                r, b = min(w, r), min(h, b)
                if r - l < 10 or b - t < 10:
                    continue

                crop_img = frame[t:b, l:r]

                should_extract = (processed_frames % REID_EXTRACTION_INTERVAL == 0)
                sim = None
                if should_extract:
                    if model_type == "clip" and clip_query_mode in ("text", "both"):
                        feat = extractor.extract_image_embedding_for_text_query(crop_img)
                    else:
                        feat = extractor.extract_image_embedding(crop_img)
                    if feat is not None:
                        feat = feat.flatten()
                        sim = float(np.dot(query_feat, feat))

                crop_jpg = _encode_crop_jpg(crop_img)
                if crop_jpg is None:
                    continue

                if tid not in tracks_db:
                    tracks_db[tid] = []
                tracks_db[tid].append({
                    'timestamp': frame_idx,
                    'crop_jpg': crop_jpg,
                    'sim': sim,
                    'bbox': (l, t, r, b)
                })

            # ── Progress reporting every 5 seconds ────────────────────────────
            elapsed_s = time.time() - _t_start
            if elapsed_s - _last_report >= 5.0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                fps_proc = processed_frames / elapsed_s if elapsed_s > 0 else 0
                eta_s = ((total_frames - frame_idx) / input_fps / (fps_proc / target_fps + 1e-6)
                         if fps_proc > 0 else 0)
                tracks_found = len(tracks_db)
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = "#" * filled + "." * (bar_len - filled)
                print(
                    f"  [Progress {idx}/{n}] [{bar}] {pct:5.1f}%  "
                    f"frame {frame_idx}/{total_frames}  |  "
                    f"{fps_proc:.1f} proc-fps  |  "
                    f"ETA {eta_s:.0f}s  |  "
                    f"tracks so far: {tracks_found}"
                )
                _last_report = elapsed_s

        cap.release()
        elapsed_total = time.time() - _t_start
        print(f"  [Done]    Frame loop finished in {elapsed_total:.1f}s  "
              f"({processed_frames} frames processed, {len(tracks_db)} unique tracks)")

        # ── Track analysis ────────────────────────────────────────────────────
        if not tracks_db:
            print(f"  [Result]  No tracks found in video {idx}. Skipping.")
            all_results.append({
                'video_name': video_name,
                'video_path': video_path,
                'output_dir': video_output_dir,
                'results': None,
                'info': {},
                'error': "No tracks found.",
            })
            continue

        print(f"  [Analysis] Scoring {len(tracks_db)} tracks...")
        final_stats = []
        for tid, frames in tracks_db.items():
            valid_sims = [f['sim'] for f in frames if f['sim'] is not None]
            score = compute_track_score(valid_sims)
            final_stats.append({'tid': tid, 'score': score, 'count': len(frames),
                                'n_sims': len(valid_sims), 'frames': frames})
        final_stats.sort(key=lambda x: x['score'], reverse=True)

        best_match = final_stats[0]
        best_score = best_match['score']
        print(f"  [Analysis] Best match: ID {best_match['tid']}  Score: {best_score:.4f}")

        # --- Diagnostic: show ALL track scores for debugging ---
        print(f"  [Debug] All track scores (top 15):")
        for rank, t in enumerate(final_stats[:15], 1):
            sims_list = [f['sim'] for f in t['frames'] if f['sim'] is not None]
            sim_range = f"sim_range=[{min(sims_list):.4f}, {max(sims_list):.4f}]" if sims_list else "no_sims"
            print(f"    #{rank:2d}  Track {t['tid']:>4s}  score={t['score']:.4f}  "
                  f"frames={t['count']:>4d}  sims_computed={t['n_sims']:>3d}  {sim_range}")

        # Threshold logic (same as single-video)
        if model_type == "clip":
            if clip_query_mode == "text":
                effective_threshold = best_score - CLIP_TEXT_RELATIVE_MARGIN
                # Hard per-frame floor for text mode — drops frames when target is absent
                FRAME_MIN_THRESHOLD = CLIP_TEXT_ABS_FRAME_THRESHOLD
            else:
                clip_abs_threshold = CLIP_SIM_THRESHOLD
                effective_threshold = max(clip_abs_threshold, best_score - CLIP_RELATIVE_SCORE_MARGIN)
                # Hard per-frame floor for image/hybrid mode — drops wrong adjacent-person frames
                # Using the configured constant (0.15) instead of effective_threshold*0.85 (was 0.65!)
                FRAME_MIN_THRESHOLD = CLIP_IMAGE_ABS_FRAME_THRESHOLD
        else:
            effective_threshold = max(SIM_THRESHOLD, best_score - RELATIVE_SCORE_MARGIN)
            FRAME_MIN_THRESHOLD = effective_threshold * 0.90

        print(f"  [Analysis] Acceptance threshold: {effective_threshold:.4f}  "
              f"Frame min: {FRAME_MIN_THRESHOLD:.4f}")

        best_track_info = {
            'id': best_match['tid'],
            'score': float(best_score),
            'model_type': model_type,
            'clip_query_mode': clip_query_mode if model_type == 'clip' else None,
            'alpha': float(alpha) if model_type == 'clip' and clip_query_mode == 'both' else None,
        }

        unique_frames = {}
        for t in final_stats:
            if t['score'] >= effective_threshold:
                print(f"  [Select]   + Accepting track {t['tid']}  "
                      f"(score={t['score']:.4f}, {t['count']} frames)")
                frames = t['frames']
                last_valid_sim = 0.0
                for f in frames:
                    if f['sim'] is not None:
                        last_valid_sim = f['sim']
                        break
                accepted_count = 0
                for f in frames:
                    if f['sim'] is not None:
                        last_valid_sim = f['sim']
                    if last_valid_sim >= FRAME_MIN_THRESHOLD:
                        ts = f['timestamp']
                        if ts not in unique_frames:
                            unique_frames[ts] = {
                                'crop_jpg': f.get('crop_jpg'),
                                'bbox': f.get('bbox'),
                                'tid': t['tid']
                            }
                            accepted_count += 1
                print(f"             Kept {accepted_count}/{len(frames)} frames after quality filter.")
            else:
                print(f"  [Select]   - Rejecting track {t['tid']}  "
                      f"(score={t['score']:.4f} < {effective_threshold:.4f})")

        if not unique_frames:
            print(f"  [Result]  No frames passed quality filter for video {idx}.")
            all_results.append({
                'video_name': video_name,
                'video_path': video_path,
                'output_dir': video_output_dir,
                'results': None,
                'info': best_track_info,
                'error': "No frames met similarity threshold.",
            })
            continue

        sorted_timestamps = sorted(unique_frames.keys())

        # ── Build output videos ───────────────────────────────────────────────
        print(f"  [Output]  Building cropped video ({len(sorted_timestamps)} segments)...")
        final_crops = [unique_frames[ts]['crop_jpg'] for ts in sorted_timestamps]
        success_crop = create_video_from_memory_crops(final_crops, OUTPUT_CLIP_PATH, fps=OUTPUT_FPS)

        saved_selected = dump_selected_crops_to_disk(unique_frames, SELECTED_CROPS_DIR)
        print(f"  [Output]  Saved {saved_selected} selected crops to {SELECTED_CROPS_DIR}")

        print(f"  [Output]  Building annotated video...")
        success_annotated = create_annotated_video(
            video_path, unique_frames, OUTPUT_ANNOTATED_PATH, fps=OUTPUT_FPS
        )

        import subprocess
        def _convert_h264(vid_path):
            temp_path = vid_path.replace(".mp4", "_h264.mp4")
            try:
                print(f"  [FFmpeg]  Converting {os.path.basename(vid_path)} to H.264...")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", vid_path, "-vcodec", "libx264", temp_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if os.path.exists(temp_path):
                    os.replace(temp_path, vid_path)
                print(f"  [FFmpeg]  Conversion done.")
            except Exception as e:
                print(f"  [FFmpeg]  Conversion failed: {e}")

        if success_crop and os.path.exists(OUTPUT_CLIP_PATH):
            _convert_h264(OUTPUT_CLIP_PATH)
        if success_annotated and os.path.exists(OUTPUT_ANNOTATED_PATH):
            _convert_h264(OUTPUT_ANNOTATED_PATH)

        video_results = {
            'crop': OUTPUT_CLIP_PATH if success_crop else None,
            'annotated': OUTPUT_ANNOTATED_PATH if success_annotated else None,
        }

        all_results.append({
            'video_name': video_name,
            'video_path': video_path,
            'output_dir': video_output_dir,
            'results': video_results,
            'info': best_track_info,
            'error': None,
        })

        print(f"  [Done]  Video {idx}/{n} complete  ->  {video_output_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  Multi-Video ReID Pipeline -- COMPLETE")
    print(f"  {n} video(s) processed")
    succeeded = sum(1 for r in all_results if r['error'] is None)
    failed    = n - succeeded
    print(f"  Succeeded: {succeeded}   Failed/no match: {failed}")
    print(sep)

    return all_results
