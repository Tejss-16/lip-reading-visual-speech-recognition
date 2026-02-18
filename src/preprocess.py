# preprocessing.py
import cv2
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm
import re

# --- Configuration ---
BASE_DIR = "mvlrs_v1"       # Base dataset folder
OUTPUT_DIR = "output"       # Output folder for frames + labels
MAX_CLIPS = 15000            # Limit number of clips per split (e.g., 10), or None for all

# --- Path Definitions ---
DATASET_DIR = os.path.join(BASE_DIR, "main")
FILELIST_DIR = os.path.join(BASE_DIR, "filelists")

# --- Create Output Folders ---
print("Creating output directories...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# --- Initialize Mediapipe ---
print("Initializing MediaPipe FaceMesh...")
mp_face_mesh = mp.solutions.face_mesh


def extract_mouth_and_get_transcript(video_path, save_dir, transcript_text):
    """Extract mouth regions from video frames and save them as images."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Collect all face landmarks
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                    # Only lip indices from FACEMESH_LIPS
                    mouth_indices_set = set()
                    for conn in mp_face_mesh.FACEMESH_LIPS:
                        mouth_indices_set.add(conn[0])
                        mouth_indices_set.add(conn[1])
                    mouth_indices = list(mouth_indices_set)

                    # Ensure valid indices
                    valid_indices = [i for i in mouth_indices if i < len(points)]
                    if not valid_indices:
                        continue

                    mouth_coords = [(points[i][0], points[i][1]) for i in valid_indices]

                    x_min = min([x for x, y in mouth_coords])
                    y_min = min([y for x, y in mouth_coords])
                    x_max = max([x for x, y in mouth_coords])
                    y_max = max([y for x, y in mouth_coords])

                    # --- Improved padding ---
                    x_pad = int(0.40 * (x_max - x_min))  # 35% wider
                    y_pad = int(0.60 * (y_max - y_min))  # 55% taller

                    x_min = max(0, x_min - x_pad)
                    y_min = max(0, y_min - y_pad)
                    x_max = min(w, x_max + x_pad)
                    y_max = min(h, y_max + y_pad)

                    # Define a target size
                    TARGET_SIZE = (112, 112)  # (width, height) - A common size for lip reading

                    # Crop and save mouth region
                    if x_max > x_min and y_max > y_min:
                        mouth_roi = frame[y_min:y_max, x_min:x_max]
                        if mouth_roi.size > 0:
                            
                            # --- ADD THIS LINE ---
                            # Resize to the standard target size
                            resized_roi = cv2.resize(mouth_roi, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                            
                            gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
                            save_path = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
                            
                            # --- CHANGE THIS LINE ---
                            # Save the resized image, not the original crop
                            cv2.imwrite(save_path, gray_roi)

            frame_count += 1

    cap.release()
    return transcript_text.strip()


# --- Process each split ---
all_labels = []

for split in ["train", "val", "test"]:
    print(f"\nProcessing '{split}' split...")
    filelist_path = os.path.join(FILELIST_DIR, f"{split}.txt")

    if not os.path.exists(filelist_path):
        print(f"⚠️ Missing filelist: {filelist_path}")
        continue

    with open(filelist_path, "r") as f:
        lines = [line.strip() for line in f.read().strip().splitlines()]

    for i, line in enumerate(tqdm(lines, desc=f"Processing {split} videos")):
        if MAX_CLIPS is not None and i >= MAX_CLIPS:
            break

        # Extract video_id and clip_id
        parts = line.split('/')
        if len(parts) < 2:
            print(f"⚠️ Skipping malformed line: {line}")
            continue

        video_id, full_clip_id = parts[0], parts[1]
        clip_id = full_clip_id.split(' ')[0]  # before any space

        video_file = os.path.join(DATASET_DIR, video_id, f"{clip_id}.mp4")
        txt_file = os.path.join(DATASET_DIR, video_id, f"{clip_id}.txt")

        # Output folder for frames
        output_folder_path = os.path.join(OUTPUT_DIR, split, f"{video_id}_{clip_id}")
        os.makedirs(output_folder_path, exist_ok=True)

        if not os.path.exists(video_file):
            print(f"⚠️ Skipping missing video: {video_file}")
            continue

        # Read transcript
        transcript_text = ""
        if os.path.exists(txt_file):
            with open(txt_file, "r") as tf:
                transcript_text = tf.read().strip()

        # Process the video
        transcript_text = extract_mouth_and_get_transcript(video_file, output_folder_path, transcript_text)

        # Add to list if successful
        # Add to list if successful
        # Add to list if successful
        if transcript_text:
            cleaned_text = transcript_text

            # 1. Remove "Text: " prefix
            if cleaned_text.startswith("Text: "):
                cleaned_text = cleaned_text[len("Text: "):]

            # 2. Remove "CONF:..." suffix using regex
            # This finds "CONF:" (case-insensitive) and any whitespace before it,
            # and removes it and everything after it.
            cleaned_text = re.sub(r'\s+CONF:.*', '', cleaned_text, flags=re.IGNORECASE)

            # 3. Clean whitespace and convert to uppercase
            cleaned_text = cleaned_text.strip().upper()

            clip_path = os.path.join(split, f"{video_id}_{clip_id}")
            all_labels.append([clip_path, cleaned_text])

# --- Save final labels.csv ---
print("\nSaving labels.csv...")
df = pd.DataFrame(all_labels, columns=["clip", "transcript"])
df.to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)

print("\n✅ Preprocessing complete! Cropped regions are now wider and more natural.")
