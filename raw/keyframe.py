from pathlib import Path

import torch



SOURCE_ROOT = Path('/kaggle/input/models/xndung/aic2026/pytorch/default/1')

INPUT_DIR = SOURCE_ROOT / 'data' / 'clips'

OUTPUT_DIR = Path('/kaggle/working/output_dinov2')

EMBEDDING_DIR = Path('/kaggle/working/keyframe_embeddings_qwen2b')

if not INPUT_DIR.is_dir():

    raise FileNotFoundError(f'Input not found: {INPUT_DIR}')

if not torch.cuda.is_available():

    raise RuntimeError('Enable Kaggle GPU Accelerator and restart the session.')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

print('GPU:', torch.cuda.get_device_name(0))

print('Input:', INPUT_DIR)

print('Output:', OUTPUT_DIR)

# Kaggle includes CUDA-enabled torch. Do not reinstall torch.

!pip install -q -U opencv-python-headless numpy 'sentence-transformers>=5.4.0' 'transformers>=4.57.0' 'qwen-vl-utils>=0.0.14'

import json

import cv2

import numpy as np

import torch.nn.functional as F



DINO_MODEL_ID = 'dinov2_vitb14'

SIMILARITY_THRESHOLD = 0.65

DINO_BATCH_SIZE = 16

DINO_IMAGE_SIZE = 224

MAX_CANDIDATES = 10

RESUME = True



def read_frames(path):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened(): raise ValueError(f'Cannot open {path}')

    frames = []

    while True:

        ok, frame = cap.read()

        if not ok: break

        frames.append(frame)

    cap.release()

    if not frames: raise ValueError(f'No frames in {path}')

    return frames



def completed_count(groups_path, clip_path):

    if groups_path.is_file():

        saved = json.loads(groups_path.read_text(encoding='utf-8'))

        if 'clip_num_frames' in saved: return int(saved['clip_num_frames'])

    cap = cv2.VideoCapture(str(clip_path)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()

    if count <= 0: raise ValueError(f'Cannot count {clip_path}')

    return count



print('Loading DINOv2 Base...')

dino = torch.hub.load('facebookresearch/dinov2', DINO_MODEL_ID).eval().to('cuda')

mean = torch.tensor((0.485, 0.456, 0.406), device='cuda').view(3,1,1)

std = torch.tensor((0.229, 0.224, 0.225), device='cuda').view(3,1,1)



def encode_dino(frames):

    output = []

    for start in range(0, len(frames), DINO_BATCH_SIZE):

        tensors = []

        for frame in frames[start:start+DINO_BATCH_SIZE]:

            rgb = cv2.cvtColor(cv2.resize(frame, (DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)), cv2.COLOR_BGR2RGB)

            tensors.append(torch.from_numpy(rgb).permute(2,0,1).float() / 255.0)

        batch = (torch.stack(tensors).to('cuda') - mean) / std

        with torch.inference_mode(): output.append(F.normalize(dino(batch), dim=1).cpu())

        print(f'  DINO: {min(start+DINO_BATCH_SIZE, len(frames))}/{len(frames)}')

    return torch.cat(output).numpy()



def group_and_select(embeddings):

    scores = np.sum(embeddings[:-1]*embeddings[1:], axis=1).astype(float).tolist() if len(embeddings)>1 else []

    groups, current = [], [0]

    for index, score in enumerate(scores, start=1):

        if score < SIMILARITY_THRESHOLD: groups.append(current); current = [index]

        else: current.append(index)

    groups.append(current)

    keyframes = []

    for group in groups:

        positions = np.linspace(0, len(group)-1, min(MAX_CANDIDATES, len(group)), dtype=int)

        candidates = [group[pos] for pos in positions]

        if len(candidates) == 1: keyframes.append(candidates[0]); continue

        vectors = embeddings[candidates]; matrix = vectors @ vectors.T

        keyframes.append(candidates[int(np.argmax((matrix.sum(1)-1)/(len(candidates)-1)))])

    return groups, keyframes, scores

def process_clip(clip_path, video_id, video_output_dir, global_offset):

    frames = read_frames(clip_path)

    groups, keyframes, scores = group_and_select(encode_dino(frames))

    clip_dir = video_output_dir / clip_path.stem

    keyframe_dir = clip_dir / 'keyframes'

    keyframe_dir.mkdir(parents=True, exist_ok=True)

    for local_index in keyframes:

        global_index = global_offset + local_index

        if not cv2.imwrite(str(keyframe_dir / f'keyframe_frame_{global_index:08d}.jpg'), frames[local_index]):

            raise IOError(f'Could not save keyframe in {keyframe_dir}')

    data = {

        'video_id': video_id, 'clip_id': clip_path.stem, 'clip_num_frames': len(frames),

        'global_frame_offset': global_offset, 'grouping_method': 'dinov2',

        'keyframe_selection_method': 'dinov2_cosine_medoid', 'dino_model': DINO_MODEL_ID,

        'similarity_threshold': SIMILARITY_THRESHOLD, 'n_groups': len(groups),

        'groups': [

            {'group_id': i, 'start_frame': int(g[0]), 'end_frame': int(g[-1]),

             'keyframe': int(keyframes[i-1]), 'global_start_frame': global_offset+int(g[0]),

             'global_end_frame': global_offset+int(g[-1]),

             'global_keyframe': global_offset+int(keyframes[i-1]), 'num_frames': len(g)}

            for i, g in enumerate(groups, start=1)],

        'similarity_values': [float(x) for x in scores]}

    (clip_dir / 'groups.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'  {clip_path.stem}: {len(groups)} groups, {len(keyframes)} keyframes')

    return len(frames)



for video_dir in sorted(path for path in INPUT_DIR.iterdir() if path.is_dir()):

    clips = sorted(video_dir.glob('*.mp4'))

    video_output = OUTPUT_DIR / video_dir.name

    video_output.mkdir(parents=True, exist_ok=True)

    offset = 0

    print(f'\
VIDEO {video_dir.name}: {len(clips)} clips')

    for position, clip in enumerate(clips, start=1):

        groups_path = video_output / clip.stem / 'groups.json'

        if RESUME and groups_path.is_file():

            count = completed_count(groups_path, clip)

            print(f'[{position}/{len(clips)}] Skip {clip.stem}')

        else:

            print(f'[{position}/{len(clips)}] Start {clip.stem}, global offset={offset}')

            count = process_clip(clip, video_dir.name, video_output, offset)

        offset += count

print('DINO output saved in:', OUTPUT_DIR)

# Free DINOv2 GPU memory before loading Qwen.

del dino

torch.cuda.empty_cache()

import re

from PIL import Image

from sentence_transformers import SentenceTransformer



QWEN_MODEL_ID = 'Qwen/Qwen3-VL-Embedding-2B'

QWEN_BATCH_SIZE = 8

MAX_IMAGE_SIDE = 512

RESIZED_DIR = Path('/kaggle/working/keyframes_for_qwen_512')

RESIZED_DIR.mkdir(parents=True, exist_ok=True)

keyframes = sorted(OUTPUT_DIR.rglob('keyframes/*.jpg'))

if not keyframes: raise FileNotFoundError(f'No keyframes below {OUTPUT_DIR}')



def global_id(path):

    match = re.search(r'(\\d+)$', path.stem)

    if not match: raise ValueError(f'Invalid keyframe filename: {path.name}')

    return int(match.group(1))



keyframes.sort(key=lambda path: (path.parent.parent.parent.name, path.parent.parent.name, global_id(path)))

metadata, prepared = [], []

for embedding_index, image_path in enumerate(keyframes):

    clip_dir = image_path.parent.parent; video_dir = clip_dir.parent; frame_id = global_id(image_path)

    groups = json.loads((clip_dir/'groups.json').read_text(encoding='utf-8'))['groups']

    group = next(item for item in groups if item['global_keyframe'] == frame_id)

    relative_path = image_path.relative_to(OUTPUT_DIR)

    metadata.append({'embedding_index': embedding_index, 'video_id': video_dir.name, 'clip_id': clip_dir.name,

                     'group_id': group['group_id'], 'local_frame_index': group['keyframe'],

                     'global_frame_index': group['global_keyframe'], 'relative_image_path': str(relative_path)})

    resized_path = RESIZED_DIR / relative_path; resized_path.parent.mkdir(parents=True, exist_ok=True)

    if not resized_path.is_file():

        with Image.open(image_path) as image:

            image = image.convert('RGB'); image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE)); image.save(resized_path, quality=90)

    prepared.append(resized_path)



qwen = SentenceTransformer(QWEN_MODEL_ID, device='cuda'); qwen.half()

print(f'Qwen embedding {len(prepared)} keyframes on CUDA')

vectors = qwen.encode([str(path) for path in prepared], batch_size=QWEN_BATCH_SIZE,

                      convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

torch.save(torch.from_numpy(vectors.astype(np.float32)).cpu(), EMBEDDING_DIR/'embeddings.pt')

payload = {'model_id': QWEN_MODEL_ID, 'embedding_dimension': int(vectors.shape[1]),

           'normalized': True, 'num_keyframes': len(metadata), 'items': metadata}

(EMBEDDING_DIR/'metadata.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

print('Embedding shape:', vectors.shape)

print('Saved:', EMBEDDING_DIR/'embeddings.pt')

print('Saved:', EMBEDDING_DIR/'metadata.json')