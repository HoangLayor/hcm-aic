import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
)

MODEL_ID = "Qwen/Qwen3.5-2B"
VIDEO_FPS = 1.0
MAX_VIDEO_FRAMES = 32
MAX_VIDEO_PIXELS = 8_388_608  # Total pixel budget across sampled frames.

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.video_processor.max_frames = MAX_VIDEO_FRAMES
processor.video_processor.size = {
    "shortest_edge": 4_096,
    "longest_edge": MAX_VIDEO_PIXELS,
}


def ask_qwen_video(
    prompt,
    video_path,
    fps=VIDEO_FPS,
    max_tokens=128,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "path": video_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        fps=fps,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return output_text[0]

prompt = """
Analyze this short news video clip and generate one concise English caption.

Describe only visually supported information.

Focus on:
- people
- actions
- objects
- scene or location
- visible text
- important visual events

Do not infer names, locations, or events unless they are clearly supported by the video.
"""

from pathlib import Path
import json

root = Path("/kaggle/input/datasets/dotrantu/aic-10-video/Segment_Video")
output_path = Path("/kaggle/working/captions.json")

results = []

for i, file in enumerate(sorted(root.glob("*/*.mp4")), 1):
    print(f"[{i}] Processing: {file}")

    try:
        caption = ask_qwen_video(
            prompt=prompt,
            video_path=str(file),
            fps=VIDEO_FPS,
            max_tokens=256,
        )

        relative_path = file.relative_to(root).as_posix()

        results.append({
            "video_path": relative_path,
            "caption": caption
        })

        print(caption)

    except Exception as e:
        print(f"ERROR: {file}")
        print(e)

# lưu JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Saved to: {output_path}")

import json
import torch
from sentence_transformers import SentenceTransformer


CAPTION_FILE = "/kaggle/working/captions.json"
OUTPUT_JSON = "/kaggle/working/caption_with_embedding.json"
OUTPUT_EMBEDDING = "/kaggle/working/caption_embeddings.pt"


# =========================
# 1. Load model
# =========================

model = SentenceTransformer(
    "Qwen/Qwen3-VL-Embedding-2B"
)


# =========================
# 2. Load captions
# =========================

with open(CAPTION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


captions = [
    item["caption"].strip()
    for item in data
]


print(f"Number of captions: {len(captions)}")


# =========================
# 3. Encode captions
# =========================

embeddings = model.encode(
    captions,
    batch_size=8,
    show_progress_bar=True,
    convert_to_tensor=True,
)

print("Embedding shape:", embeddings.shape)


embeddings = embeddings.cpu()


torch.save(
    embeddings,
    OUTPUT_EMBEDDING,
)


# =========================
# 6. Add embedding index
# =========================

for index, item in enumerate(data):
    item["embedding_index"] = index


# =========================
# 7. Save updated metadata
# =========================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2,
    )


print(f"Saved embeddings to: {OUTPUT_EMBEDDING}")
print(f"Saved metadata to: {OUTPUT_JSON}")