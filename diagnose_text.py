"""
Diagnostic script for CLIP text-only query pipeline.
Checks checkpoint loading, embedding spaces, and similarity scores.
"""
import os, sys, torch, numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _PROJECT_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# 1. Load ClipFeatureExtractor (watch the loading messages carefully)
print("=" * 60)
print("STEP 1: Loading ClipFeatureExtractor")
print("=" * 60)
from clip_inference import ClipFeatureExtractor

extractor = ClipFeatureExtractor(
    device=device,
    image_ckpt_path=os.path.join(_PROJECT_ROOT, "clip_models", "best_model_image.pth"),
    text_ckpt_path=os.path.join(_PROJECT_ROOT, "clip_models", "best_model_text.pth"),
)

# 2. Check what got loaded
print("\n" + "=" * 60)
print("STEP 2: Checkpoint loading status")
print("=" * 60)
print(f"  image_ckpt_loaded : {extractor.image_ckpt_loaded}")
print(f"  text_ckpt_loaded  : {extractor.text_ckpt_loaded}")
print(f"  n_ctx (prompt tokens): {extractor.n_ctx}")
print(f"  ctx is None?      : {extractor.ctx is None}")
if extractor.ctx is not None:
    print(f"  ctx shape         : {extractor.ctx.shape}")
    print(f"  ctx dtype         : {extractor.ctx.dtype}")
    print(f"  ctx device        : {extractor.ctx.device}")
print(f"  image_size_for_image_model: {extractor.image_size_for_image_model}")
print(f"  image_size_for_text_model : {extractor.image_size_for_text_model}")

# 3. Check text_clip_model pos embed
print(f"\n  text_clip_model visual pos embed shape: {extractor.text_clip_model.visual.positional_embedding.shape}")
print(f"  image_clip_model visual pos embed shape: {extractor.image_clip_model.visual.positional_embedding.shape}")

# 4. Check model dtype
model_dtype = getattr(extractor.text_clip_model, "dtype", extractor.text_clip_model.text_projection.dtype)
print(f"  text_clip_model dtype: {model_dtype}")

# 5. Test text embedding
print("\n" + "=" * 60)
print("STEP 3: Text embedding extraction")
print("=" * 60)
test_queries = [
    "A man wearing a red shirt and blue jeans",
    "A woman in a black dress carrying a handbag",
    "A person with a white t-shirt and shorts",
]
for q in test_queries:
    feat = extractor.extract_text_embedding(q)
    if feat is None:
        print(f"  FAILED for: '{q}'")
    else:
        feat_flat = feat.flatten()
        print(f"  '{q[:50]}...'")
        print(f"    shape={feat.shape}, norm={np.linalg.norm(feat_flat):.4f}")
        print(f"    mean={feat_flat.mean():.6f}, std={feat_flat.std():.6f}")
        print(f"    min={feat_flat.min():.6f}, max={feat_flat.max():.6f}")

# 6. Check if text embeddings are discriminative (different queries should give different embeddings)
print("\n" + "=" * 60)
print("STEP 4: Text embedding discrimination")
print("=" * 60)
feats = []
for q in test_queries:
    f = extractor.extract_text_embedding(q)
    if f is not None:
        feats.append(f.flatten())

if len(feats) >= 2:
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            sim = np.dot(feats[i], feats[j])
            print(f"  sim(q{i}, q{j}) = {sim:.4f}")
    print("  (Good: sims should be < 0.95, showing discrimination)")
else:
    print("  Not enough embeddings to compare")

# 7. Test image embedding for text query (using a dummy image if no real one available)
print("\n" + "=" * 60)
print("STEP 5: Image embedding (for text query) test")
print("=" * 60)

import cv2
# Create a dummy person-like image (solid color, 256x128)
dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

img_feat_text_space = extractor.extract_image_embedding_for_text_query(dummy_img)
img_feat_image_space = extractor.extract_image_embedding(dummy_img)

if img_feat_text_space is not None and img_feat_image_space is not None:
    itf = img_feat_text_space.flatten()
    iif = img_feat_image_space.flatten()
    print(f"  Image feat (text model): shape={img_feat_text_space.shape}, norm={np.linalg.norm(itf):.4f}")
    print(f"    mean={itf.mean():.6f}, std={itf.std():.6f}")
    print(f"  Image feat (image model): shape={img_feat_image_space.shape}, norm={np.linalg.norm(iif):.4f}")
    print(f"    mean={iif.mean():.6f}, std={iif.std():.6f}")
    
    # Cross-space similarity (should be low/random - different spaces)
    cross_sim = np.dot(itf, iif)
    print(f"  Cross-space sim (text_model_img vs image_model_img): {cross_sim:.4f}")
    
    # Text-to-image similarity in text model's space
    if len(feats) > 0:
        text_img_sim = np.dot(feats[0], itf)
        print(f"  Text-to-Image sim (text query vs dummy img, text space): {text_img_sim:.4f}")
        print(f"  (Expected: ~0.05-0.20 for random/unrelated content)")
else:
    print(f"  text_space feat: {'OK' if img_feat_text_space is not None else 'FAILED'}")
    print(f"  image_space feat: {'OK' if img_feat_image_space is not None else 'FAILED'}")

# 8. Check raw checkpoint keys
print("\n" + "=" * 60)
print("STEP 6: Raw checkpoint key analysis")
print("=" * 60)
text_ckpt_path = os.path.join(_PROJECT_ROOT, "clip_models", "best_model_text.pth")
ckpt = torch.load(text_ckpt_path, map_location="cpu", weights_only=False)
if isinstance(ckpt, dict):
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        print(f"  Checkpoint has 'state_dict' key with {len(sd)} entries")
    else:
        tensor_keys = [k for k, v in ckpt.items() if isinstance(v, torch.Tensor)]
        non_tensor_keys = [k for k in ckpt if k not in tensor_keys]
        print(f"  Checkpoint is flat dict: {len(tensor_keys)} tensor keys, {len(non_tensor_keys)} non-tensor keys")
        if non_tensor_keys:
            print(f"  Non-tensor keys: {non_tensor_keys}")
        
        # Check for ctx
        if "ctx" in ckpt:
            print(f"  ctx shape: {ckpt['ctx'].shape}, dtype: {ckpt['ctx'].dtype}")
        else:
            print(f"  WARNING: 'ctx' NOT FOUND in checkpoint!")
        
        # Show first 20 keys
        print(f"\n  First 20 keys:")
        for k in sorted(tensor_keys)[:20]:
            print(f"    {k}: {ckpt[k].shape}")
        
        # Check how many match CLIP model
        clip_sd = extractor.text_clip_model.state_dict()
        matched = sum(1 for k in tensor_keys if k in clip_sd)
        mismatched_shape = sum(1 for k in tensor_keys if k in clip_sd and clip_sd[k].shape != ckpt[k].shape)
        not_in_clip = [k for k in tensor_keys if k not in clip_sd and not any(
            k.startswith(p) for p in ("cross_attn", "mlm_head", "img_bn", "txt_bn", "img_classifier", "txt_classifier", "ctx")
        )]
        print(f"\n  Keys matching CLIP model: {matched}/{len(tensor_keys)}")
        print(f"  Shape mismatches: {mismatched_shape}")
        if not_in_clip:
            print(f"  Unmatched non-IRRA keys: {not_in_clip[:10]}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
