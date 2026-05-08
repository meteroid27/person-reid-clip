import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

try:
    import clip
except Exception as e:
    raise RuntimeError(
        "CLIP package is required for CLIP inference. Install with: pip install git+https://github.com/openai/CLIP.git"
    ) from e


class ClipFeatureExtractor:
    """
    CLIP-based feature extractor that properly loads fine-tuned checkpoints
    trained with IRRA (text model) and CLIPReID_Prompt (image model).

    Key fixes over the previous version:
    1. Positional embeddings are interpolated (not silently skipped) when shapes differ.
    2. Image and text checkpoints are loaded selectively to avoid overwriting each other.
    3. IRRA's learned prompt tokens (ctx) are restored for text encoding.
    """

    def __init__(self, device, image_ckpt_path=None, text_ckpt_path=None, clip_backbone="ViT-B/16"):
        self.device = device
        self.clip_backbone = clip_backbone

        # --- Load TWO separate CLIP model instances to avoid checkpoint conflicts ---
        # Image model: used for encode_image
        # Text model:  used for encode_text (with learned prompt tokens from IRRA)
        self.image_clip_model, _ = clip.load(clip_backbone, device=self.device, jit=False)
        self.image_clip_model.eval()

        self.text_clip_model, _ = clip.load(clip_backbone, device=self.device, jit=False)
        self.text_clip_model.eval()

        # IRRA prompt tokens (loaded from text checkpoint if available)
        self.ctx = None       # nn.Parameter or None
        self.n_ctx = 0        # number of prompt tokens

        # Track which models are loaded
        self.image_ckpt_loaded = False
        self.text_ckpt_loaded = False

        self.image_ckpt_path = image_ckpt_path
        self.text_ckpt_path = text_ckpt_path

        # Load checkpoints into their respective models
        if image_ckpt_path:
            self._load_image_checkpoint(image_ckpt_path)
        else:
            print("CLIP image checkpoint not provided; using base CLIP weights for image encoding.")

        if text_ckpt_path:
            self._load_text_checkpoint(text_ckpt_path)
        else:
            print("CLIP text checkpoint not provided; using base CLIP weights for text encoding.")

        # Determine image sizes from the loaded models
        self.image_size_for_image_model = self._infer_image_size(self.image_clip_model)
        self.image_size_for_text_model = self._infer_image_size(self.text_clip_model)

        # Primary image transform (used by extract_image_embedding)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size_for_image_model),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    # -----------------------------------------------------------------
    # Positional Embedding Interpolation
    # -----------------------------------------------------------------

    @staticmethod
    def _interpolate_pos_embed(model_pos_embed, ckpt_pos_embed):
        """
        Interpolate checkpoint positional embedding to match model's expected shape.
        Both are [num_tokens, dim] where token 0 is CLS.

        Returns the interpolated embedding matching model_pos_embed's shape.
        """
        if model_pos_embed.shape == ckpt_pos_embed.shape:
            return ckpt_pos_embed

        model_len = model_pos_embed.shape[0]
        ckpt_len = ckpt_pos_embed.shape[0]
        dim = ckpt_pos_embed.shape[1]

        # Separate CLS token
        ckpt_cls = ckpt_pos_embed[:1]
        ckpt_spatial = ckpt_pos_embed[1:]

        # Infer source grid size
        ckpt_grid = int((ckpt_len - 1) ** 0.5)
        if ckpt_grid * ckpt_grid == ckpt_len - 1:
            # Square grid (e.g., 14x14 = 196 patches → 197 tokens)
            src_h, src_w = ckpt_grid, ckpt_grid
        else:
            # Rectangular grid (e.g., 16x8 = 128 patches → 129 tokens)
            # Try common ReID aspect ratios
            for h, w in [(16, 8), (8, 16), (32, 16), (16, 32)]:
                if h * w == ckpt_len - 1:
                    src_h, src_w = h, w
                    break
            else:
                print(f"  WARNING: Cannot determine grid shape for {ckpt_len-1} patches. Skipping pos embed.")
                return model_pos_embed

        # Infer target grid size
        model_grid = int((model_len - 1) ** 0.5)
        if model_grid * model_grid == model_len - 1:
            tgt_h, tgt_w = model_grid, model_grid
        else:
            for h, w in [(16, 8), (8, 16), (32, 16), (16, 32)]:
                if h * w == model_len - 1:
                    tgt_h, tgt_w = h, w
                    break
            else:
                print(f"  WARNING: Cannot determine target grid shape for {model_len-1} patches. Skipping pos embed.")
                return model_pos_embed

        print(f"  Interpolating positional embedding: [{src_h}x{src_w}] -> [{tgt_h}x{tgt_w}]")

        # Reshape to spatial, interpolate, reshape back
        spatial = ckpt_spatial.reshape(1, src_h, src_w, dim).permute(0, 3, 1, 2).float()
        spatial = F.interpolate(spatial, size=(tgt_h, tgt_w), mode='bicubic', align_corners=False)
        spatial = spatial.permute(0, 2, 3, 1).reshape(-1, dim)

        result = torch.cat([ckpt_cls, spatial], dim=0)

        # Match dtype
        if result.dtype != model_pos_embed.dtype:
            result = result.to(model_pos_embed.dtype)

        return result

    # -----------------------------------------------------------------
    # Image Size Inference
    # -----------------------------------------------------------------

    def _infer_image_size(self, model):
        """
        Choose input size that matches the visual positional embedding tokens.
        - 197 tokens -> 14x14 grid -> 224x224
        - 129 tokens -> 16x8 grid -> 256x128
        """
        try:
            pos_len = int(model.visual.positional_embedding.shape[0])
        except Exception:
            pos_len = None

        if pos_len == 129:
            print(f"CLIP visual positional length=129 detected -> resize (256, 128).")
            return (256, 128)

        default_res = getattr(model.visual, "input_resolution", 224)
        if isinstance(default_res, int):
            print(f"Using CLIP default square resize ({default_res}, {default_res}).")
            return (default_res, default_res)

        print("Using fallback CLIP resize (224, 224).")
        return (224, 224)

    # -----------------------------------------------------------------
    # Checkpoint Loading Utilities
    # -----------------------------------------------------------------

    def _read_state_dict(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            model_like = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
            if model_like:
                return model_like
        if hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
        return None

    def _candidate_key_variants(self, key):
        """Generate possible key mappings from checkpoint key to base CLIP key."""
        return [
            key,
            key.replace("module.", ""),
            key.replace("clip_model.", ""),
            key.replace("model.", ""),
            key.replace("image_encoder.", "visual."),
        ]

    # -----------------------------------------------------------------
    # Image Checkpoint Loading (CLIPReID_Prompt)
    # -----------------------------------------------------------------

    def _load_image_checkpoint(self, ckpt_path):
        """
        Load image model checkpoint (CLIPReID_Prompt) into self.image_clip_model.

        CLIPReID_Prompt saves keys like:
          image_encoder.* -> maps to visual.*
          bottleneck.*, classifier.*, prompt_learner.*, text_encoder.* -> ignored (not in base CLIP)
        """
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"CLIP image checkpoint not found: {ckpt_path}")

        state_dict = self._read_state_dict(ckpt_path)
        if not state_dict:
            raise RuntimeError(f"CLIP image checkpoint has no readable state_dict: {ckpt_path}")

        print(f"\n{'='*60}")
        print(f"Loading IMAGE checkpoint: {os.path.basename(ckpt_path)}")
        print(f"  Checkpoint keys: {len(state_dict)}")

        model_dict = self.image_clip_model.state_dict()
        filtered = {}
        pos_embed_handled = False

        for source_key, source_value in state_dict.items():
            for candidate in self._candidate_key_variants(source_key):
                if candidate in model_dict:
                    # Special handling for positional embedding shape mismatch
                    if "positional_embedding" in candidate and candidate.startswith("visual"):
                        if model_dict[candidate].shape != source_value.shape:
                            print(f"  Pos embed shape mismatch: ckpt={source_value.shape} vs model={model_dict[candidate].shape}")
                            interpolated = self._interpolate_pos_embed(model_dict[candidate], source_value)
                            # Now interpolate the model's pos embed to match ckpt's trained size
                            # We need to replace the model's pos embed entirely
                            self._replace_visual_pos_embed(self.image_clip_model, source_value)
                            pos_embed_handled = True
                            break
                        else:
                            filtered[candidate] = source_value
                            break
                    elif model_dict[candidate].shape == source_value.shape:
                        filtered[candidate] = source_value
                        break

        if not filtered and not pos_embed_handled:
            print(f"  WARNING: No compatible keys matched. Using base CLIP weights.")
            return

        # Re-capture model_dict if pos embed was replaced (shape changed)
        if pos_embed_handled:
            model_dict = self.image_clip_model.state_dict()

        model_dict.update(filtered)
        self.image_clip_model.load_state_dict(model_dict)
        self.image_ckpt_loaded = True
        print(f"  Matched {len(filtered)} layers" + (" + positional embedding (replaced)" if pos_embed_handled else ""))
        print(f"{'='*60}\n")

    # -----------------------------------------------------------------
    # Text Checkpoint Loading (IRRA)
    # -----------------------------------------------------------------

    def _load_text_checkpoint(self, ckpt_path):
        """
        Load text model checkpoint (IRRA) into self.text_clip_model.

        IRRA saves keys like:
          visual.* -> CLIP visual encoder (fine-tuned end-to-end)
          token_embedding.*, transformer.*, ln_final.*, text_projection, positional_embedding
          ctx -> learnable prompt tokens (critical for text encoding!)
          cross_attn.*, mlm_head.*, img_bn.*, txt_bn.*, *_classifier.* -> IRRA-specific (ignored)
        """
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"CLIP text checkpoint not found: {ckpt_path}")

        state_dict = self._read_state_dict(ckpt_path)
        if not state_dict:
            raise RuntimeError(f"CLIP text checkpoint has no readable state_dict: {ckpt_path}")

        print(f"\n{'='*60}")
        print(f"Loading TEXT checkpoint: {os.path.basename(ckpt_path)}")
        print(f"  Checkpoint keys: {len(state_dict)}")

        model_dict = self.text_clip_model.state_dict()
        filtered = {}
        pos_embed_handled = False

        # Extract IRRA-specific parameters
        if "ctx" in state_dict:
            ctx_tensor = state_dict["ctx"]
            self.ctx = ctx_tensor.to(self.device)
            self.n_ctx = ctx_tensor.shape[0]
            print(f"  Found learned prompt tokens (ctx): {self.n_ctx} tokens, dim={ctx_tensor.shape[1]}")

        for source_key, source_value in state_dict.items():
            # Skip IRRA-specific keys that don't exist in base CLIP
            if source_key in ("ctx",):
                continue
            skip_prefixes = ("cross_attn", "mlm_head", "img_bn", "txt_bn", "img_classifier", "txt_classifier")
            if any(source_key.startswith(p) for p in skip_prefixes):
                continue

            for candidate in self._candidate_key_variants(source_key):
                if candidate in model_dict:
                    # Special handling for positional embedding shape mismatch
                    if "positional_embedding" in candidate:
                        if model_dict[candidate].shape != source_value.shape:
                            if candidate.startswith("visual"):
                                print(f"  Visual pos embed shape mismatch: ckpt={source_value.shape} vs model={model_dict[candidate].shape}")
                                self._replace_visual_pos_embed(self.text_clip_model, source_value)
                                pos_embed_handled = True
                            else:
                                # Text positional embedding -- should match
                                print(f"  Text pos embed shape mismatch: ckpt={source_value.shape} vs model={model_dict[candidate].shape}")
                                # Text pos embed is usually the same size, but if not, skip
                            break
                        else:
                            filtered[candidate] = source_value
                            break
                    elif model_dict[candidate].shape == source_value.shape:
                        filtered[candidate] = source_value
                        break

        if not filtered and not pos_embed_handled:
            print(f"  WARNING: No compatible keys matched. Using base CLIP weights.")
            return

        # Re-capture model_dict if pos embed was replaced (shape changed)
        if pos_embed_handled:
            model_dict = self.text_clip_model.state_dict()

        model_dict.update(filtered)
        self.text_clip_model.load_state_dict(model_dict)
        self.text_ckpt_loaded = True
        print(f"  Matched {len(filtered)} layers" + (" + visual positional embedding (replaced)" if pos_embed_handled else ""))

        if self.n_ctx > 0:
            print(f"  [OK] IRRA prompt tokens loaded ({self.n_ctx} tokens) -- will be used for text encoding")
        else:
            print(f"  [WARN] No IRRA prompt tokens found -- text encoding uses standard CLIP path")

        print(f"{'='*60}\n")

    # -----------------------------------------------------------------
    # Positional Embedding Replacement
    # -----------------------------------------------------------------

    @staticmethod
    def _replace_visual_pos_embed(model, new_pos_embed):
        """
        Replace the visual positional embedding in a CLIP model.
        This changes the model's expected input resolution.
        """
        new_pos = new_pos_embed.to(
            device=model.visual.positional_embedding.device,
            dtype=model.visual.positional_embedding.dtype
        )
        model.visual.positional_embedding = nn.Parameter(new_pos)
        num_patches = new_pos.shape[0] - 1  # subtract CLS token
        print(f"  Replaced visual positional embedding: {num_patches} patches")

    # -----------------------------------------------------------------
    # Feature Extraction: Image
    # -----------------------------------------------------------------

    def extract_image_embedding(self, img_numpy):
        """
        Extract image embedding using the image CLIP model.
        Uses the image checkpoint's fine-tuned visual encoder.
        """
        try:
            img_rgb = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
            image_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.image_clip_model.encode_image(image_tensor)
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            return feat.detach().cpu().numpy()
        except Exception as e:
            print(f"CLIP image extraction error: {e}")
            # Fallback: try with 224x224
            try:
                fallback_transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ])
                img_rgb = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
                image_tensor = fallback_transform(img_rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.image_clip_model.encode_image(image_tensor)
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
                print("CLIP fallback extraction succeeded with resize (224, 224).")
                return feat.detach().cpu().numpy()
            except Exception as e2:
                print(f"CLIP fallback image extraction failed: {e2}")
                return None

    def extract_image_embedding_for_text_query(self, img_numpy):
        """
        Extract image embedding using the TEXT CLIP model's visual encoder (IRRA).

        CRITICAL: When doing text-to-image retrieval, gallery images MUST be
        encoded by the same model that encoded the text query (IRRA). The IRRA
        model co-trained both visual and text encoders, so their embeddings are
        in the same space. Using the image_clip_model's encoder (CLIPReID_Prompt)
        would produce embeddings in a different space, leading to ~0.1 similarities.
        """
        try:
            img_rgb = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
            # Use text model's image size (IRRA trained with 256x128)
            text_model_transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.image_size_for_text_model),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            image_tensor = text_model_transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.text_clip_model.encode_image(image_tensor)
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            return feat.detach().cpu().numpy()
        except Exception as e:
            print(f"CLIP text-model image extraction error: {e}")
            return None

    # -----------------------------------------------------------------
    # Feature Extraction: Text (with IRRA prompt tokens)
    # -----------------------------------------------------------------

    def extract_text_embedding(self, text_query):
        """
        Extract text embedding using the text CLIP model.

        If IRRA prompt tokens (ctx) were loaded, they are prepended to the
        token embeddings before passing through the text transformer.
        This matches IRRA's _encode_text() at training time.
        """
        try:
            text_tokens = clip.tokenize([str(text_query)], truncate=True).to(self.device)

            if self.n_ctx > 0 and self.ctx is not None:
                # IRRA-style text encoding with learned prompt tokens
                feat = self._encode_text_with_prompts(text_tokens)
            else:
                # Standard CLIP text encoding
                with torch.no_grad():
                    feat = self.text_clip_model.encode_text(text_tokens)

            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            return feat.detach().cpu().numpy()
        except Exception as e:
            print(f"CLIP text extraction error: {e}")
            return None

    def _encode_text_with_prompts(self, tokens):
        """
        Reproduce IRRA's _encode_text() at inference time.

        IRRA prepends n_ctx learned prompt tokens after the SOS token:
          [SOS] [ctx_1] ... [ctx_n] [word_1] ... [word_k] [EOT] [PAD...]

        The EOT position shifts by n_ctx, which must be accounted for
        when extracting the global text feature.
        """
        model = self.text_clip_model
        B = tokens.shape[0]
        model_dtype = getattr(model, "dtype", model.text_projection.dtype)

        with torch.no_grad():
            # Get token embeddings
            x = model.token_embedding(tokens).to(dtype=model_dtype)  # [B, L, D]

            # Prepend prompt tokens after SOS
            ctx = self.ctx.unsqueeze(0).expand(B, -1, -1).to(device=x.device, dtype=model_dtype)  # [B, n_ctx, D]
            # [SOS, ctx_1..ctx_n, word_1..word_(L-1-n_ctx)]
            # We drop the last n_ctx tokens to keep sequence length the same
            x = torch.cat([x[:, :1], ctx, x[:, 1:-self.n_ctx]], dim=1)

            # Add positional embedding
            x = x + model.positional_embedding.to(dtype=model_dtype)

            # Transformer expects [seq_len, batch, dim]
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)  # back to [B, L, D]

            # Layer norm
            x = model.ln_final(x).to(dtype=model_dtype)

            # Extract global feature at shifted EOT position
            eot_idx = tokens.argmax(dim=-1)
            shifted_eot = torch.clamp(eot_idx + self.n_ctx, max=x.shape[1] - 1)
            text_projection = model.text_projection.to(dtype=model_dtype)
            global_feat = x[torch.arange(B, device=x.device), shifted_eot] @ text_projection

        return F.normalize(global_feat, dim=-1)
