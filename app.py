import io
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="Dual-Layer Image–Text Verification", layout="centered")
st.title("Dual-Layer Image–Text Verification")

# -----------------------------
# Config
# -----------------------------
MLP_MODEL_PATH = "mlp_model_v2.pth"
BEST_THRESHOLD = 0.5898
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cosine similarity threshold for RDH caption matching.
# Range is [-1, 1] but CLIP text embeddings are always positive in practice,
# so the effective range is [0, 1].  0.90 = strict semantic match.
CAPTION_SIMILARITY_THRESHOLD = 0.90

# -----------------------------
# RDH config
# -----------------------------
MAGIC = "RDH1"
MAGIC_BITS = "".join(format(ord(c), "08b") for c in MAGIC)
LEN_BITS = 32  # store payload bit length in 32 bits


# =========================================================
# RDH HELPERS
# =========================================================
def text_to_bits(text: str) -> str:
    return "".join(format(ord(c), "08b") for c in text)


def bits_to_text(bits: str) -> str:
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


def int_to_bits(x: int, width: int) -> str:
    return format(x, f"0{width}b")


def bits_to_int(bits: str) -> int:
    return int(bits, 2)


def find_peak_and_zero(image_np: np.ndarray) -> tuple[int, int]:
    """
    Find the histogram peak and a suitable zero bin for histogram shifting.

    FIX: The original fallback chose the farthest bin from peak regardless of
    frequency, which could select a heavily-occupied bin and break reversibility.
    We now prioritise empty bins (freq=0), then lowest-frequency bins, breaking
    ties by distance from peak.
    """
    hist = np.bincount(image_np.flatten(), minlength=256)
    peak = int(np.argmax(hist))

    zero_candidates = np.where(hist == 0)[0]
    if len(zero_candidates) > 0:
        distances_from_peak = np.abs(zero_candidates - peak)
        zero = int(zero_candidates[np.argmax(distances_from_peak)])
    else:
        min_freq = np.min(hist)
        low_freq_bins = np.where(hist == min_freq)[0]
        distances_from_peak = np.abs(low_freq_bins - peak)
        zero = int(low_freq_bins[np.argmax(distances_from_peak)])

    return peak, zero


def build_payload_bits(message: str) -> str:
    msg_bits = text_to_bits(message)
    length_bits = int_to_bits(len(msg_bits), LEN_BITS)
    return MAGIC_BITS + length_bits + msg_bits


def can_embed(image_np: np.ndarray, payload_bits: str, peak: int) -> bool:
    capacity = int(np.sum(image_np.flatten() == peak))
    return capacity >= len(payload_bits)


# =========================================================
# RDH CORE (Histogram Shifting on Y channel)
# =========================================================
def embed_data(image_np: np.ndarray, message: str) -> np.ndarray:
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    payload_bits = build_payload_bits(message)
    peak, zero = find_peak_and_zero(image_np)
    flat = image_np.flatten().copy()

    if not can_embed(image_np, payload_bits, peak):
        raise ValueError("Message too large for embedding capacity of this image.")

    if zero > peak:
        one_value = peak + 1
        if zero <= one_value:
            raise ValueError("No room between peak and zero bin; cannot embed safely.")
        mask = (flat > peak) & (flat < zero)
        flat[mask] += 1
    else:
        one_value = peak - 1
        if zero >= one_value:
            raise ValueError("No room between peak and zero bin; cannot embed safely.")
        mask = (flat < peak) & (flat > zero)
        flat[mask] -= 1

    bit_idx = 0
    for i in range(len(flat)):
        if flat[i] == peak:
            if bit_idx >= len(payload_bits):
                break
            if payload_bits[bit_idx] == "1":
                flat[i] = one_value
            bit_idx += 1

    return flat.reshape(image_np.shape).astype(np.uint8)


def extract_data(image_np: np.ndarray) -> str:
    """
    Extract a message previously embedded with embed_data.

    FIX: The original code re-ran find_peak_and_zero on the stego image, whose
    histogram changes after embedding — causing the wrong one_value and garbled
    bits.  We now try the top-10 most frequent bins as peak candidates in both
    ±1 directions and validate each attempt against the MAGIC header.
    """
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    flat = image_np.flatten()
    hist = np.bincount(flat, minlength=256)
    peak_candidates = np.argsort(hist)[::-1][:10]
    needed_prefix = len(MAGIC_BITS) + LEN_BITS

    for peak in peak_candidates:
        for one_value in [int(peak) + 1, int(peak) - 1]:
            if one_value < 0 or one_value > 255:
                continue

            bits: list[str] = []
            payload_len = None
            total_needed = None
            found = False

            for val in flat:
                if val == peak:
                    bits.append("0")
                elif val == one_value:
                    bits.append("1")
                else:
                    continue

                bitstr = "".join(bits)

                if payload_len is None and len(bitstr) >= needed_prefix:
                    magic = bitstr[: len(MAGIC_BITS)]
                    if magic != MAGIC_BITS:
                        break
                    length_part = bitstr[len(MAGIC_BITS) : needed_prefix]
                    payload_len = bits_to_int(length_part)
                    if payload_len <= 0 or payload_len > 100_000:
                        break
                    total_needed = needed_prefix + payload_len

                if total_needed is not None and len(bitstr) >= total_needed:
                    msg_bits = bitstr[needed_prefix:total_needed]
                    found = True
                    break

            if found:
                return bits_to_text(msg_bits)

    return ""


# =========================================================
# COLOR RDH WRAPPERS
# =========================================================
def embed_rdh(image_pil: Image.Image, caption: str) -> Image.Image:
    existing, _ = extract_rdh(image_pil)
    if existing:
        raise ValueError(
            "This image already contains an embedded caption. "
            "Re-embedding would corrupt the data."
        )

    image_ycbcr = image_pil.convert("YCbCr")
    y, cb, cr = image_ycbcr.split()
    y_np = np.array(y, dtype=np.uint8)
    y_embedded = embed_data(y_np, caption)
    y_img = Image.fromarray(y_embedded, mode="L")
    return Image.merge("YCbCr", (y_img, cb, cr)).convert("RGB")


def extract_rdh(image_pil: Image.Image) -> tuple[str, str]:
    try:
        image_ycbcr = image_pil.convert("YCbCr")
        y, _, _ = image_ycbcr.split()
        y_np = np.array(y, dtype=np.uint8)
        extracted = extract_data(y_np)
        if extracted == "":
            return "", "NoData"
        return extracted, "Safe"
    except Exception as exc:
        return "", f"Error: {exc}"


# =========================================================
# MLP MODEL
# =========================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1025, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    mlp_model = MLP().to(DEVICE)
    mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=DEVICE))
    mlp_model.eval()

    return clip_model, clip_processor, mlp_model


clip_model, clip_processor, mlp_model = load_models()


# =========================================================
# FEATURE BUILDER
# =========================================================
def build_features(v_img: np.ndarray, v_txt: np.ndarray) -> np.ndarray:
    diff = np.abs(v_img - v_txt)                                        # (1, 512)
    prod = v_img * v_txt                                                 # (1, 512)
    v_img_norm = v_img / (np.linalg.norm(v_img, axis=1, keepdims=True) + 1e-10)
    v_txt_norm = v_txt / (np.linalg.norm(v_txt, axis=1, keepdims=True) + 1e-10)
    cos_sim = np.sum(v_img_norm * v_txt_norm, axis=1, keepdims=True)    # (1, 1)
    return np.hstack([diff, prod, cos_sim]).astype(np.float32)          # (1, 1025)


# =========================================================
# SEMANTIC TEXT SIMILARITY  (reuses the already-loaded CLIP model)
# =========================================================
def compute_text_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two captions via CLIP text embeddings.

    Encodes each caption separately (matching the pattern used in predict()) so
    the processor always produces the exact input structure the model expects.
    Uses clip_model(**inputs).text_embeds — a plain tensor guaranteed by the
    CLIPModel forward pass — instead of get_text_features(), which can return a
    BaseModelOutputWithPooling object on some transformers versions.
    """
    def encode(text: str) -> torch.Tensor:
        # Pass a dummy image so CLIPProcessor produces a complete input dict.
        # We only use the text_embeds output, so the image content is irrelevant.
        dummy_image = Image.new("RGB", (224, 224))
        inputs = clip_processor(
            images=dummy_image,
            text=[text.strip()],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        # text_embeds is always a plain (1, 512) tensor from CLIPModel.forward()
        emb = outputs.text_embeds  # shape (1, 512)
        return emb / (emb.norm(dim=-1, keepdim=True) + 1e-10)

    emb_a = encode(text_a)  # (1, 512), unit norm
    emb_b = encode(text_b)  # (1, 512), unit norm

    # Dot product of two unit vectors == cosine similarity
    similarity = (emb_a[0] @ emb_b[0]).item()
    return float(similarity)


# =========================================================
# IMAGE–CAPTION PREDICTION
# =========================================================
def predict(image: Image.Image, caption: str) -> float:
    caption = caption.strip()
    if not caption:
        raise ValueError("Caption must not be empty.")

    image = image.convert("RGB")

    inputs = clip_processor(
        images=image,
        text=[caption],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    v_img = outputs.image_embeds.detach().cpu().numpy()
    v_txt = outputs.text_embeds.detach().cpu().numpy()

    features = build_features(v_img, v_txt)
    x = torch.tensor(features, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        prob = mlp_model(x).item()

    return float(prob)


# =========================================================
# DOWNLOAD HELPER
# =========================================================
def pil_image_to_png_bytes(image: Image.Image) -> bytes:
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        st.error(f"Failed to encode image for download: {exc}")
        return b""


# =========================================================
# UI
# =========================================================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
user_caption = st.text_input("Enter Caption")

if uploaded_file is not None:
    uploaded_file.seek(0)
    preview_image = Image.open(uploaded_file).convert("RGB")
    st.image(preview_image, caption="Uploaded Image", use_container_width=True)


# =========================================================
# SESSION STATE
# =========================================================
if "embedded_image" not in st.session_state:
    st.session_state.embedded_image = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded_name:
        st.session_state.embedded_image = None
        st.session_state.last_uploaded_name = uploaded_file.name


# =========================================================
# FLOW
# =========================================================
if st.button("Verify"):
    if uploaded_file is None or user_caption.strip() == "":
        st.warning("Please provide both an image and a caption.")
    else:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        extracted_caption, status = extract_rdh(image)

        # ── Extraction error ──────────────────────────────────────────────────
        if status.startswith("Error"):
            st.error(f"RDH extraction failed: {status}")

        # ── CASE 1: No embedded caption ───────────────────────────────────────
        elif extracted_caption == "":
            st.warning("No embedded caption found in the image.")

            try:
                prob = predict(image, user_caption)
            except Exception as exc:
                st.error(f"CLIP prediction failed: {exc}")
                prob = None

            if prob is not None:
                st.write("CLIP + MLP Score:", round(prob, 4))

                if prob >= BEST_THRESHOLD:
                    st.success("Caption matches image well enough. Embedding now…")
                    try:
                        embedded_img = embed_rdh(image, user_caption)
                        st.session_state.embedded_image = embedded_img
                    except Exception as exc:
                        st.session_state.embedded_image = None
                        st.error(f"Embedding failed: {exc}")
                else:
                    st.error("Caption does not match image well enough. Embedding not allowed.")
                    st.session_state.embedded_image = None

        # ── CASE 2 & 3: Embedded caption found — semantic similarity gate ─────
        else:
            try:
                similarity = compute_text_similarity(extracted_caption, user_caption)
            except Exception as exc:
                st.error(f"Similarity computation failed: {exc}")
                similarity = None

            if similarity is not None:
                st.write("Extracted Caption:", extracted_caption)
                st.write(
                    "Caption Similarity Score:",
                    round(similarity, 4),
                    f"(threshold ≥ {CAPTION_SIMILARITY_THRESHOLD})",
                )

                # ── CASE 2: Captions are semantically too different ───────────
                if similarity < CAPTION_SIMILARITY_THRESHOLD:
                    st.error(
                        f"Caption Mismatch (RDH Layer) — "
                        f"similarity {round(similarity, 4)} is below "
                        f"the required threshold of {CAPTION_SIMILARITY_THRESHOLD}."
                    )

                # ── CASE 3: RDH verified — proceed to CLIP + MLP ─────────────
                else:
                    st.success(
                        f"RDH Verified — similarity {round(similarity, 4)} "
                        f">= {CAPTION_SIMILARITY_THRESHOLD}"
                    )

                    try:
                        prob = predict(image, user_caption)
                    except Exception as exc:
                        st.error(f"CLIP prediction failed: {exc}")
                        prob = None

                    if prob is not None:
                        st.write("CLIP + MLP Score:", round(prob, 4))

                        if prob >= BEST_THRESHOLD:
                            st.success("Final Verification Passed")
                        else:
                            st.error("Semantic Mismatch Detected")


# =========================================================
# SHOW EMBED RESULT
# =========================================================
if st.session_state.embedded_image is not None:
    st.subheader("Embedded Output")
    st.image(
        st.session_state.embedded_image,
        caption="Caption Embedded Image",
        use_container_width=True,
    )
    png_bytes = pil_image_to_png_bytes(st.session_state.embedded_image)
    if png_bytes:
        st.download_button(
            label="Download Embedded Image",
            data=png_bytes,
            file_name="embedded_image.png",
            mime="image/png",
        )