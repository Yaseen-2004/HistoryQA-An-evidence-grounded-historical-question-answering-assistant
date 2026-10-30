
import os
import re
import json
import nltk
import faiss
import torch
import streamlit as st
import numpy as np

from pathlib import Path
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer as NLI_Tokenizer
)
from transformers import AutoModelForSequenceClassification


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HistoryQA | Historical Research Assistant",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path("/content/drive/MyDrive/HistoryQA")

MERGED_MODEL_DIR = PROJECT_DIR / "models/merged_model"
DATA_DIR = PROJECT_DIR / "data"

FAISS_PATH = DATA_DIR / "historyqa_faiss.index"
METADATA_PATH = DATA_DIR / "historyqa_metadata.json"
CHUNKS_PATH = DATA_DIR / "historical_chunks.json"

# Standalone merged Qwen model stored in Safetensors format.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263247;
    }

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        color: #f8fafc;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 5px;
        margin-bottom: 24px;
    }

    .status-card {
        background: #111c30;
        border: 1px solid #26364f;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    .answer-card {
        background: #111c30;
        border: 1px solid #26364f;
        border-radius: 12px;
        padding: 18px;
        margin-top: 8px;
    }

    .small-muted {
        color: #9ca3af;
        font-size: 0.83rem;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #263247;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
        background: #111827;
    }

    .stButton button {
        border-radius: 9px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD ALL MODELS AND RETRIEVAL FILES
# ============================================================

@st.cache_resource(show_spinner="Loading HistoryQA models and knowledge base...")
def load_historyqa():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Run this application "
            "in a GPU-enabled environment."
        )

    # Download sentence tokenizer resources if needed.
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        str(MERGED_MODEL_DIR),
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # Load standalone merged Safetensors model in 4-bit
    # --------------------------------------------------------

    if not (MERGED_MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Merged model configuration not found: {MERGED_MODEL_DIR}"
        )

    if not (MERGED_MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Merged Safetensors weights not found: {MERGED_MODEL_DIR}"
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_MODEL_DIR),
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model.eval()
    model.config.use_cache = True

    # --------------------------------------------------------
    # Load sentence embedding model
    # --------------------------------------------------------

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda"
    )

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    index = faiss.read_index(str(FAISS_PATH))

    # --------------------------------------------------------
    # Load retrieval metadata
    # --------------------------------------------------------

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        raw_metadata = json.load(f)

    if isinstance(raw_metadata, list):
        metadata = raw_metadata

    elif isinstance(raw_metadata, dict):
        metadata = None

        for key in ["metadata", "chunks", "documents", "records"]:
            if isinstance(raw_metadata.get(key), list):
                metadata = raw_metadata[key]
                break

        if metadata is None:
            raise ValueError(
                "Could not identify the metadata list in "
                "historyqa_metadata.json."
            )
    else:
        raise ValueError("Unexpected retrieval metadata format.")

    # Fallback: use historical_chunks.json if metadata
    # does not contain the actual text.
    if metadata and not any(
        isinstance(item, dict) and
        any(k in item for k in ["text", "content", "chunk"])
        for item in metadata
    ):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        if isinstance(raw_chunks, list):
            metadata = raw_chunks
        elif isinstance(raw_chunks, dict):
            for key in ["chunks", "documents", "records"]:
                if isinstance(raw_chunks.get(key), list):
                    metadata = raw_chunks[key]
                    break

    if not isinstance(metadata, list):
        raise ValueError("Could not load historical text chunks.")

    # Normalize metadata field names.
    normalized_metadata = []

    for item in metadata:
        if not isinstance(item, dict):
            continue

        text_value = (
            item.get("text")
            or item.get("content")
            or item.get("chunk")
            or ""
        )

        title = (
            item.get("title")
            or item.get("source_title")
            or item.get("article_title")
            or "Historical source"
        )

        url = (
            item.get("url")
            or item.get("source_url")
            or item.get("source")
            or ""
        )

        normalized_metadata.append({
            "title": str(title),
            "text": str(text_value),
            "url": str(url)
        })

    if len(normalized_metadata) != index.ntotal:
        raise ValueError(
            f"FAISS index contains {index.ntotal} vectors, "
            f"but metadata contains {len(normalized_metadata)} chunks. "
            "The index and metadata must have matching order and length."
        )

    # --------------------------------------------------------
    # Load NLI verifier
    # --------------------------------------------------------

    nli_tokenizer = NLI_Tokenizer.from_pretrained(NLI_MODEL)

    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL
    ).to("cuda")

    nli_model.eval()

    label_map = {
        int(k): v.lower()
        for k, v in nli_model.config.id2label.items()
    }

    entailment_id = next(
        idx for idx, label in label_map.items()
        if "entailment" in label
    )

    contradiction_id = next(
        idx for idx, label in label_map.items()
        if "contradiction" in label
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "embedding_model": embedding_model,
        "index": index,
        "metadata": normalized_metadata,
        "nli_tokenizer": nli_tokenizer,
        "nli_model": nli_model,
        "label_map": label_map,
        "entailment_id": entailment_id,
        "contradiction_id": contradiction_id
    }


# ============================================================
# 4. RETRIEVE HISTORICAL EVIDENCE
# ============================================================

def retrieve_history(question, resources, top_k=5):

    query_embedding = resources["embedding_model"].encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = resources["index"].search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        item = resources["metadata"][int(idx)]

        results.append({
            "title": item["title"],
            "text": item["text"],
            "url": item["url"],
            "score": float(score)
        })

    return results


# ============================================================
# 5. GENERATE ANSWER USING FINE-TUNED MODEL
# ============================================================

def generate_historyqa_answer(question, retrieved, resources):

    evidence_blocks = []

    for i, item in enumerate(retrieved, 1):
        evidence_blocks.append(
            f"""
Evidence {i}
Title: {item['title']}
Content: {item['text'][:1200]}
Source URL: {item['url']}
"""
        )

    evidence = "\n".join(evidence_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are HistoryQA, an evidence-grounded historical "
                "assistant. Answer only using the supplied historical "
                "evidence. Never invent facts, dates, names, or sources. "
                "If the evidence does not contain enough information, "
                "state that the available historical evidence is "
                "insufficient. Keep your answer concise."
            )
        },
        {
            "role": "user",
            "content": (
                f"Historical Evidence:\n{evidence}\n\n"
                f"Question: {question}\n\n"
                "Answer using only the evidence above."
            )
        }
    ]

    model = resources["model"]
    tokenizer = resources["tokenizer"]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():

        outputs = model.generate(
            inputs,
            max_new_tokens=180,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs.shape[-1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer


# ============================================================
# 6. CREATE CONTEXTUAL EVIDENCE WINDOWS
# ============================================================

def create_evidence_windows(text, window_size=3):

    sentences = sent_tokenize(text)
    windows = []

    for i in range(len(sentences)):

        for size in range(1, window_size + 1):

            window = sentences[i:i + size]

            if len(window) == size:
                windows.append(" ".join(window))

    return list(dict.fromkeys(windows))


# ============================================================
# 7. RANK EVIDENCE FOR EACH ANSWER CLAIM
# ============================================================

def rank_evidence_for_answer(
    question,
    answer,
    retrieved,
    resources,
    window_size=3,
    top_n=12
):

    candidates = []

    for item in retrieved:

        windows = create_evidence_windows(
            item["text"],
            window_size=window_size
        )

        for window in windows:

            candidates.append({
                "title": item["title"],
                "url": item["url"],
                "text": window
            })

    if not candidates:
        return []

    search_text = question + " " + answer

    query_embedding = resources["embedding_model"].encode(
        [search_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    embeddings = resources["embedding_model"].encode(
        [item["text"] for item in candidates],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    semantic_scores = np.dot(
        embeddings,
        query_embedding
    )

    stopwords = {
        "the", "is", "was", "were", "a", "an", "and", "or",
        "of", "to", "in", "on", "when", "that", "with",
        "it", "also", "as", "by", "for", "this", "its"
    }

    keywords = (
        set(re.findall(r"\b[a-z0-9]+\b", answer.lower()))
        - stopwords
    )

    results = []

    for i, item in enumerate(candidates):

        passage_words = set(
            re.findall(r"\b[a-z0-9]+\b", item["text"].lower())
        )

        coverage = (
            len(keywords & passage_words) / len(keywords)
            if keywords else 0
        )

        semantic = float(semantic_scores[i])

        combined = 0.65 * semantic + 0.35 * coverage

        results.append({
            **item,
            "semantic_score": semantic,
            "keyword_coverage": coverage,
            "combined_score": combined
        })

    return sorted(
        results,
        key=lambda x: x["combined_score"],
        reverse=True
    )[:top_n]


# ============================================================
# 8. NLI SCORE CALCULATION
# ============================================================

def get_nli_scores(premise, hypothesis, resources):

    inputs = resources["nli_tokenizer"](
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to("cuda")

    with torch.inference_mode():

        logits = resources["nli_model"](**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    label_map = resources["label_map"]

    return {
        label_map[i]: float(probabilities[i])
        for i in label_map
    }


# ============================================================
# 9. CLAIM-BY-CLAIM ANSWER VERIFICATION
# ============================================================

def verify_historyqa_answer(
    question,
    answer,
    retrieved,
    resources,
    threshold=0.85,
    relevance_threshold=0.45,
    max_candidates=12
):

    clean_answer = answer.split("Source:")[0].strip()

    claims = sent_tokenize(clean_answer)

    claim_results = []

    for claim in claims:

        ranked = rank_evidence_for_answer(
            question=question,
            answer=claim,
            retrieved=retrieved,
            resources=resources,
            window_size=3,
            top_n=max_candidates
        )

        evaluated = []

        for item in ranked:

            scores = get_nli_scores(
                item["text"],
                claim,
                resources
            )

            evaluated.append({
                **item,
                "entailment": scores["entailment"],
                "contradiction": scores["contradiction"],
                "neutral": scores["neutral"]
            })

        eligible = [
            item for item in evaluated
            if item["combined_score"] >= relevance_threshold
        ]

        if eligible:

            best = max(
                eligible,
                key=lambda x: x["entailment"]
            )

            claim_supported = (
                best["entailment"] >= threshold
                and best["contradiction"] < 0.15
            )

        else:
            best = None
            claim_supported = False

        claim_results.append({
            "claim": claim,
            "supported": claim_supported,
            "best_evidence": best
        })

    all_supported = (
        len(claim_results) > 0
        and all(item["supported"] for item in claim_results)
    )

    scores = [
        item["best_evidence"]["entailment"]
        for item in claim_results
        if item["best_evidence"] is not None
    ]

    verification_score = min(scores) if scores else 0.0

    final_answer = (
        clean_answer
        if all_supported
        else (
            "The available historical evidence is insufficient "
            "to support the complete answer."
        )
    )

    return {
        "supported": all_supported,
        "verification_score": verification_score,
        "final_answer": final_answer,
        "claim_results": claim_results
    }


# ============================================================
# 10. COMPLETE QUESTION-ANSWER PIPELINE
# ============================================================

def historyqa_pipeline(question, resources):

    retrieved = retrieve_history(
        question,
        resources,
        top_k=5
    )

    if not retrieved:
        return {
            "answer": (
                "The available historical evidence is insufficient "
                "to answer this question."
            ),
            "supported": False,
            "verification_score": 0.0,
            "sources": [],
            "claim_results": []
        }

    generated_answer = generate_historyqa_answer(
        question,
        retrieved,
        resources
    )

    verification = verify_historyqa_answer(
        question,
        generated_answer,
        retrieved,
        resources
    )

    # Keep unique source references.
    sources = []
    seen = set()

    for item in retrieved:

        key = (item["title"], item["url"])

        if key not in seen:

            seen.add(key)

            sources.append({
                "title": item["title"],
                "url": item["url"],
                "score": item["score"]
            })

    return {
        "answer": verification["final_answer"],
        "generated_answer": generated_answer,
        "supported": verification["supported"],
        "verification_score": verification["verification_score"],
        "sources": sources,
        "claim_results": verification["claim_results"]
    }


# ============================================================
# 11. APPLICATION HEADER AND SIDEBAR
# ============================================================

st.markdown(
    '<div class="main-title">📜 HistoryQA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An evidence-grounded historical question-answering assistant'
    '</div>',
    unsafe_allow_html=True
)

with st.sidebar:

    st.markdown("## HistoryQA")
    st.caption("Historical Research Assistant")

    st.divider()

    st.markdown("### System architecture")

    st.markdown("""
    - **LLM:** Qwen2.5-0.5B-Instruct
    - **Fine-tuning:** LoRA / QLoRA
    - **Retrieval:** FAISS
    - **Embeddings:** MiniLM
    - **Verification:** DeBERTa NLI
    - **Knowledge base:** Wikipedia articles
    """)

    st.divider()

    st.markdown("### Verification policy")

    st.caption(
        "Generated answers are checked claim by claim against "
        "retrieved historical evidence. Unsupported complete answers "
        "are rejected."
    )

    st.divider()

    if st.button("Clear conversation", use_container_width=True):

        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("HistoryQA Prototype | Research and Educational Use")


# ============================================================
# 12. LOAD RESOURCES
# ============================================================

try:

    resources = load_historyqa()

except Exception as error:

    st.error("Unable to initialize HistoryQA.")
    st.exception(error)

    st.stop()


# ============================================================
# 13. MODEL STATUS
# ============================================================

with st.expander("System status", expanded=False):

    st.success("All models and retrieval resources loaded.")

    st.write("GPU:", torch.cuda.get_device_name(0))
    st.write("FAISS vectors:", resources["index"].ntotal)
    st.write("Historical chunks:", len(resources["metadata"]))
    st.write("Model:", "HistoryQA - Merged Qwen2.5-0.5B")
    st.write("Model format:", "FP16 Safetensors")
    st.write("Model location:", str(MERGED_MODEL_DIR))


# ============================================================
# 14. CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


def display_assistant_result(result):

    if result["supported"]:

        st.success("Evidence verification passed")

    else:

        st.warning(
            "The generated answer did not pass evidence verification. "
            "HistoryQA has withheld the unsupported answer."
        )

    st.markdown("### Answer")
    st.markdown(result["answer"])

    if result["supported"]:

        st.caption(
            "Minimum NLI entailment score (diagnostic): "
            f"{result['verification_score']:.4f}"
        )

    # Show sources.
    with st.expander(
        f"📚 Retrieved historical sources ({len(result['sources'])})"
    ):

        if not result["sources"]:
            st.write("No sources were retrieved.")

        for source in result["sources"]:

            st.markdown(f"**{source['title']}**")

            st.caption(
                f"Retrieval similarity: {source['score']:.4f}"
            )

            if source["url"].startswith(("http://", "https://")):
                st.markdown(f"[Open source]({source['url']})")

            st.divider()

    # Show claim-level verification.
    with st.expander("Claim-level verification details"):

        for i, claim in enumerate(result["claim_results"], 1):

            st.markdown(f"**Claim {i}:** {claim['claim']}")

            if claim["supported"]:
                st.success("Supported by retrieved evidence")
            else:
                st.error("Not sufficiently supported")

            evidence = claim.get("best_evidence")

            if evidence:

                st.caption(
                    "NLI entailment: "
                    f"{evidence['entailment']:.4f} | "
                    "Contradiction: "
                    f"{evidence['contradiction']:.4f}"
                )

                st.markdown(
                    f"**Evidence source:** {evidence['title']}"
                )

                st.write(evidence["text"])

            st.divider()


# Render previous messages.
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":

            st.write(message["content"])

        else:

            display_assistant_result(message["result"])


# ============================================================
# 15. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a historical question..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching historical sources, generating and verifying..."
        ):

            try:

                result = historyqa_pipeline(
                    question,
                    resources
                )

                display_assistant_result(result)

                st.session_state.messages.append({
                    "role": "assistant",
                    "result": result
                })

            except Exception as error:

                st.error("An error occurred while processing the question.")
                st.exception(error)
