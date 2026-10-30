from pathlib import Path
import re
import shutil
import sys

APP_PATH = Path('app') / 'app.py'
if not APP_PATH.exists():
    print('ERROR: app/app.py was not found. Run this script from the HistoryQA_Complete_Project folder.')
    sys.exit(1)

source = APP_PATH.read_text(encoding='utf-8')
original = source

# 1) Remove the GPU-only quantization import and simplify Transformers imports.
source = source.replace('''from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer as NLI_Tokenizer
)
from transformers import AutoModelForSequenceClassification''', '''from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)''')

# 2) Use the local project root rather than the former Colab Drive path.
source = source.replace(
    'PROJECT_DIR = Path("/content/drive/MyDrive/HistoryQA")',
    'PROJECT_DIR = Path(__file__).resolve().parent.parent'
)

# 3) Replace the entire resource loader with a CPU-only implementation.
pattern = re.compile(
    r'@st\.cache_resource\(show_spinner="Loading HistoryQA models and knowledge base\.\.\."\)\s*\ndef load_historyqa\(\):.*?(?=\n\n# ={5,}\n# 4\. RETRIEVE HISTORICAL EVIDENCE)',
    re.S
)
replacement = '''@st.cache_resource(show_spinner="Loading HistoryQA models and knowledge base...")
def load_historyqa():

    # CPU-only configuration for Windows systems without an NVIDIA GPU.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    device = torch.device("cpu")

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    if not (MERGED_MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Merged model configuration not found: {MERGED_MODEL_DIR}"
        )
    if not (MERGED_MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Merged Safetensors weights not found: {MERGED_MODEL_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MERGED_MODEL_DIR), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the merged Qwen model in FP32 for reliable CPU inference.
    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_MODEL_DIR),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_safetensors=True
    ).to(device)
    model.eval()
    model.config.use_cache = True

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL, device="cpu"
    )

    index = faiss.read_index(str(FAISS_PATH))

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
                "Could not identify the metadata list in historyqa_metadata.json."
            )
    else:
        raise ValueError("Unexpected retrieval metadata format.")

    if metadata and not any(
        isinstance(item, dict)
        and any(k in item for k in ["text", "content", "chunk"])
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

    normalized_metadata = []
    for item in metadata:
        if not isinstance(item, dict):
            continue
        text_value = item.get("text") or item.get("content") or item.get("chunk") or ""
        title = (
            item.get("title") or item.get("source_title")
            or item.get("article_title") or "Historical source"
        )
        url = item.get("url") or item.get("source_url") or item.get("source") or ""
        normalized_metadata.append({
            "title": str(title), "text": str(text_value), "url": str(url)
        })

    if len(normalized_metadata) != index.ntotal:
        raise ValueError(
            f"FAISS index contains {index.ntotal} vectors, but metadata contains "
            f"{len(normalized_metadata)} chunks. The index and metadata must match."
        )

    # NLI model and tokenizer are also loaded on CPU.
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL
    ).to(device)
    nli_model.eval()

    label_map = {
        int(k): v.lower() for k, v in nli_model.config.id2label.items()
    }
    entailment_matches = [
        idx for idx, label in label_map.items() if "entailment" in label
    ]
    contradiction_matches = [
        idx for idx, label in label_map.items() if "contradiction" in label
    ]
    if not entailment_matches or not contradiction_matches:
        raise ValueError(f"Unexpected NLI label mapping: {label_map}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "embedding_model": embedding_model,
        "index": index,
        "metadata": normalized_metadata,
        "nli_tokenizer": nli_tokenizer,
        "nli_model": nli_model,
        "label_map": label_map,
        "entailment_id": entailment_matches[0],
        "contradiction_id": contradiction_matches[0],
        "device": device
    }
'''

match = pattern.search(source)
if not match:
    print('ERROR: Could not locate the original load_historyqa() block. No changes made.')
    print('Please ensure app/app.py is the version you pasted in the chat.')
    sys.exit(2)
source = source[:match.start()] + replacement + source[match.end():]

# 4) CPU placement for NLI token tensors.
source = source.replace(').to("cuda")', ').to("cpu")')
source = source.replace('st.write("GPU:", torch.cuda.get_device_name(0))', 'st.write("Execution Device:", "CPU")')
source = source.replace('st.write("Model format:", "FP16 Safetensors")', 'st.write("Model format:", "Safetensors loaded as FP32 for CPU inference")')

# 5) Keep the script from reporting a successful patch if GPU-only code remains.
source = source.replace('device="cuda"', 'device="cpu"')
source = source.replace('.to("cuda")', '.to("cpu")')
if 'torch.cuda.get_device_name(0)' in source or 'BitsAndBytesConfig' in source or 'quantization_config=' in source:
    print('ERROR: GPU-specific code remains after patch. No changes made.')
    sys.exit(3)

if source == original:
    print('No changes were necessary; app.py already appears patched.')
    sys.exit(0)

backup = APP_PATH.with_name('app_before_cpu_update.py')
if not backup.exists():
    shutil.copy2(APP_PATH, backup)
APP_PATH.write_text(source, encoding='utf-8')

# Syntax validation before asking Streamlit to launch.
try:
    compile(source, str(APP_PATH), 'exec')
except SyntaxError as exc:
    shutil.copy2(backup, APP_PATH)
    print(f'ERROR: Patched file failed syntax validation; original restored. {exc}')
    sys.exit(4)

print('SUCCESS: app/app.py updated for CPU inference.')
print(f'Backup: {backup}')
print('Next: python -m py_compile app/app.py')
print('Then: python -m streamlit run app/app.py')
