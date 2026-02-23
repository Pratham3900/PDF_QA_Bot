from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import re
import uvicorn
import torch
import threading
import logging
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

# -------------------------------------------------------------------
# APP SETUP
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

LLM_GENERATION_TIMEOUT = int(os.getenv("LLM_GENERATION_TIMEOUT", "30"))
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-base")

# -------------------------------------------------------------------
# GLOBAL STATE (SINGLE PDF FLOW)
# -------------------------------------------------------------------
vectorstore = None
qa_ready = False

generation_tokenizer = None
generation_model = None
generation_is_encoder_decoder = False

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------------------------
# TEXT NORMALIZATION
# -------------------------------------------------------------------
def normalize_spaced_text(text: str) -> str:
    def fix(match):
        return match.group(0).replace(" ", "")
    pattern = r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b"
    return re.sub(pattern, fix, text)


def normalize_answer(text: str) -> str:
    text = normalize_spaced_text(text)
    text = re.sub(
        r"^(Answer[^:]*:|Context:|Question:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------
def load_generation_model():
    global generation_tokenizer, generation_model, generation_is_encoder_decoder

    if generation_model and generation_tokenizer:
        return generation_tokenizer, generation_model, generation_is_encoder_decoder

    config = AutoConfig.from_pretrained(HF_GENERATION_MODEL)
    generation_is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))

    generation_tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)

    if generation_is_encoder_decoder:
        generation_model = AutoModelForSeq2SeqLM.from_pretrained(HF_GENERATION_MODEL)
    else:
        generation_model = AutoModelForCausalLM.from_pretrained(HF_GENERATION_MODEL)

    if torch.cuda.is_available():
        generation_model = generation_model.to("cuda")

    generation_model.eval()
    return generation_tokenizer, generation_model, generation_is_encoder_decoder

# -------------------------------------------------------------------
# TIMEOUT GENERATION
# -------------------------------------------------------------------
class TimeoutException(Exception):
    pass


def generate_with_timeout(model, encoded, max_new_tokens, pad_token_id, timeout):
    result = {"output": None, "error": None}

    def run():
        try:
            with torch.no_grad():
                result["output"] = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutException("LLM generation timed out")

    if result["error"]:
        raise Exception(result["error"])

    return result["output"]


def generate_response(prompt: str, max_new_tokens: int) -> str:
    tokenizer, model, is_encoder_decoder = load_generation_model()
    device = next(model.parameters()).device

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    try:
        output_ids = generate_with_timeout(
            model,
            encoded,
            max_new_tokens,
            pad_token_id,
            LLM_GENERATION_TIMEOUT,
        )
    except TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Model took too long to respond.",
        )

    if is_encoder_decoder:
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    input_len = encoded["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# -------------------------------------------------------------------
# REQUEST MODELS
# -------------------------------------------------------------------
class PDFPath(BaseModel):
    filePath: str


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

    @validator("question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SummarizeRequest(BaseModel):
    pdf: str | None = None

# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------
@app.post("/process-pdf")
def process_pdf(data: PDFPath):
    global vectorstore, qa_ready

    if not os.path.exists(data.filePath):
        return {"error": "File not found."}

    loader = PyPDFLoader(data.filePath)
    raw_docs = loader.load()

    cleaned_docs = [
        Document(
            page_content=normalize_spaced_text(doc.page_content),
            metadata=doc.metadata,
        )
        for doc in raw_docs
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(cleaned_docs)

    if not chunks:
        return {"error": "No text extracted from PDF."}

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    qa_ready = True

    return {"message": "PDF processed successfully"}


@app.post("/ask")
def ask_question(data: Question):
    global vectorstore, qa_ready

    if not qa_ready or vectorstore is None:
        return {"answer": "Please upload a PDF first!"}

    docs = vectorstore.similarity_search(data.question, k=4)
    if not docs:
        return {"answer": "No relevant context found."}

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = (
        "You are a helpful assistant answering questions using ONLY the context below.\n"
        "If the answer is not present, say so briefly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {data.question}\n"
        "Answer:"
    )

    raw_answer = generate_response(prompt, max_new_tokens=256)
    return {"answer": normalize_answer(raw_answer)}


@app.post("/summarize")
def summarize_pdf(_: SummarizeRequest):
    global vectorstore, qa_ready

    if not qa_ready or vectorstore is None:
        return {"summary": "Please upload a PDF first!"}

    docs = vectorstore.similarity_search(
        "Give a concise summary of the document.",
        k=6,
    )
    if not docs:
        return {"summary": "No content available."}

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = (
        "Summarize the document in 6-8 concise bullet points.\n"
        "Use ONLY the context below.\n\n"
        f"Context:\n{context}\n\n"
        "Summary:"
    )

    raw_summary = generate_response(prompt, max_new_tokens=220)
    return {"summary": normalize_answer(raw_summary)}

# -------------------------------------------------------------------
# START SERVER
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)