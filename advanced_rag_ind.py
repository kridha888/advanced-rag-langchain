import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.chains import LLMChain
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")          # FIX 1: was os.getenv("") — empty string key
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL    = "llama3-8b-8192"

# Chunk sizes
CHILD_CHUNK   = 300       # small chunks for precise retrieval
PARENT_CHUNK  = 1200      # large chunks sent to LLM for full context
CHUNK_OVERLAP = 50

TOP_K_RETRIEVE = 10       # fetch more, re-rank to best N
TOP_K_FINAL    = 4        # final chunks after re-ranking

FAISS_DIR  = "faiss_index_adv"
BM25_CACHE = "bm25_docs.json"

@dataclass
class RagResult:
    question:         str
    rewritten_query:  str
    hyde_doc:         str           # FIX 2: was `hye_doc` (typo) — server references `hyde_doc`
    answer:           str           # FIX 3: was `ans` (typo) — server references `answer`
    sources:          List[Dict]
    scores:           List[float]
    technique_log:    List[str]
    latency_ms:       float
    hallucination_ok: bool

# FIX 4 (both prompts): missing comma after input_variables=[...],
#         and input_variables had "questions" but template used {question}
REWRITE_PROMPT = PromptTemplate(
    input_variables=["question"],   # was ["questions"]
    template="""You are an expert in re formulating user question to improve document retrieval.
    Rewrite following question to be more specific, include relevant synonyms, and remove ambiguity.
    Return only the rewritten question, nothing else.
    
    Original question: {question}
    Rewritten question:"""
)

def rewrite_query(llm: ChatGroq, question: str) -> str:
    """
    TECHNIQUE 1: Query Rewriting
    ─────────────────────────────
    Why: Vague or short queries miss relevant chunks.
    How: Ask the LLM to expand and clarify the query before retrieval.
    Gain: +15-25% retrieval recall on ambiguous queries.
    """
    chain = LLMChain(llm=llm, prompt=REWRITE_PROMPT)
    rewritten = chain.invoke({"question": question})["text"].strip()
    return rewritten

HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],   # FIX 5: was ["questions"] + missing comma
    template=""" Write a short, factual passage (2-3 sentences) that would perfectly answer this question.
    Write as if from an encyclopedia. Do NOT say "I" or "the answer is". Just write the passage.
    Original question: {question}
    Passage:"""
)

def generate_hyde_doc(llm: ChatGroq, question: str) -> str:
    """
    TECHNIQUE 2: HyDE — Hypothetical Document Embeddings
    ──────────────────────────────────────────────────────
    Why: Queries and documents live in different embedding spaces.
         Embedding a hypothetical answer is closer to real answer embeddings.
    How: Generate a fake answer → embed it → search with that embedding.
    Paper: Gao et al. 2022 "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    Gain: +10-20% precision on factoid questions.
    """
    chain = LLMChain(llm=llm, prompt=HYDE_PROMPT)
    doc = chain.invoke({"question": question})["text"].strip()
    return doc

def build_hybrid_retriever(chunks: List[Document], vectorstore: FAISS, K: int = TOP_K_RETRIEVE) -> EnsembleRetriever:
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = K

    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K}
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6]
    )


def rerank_document(query: str, docs: List[Document], top_n: int = TOP_K_FINAL) -> Tuple[List[Document], List[float]]:
    # FIX 6: `docs=List[Document]` was using `=` (default value) instead of `:` (type hint)
    _reranker = CrossEncoder(RERANK_MODEL)

    pairs = [(query, doc.page_content) for doc in docs]
    scores = _reranker.predict(pairs).tolist()

    ranked    = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs  = [d for d, _ in ranked[:top_n]]
    top_scores = [s for _, s in ranked[:top_n]]

    return top_docs, top_scores

def build_parent_child_store(docs: List[Document]) -> Tuple[List[Document], Dict[str, str], FAISS, HuggingFaceEmbeddings]:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK, chunk_overlap=CHUNK_OVERLAP
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK, chunk_overlap=CHUNK_OVERLAP
    )

    parent_chunks = parent_splitter.split_documents(docs)
    parent_store: Dict[str, str] = {}
    child_chunks: List[Document] = []

    for i, parent in enumerate(parent_chunks):
        pid = f"parent_{i}"
        parent_store[pid] = parent.page_content

        children = child_splitter.split_documents([parent])

        for child in children:
            child.metadata["parent_id"] = pid
            child.metadata["source"]    = parent.metadata.get("source", "unknown")
            child_chunks.append(child)

    vectorstore = FAISS.from_documents(child_chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)

    # Cache BM25 docs
    Path(BM25_CACHE).write_text(
        json.dumps([{"text": c.page_content, "meta": c.metadata} for c in child_chunks])
    )

    return child_chunks, parent_store, vectorstore, embeddings

def resolve_parents(child_docs: List[Document], parent_store: Dict[str, str]) -> List[Document]:
    seen_parents = set()
    parent_docs  = []
    for child in child_docs:
        pid = child.metadata.get("parent_id")
        if pid and pid not in seen_parents:
            seen_parents.add(pid)           # FIX 7: was `seen_parents.a(pid)` — `.a()` doesn't exist
            parent_docs.append(
                Document(
                    page_content=parent_store[pid],
                    metadata={**child.metadata, "resolved": "parent"}
                )
            )
    return parent_docs

GUARD_PROMPT = PromptTemplate(
    input_variables=["context", "answer"],  # FIX 8: missing comma after input_variables=[...]
    template="""You are a strict fact-checker. Given the context below and an AI-generated answer,
decide if the answer is fully supported by the context.
Reply with exactly YES or NO on the first line, then explain briefly.

Context: {context}

Answer: {answer}
"""
)

def self_rag_guard(llm: ChatGroq, context: str, answer: str) -> bool:
    chain = LLMChain(llm=llm, prompt=GUARD_PROMPT)
    verdict = chain.invoke({"context": context[:3000], "answer": answer})["text"].strip().upper()
    return verdict.startswith("YES")

ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise, helpful assistant. Answer the question using ONLY the provided context.
Cite which part of the context supports each claim. If unsure, say so.

Context:
{context}

Question: {question}

Answer (be concise and factual):"""
)

STRICT_ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""STRICT MODE: Answer ONLY what is explicitly stated in the context.
Do not infer or extrapolate. If the context does not contain the answer, say "Not found in document."

Context:
{context}

Question: {question}

Answer:"""
)

class AdvancedRAGPipeline:

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=0.2,
        )

        self.embeddings:      Optional[HuggingFaceEmbeddings] = None
        self.vectorstore:     Optional[FAISS] = None
        self.child_chunks:    List[Document] = []
        self.parent_store:    Dict[str, str] = {}
        self.hybrid_retriever = None
        self._ready = False

        if Path(FAISS_DIR).exists() and Path(BM25_CACHE).exists():
            self._load_existing()

    def ingest(self, source_path: str) -> Dict[str, Any]:
        """Full ingestion: load → parent-child split → embed → index."""
        t0 = time.time()

        path = Path(source_path)
        if path.is_dir():
            loader = DirectoryLoader(str(path), glob="**/*.{pdf,txt}", show_progress=True)
        elif path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path))
        raw_docs = loader.load()

        self.child_chunks, self.parent_store, self.vectorstore, self.embeddings = \
            build_parent_child_store(raw_docs)

        self.hybrid_retriever = build_hybrid_retriever(self.child_chunks, self.vectorstore)
        self._ready = True

        elapsed = round((time.time() - t0) * 1000)
        return {
            "status":        "ok",
            "raw_docs":      len(raw_docs),
            "parent_chunks": len(self.parent_store),
            "child_chunks":  len(self.child_chunks),
            "elapsed_ms":    elapsed
        }

    def _load_existing(self):
        """Reload from persisted index."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = FAISS.load_local(
            FAISS_DIR, self.embeddings, allow_dangerous_deserialization=True
        )
        cached = json.loads(Path(BM25_CACHE).read_text())
        self.child_chunks = [
            Document(page_content=c["text"], metadata=c["meta"]) for c in cached
        ]
        self.parent_store = {}
        for chunk in self.child_chunks:
            pid = chunk.metadata.get("parent_id")
            if pid and pid not in self.parent_store:
                self.parent_store[pid] = chunk.page_content

        self.hybrid_retriever = build_hybrid_retriever(self.child_chunks, self.vectorstore)
        self._ready = True
        print("[LOAD] Advanced RAG pipeline restored from disk ✓")

    # FIX 9: `ask` was nested INSIDE `_load_existing` due to wrong indentation
    def ask(self, question: str) -> RagResult:
        if not self._ready:
            raise RuntimeError("No documents ingested. Call .ingest() first.")

        t0  = time.time()
        log: List[str] = []

        # ── T1: Query Rewriting ──────────────────────────
        log.append("T1: Rewriting query with LLM…")
        rewritten = rewrite_query(self.llm, question)
        log.append(f"    Original : {question}")
        log.append(f"    Rewritten: {rewritten}")

        # ── T2: HyDE ─────────────────────────────────────
        log.append("T2: Generating hypothetical document (HyDE)…")
        hyde_doc = generate_hyde_doc(self.llm, rewritten)
        log.append(f"    HyDE doc : {hyde_doc[:120]}…")

        # Build a blended search query: rewritten + hyde
        search_query = f"{rewritten}\n{hyde_doc}"

        log.append("T3: Hybrid search (BM25 + FAISS fusion)…")
        retrieved_children = self.hybrid_retriever.invoke(search_query)
        log.append(f"    Retrieved {len(retrieved_children)} child chunks")

        log.append("T4: Cross-encoder re-ranking…")
        ranked_children, scores = rerank_document(rewritten, retrieved_children, TOP_K_FINAL)
        log.append(f"    Top scores: {[round(s, 3) for s in scores[:4]]}")

        log.append("T5: Resolving child chunks → parent chunks…")
        parent_docs = resolve_parents(ranked_children, self.parent_store)
        context = "\n\n---\n\n".join(d.page_content for d in parent_docs)
        log.append(f"    Resolved {len(parent_docs)} parent chunks for context")

        # ── Generate ──────────────────────────────────────
        log.append("GEN: Generating answer with Llama 3…")
        answer_chain = LLMChain(llm=self.llm, prompt=ANSWER_PROMPT)
        answer = answer_chain.invoke({"context": context, "question": question})["text"].strip()

        # ── T6: Self-RAG Guard ────────────────────────────
        log.append("T6: Self-RAG hallucination check…")
        hallucination_ok = self_rag_guard(self.llm, context, answer)
        log.append(f"    Verdict: {'✓ GROUNDED' if hallucination_ok else '✗ REGENERATING'}")

        if not hallucination_ok:
            strict_chain = LLMChain(llm=self.llm, prompt=STRICT_ANSWER_PROMPT)
            answer = strict_chain.invoke({"context": context, "question": question})["text"].strip()
            log.append("    Strict mode answer generated.")

        sources = []
        for i, doc in enumerate(ranked_children[:TOP_K_FINAL]):
            sources.append({
                "rank":      i + 1,
                "score":     round(scores[i], 3) if i < len(scores) else 0.0,
                "source":    doc.metadata.get("source", "unknown"),
                "page":      doc.metadata.get("page", "N/A"),
                "parent_id": doc.metadata.get("parent_id", ""),
                "snippet":   doc.page_content[:250] + "…",
            })

        return RagResult(
            question=question,
            rewritten_query=rewritten,
            hyde_doc=hyde_doc,
            answer=answer,
            sources=sources,
            scores=scores[:TOP_K_FINAL],
            technique_log=log,
            latency_ms=round((time.time() - t0) * 1000, 1),
            hallucination_ok=hallucination_ok,
        )