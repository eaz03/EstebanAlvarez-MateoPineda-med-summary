"""Retrievers over the Corpus: keyword (naive baseline), semantic (ChromaDB), BM25, hybrid (RRF) and a cross-encoder
reranker (notebook cells 47, 49, 59, 60, 63, 64). Every retriever returns tuples (score, chunk id, text, source)."""
import re

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from medsummary.core import config
from medsummary.core.models.huggingface import HuggingFaceModel
from medsummary.core.rag.corpus import Corpus


class KeywordRetriever:
    """Naive baseline: counts how many query words appear in each chunk. Only used to compare against the others.
    Returns (score, chunk id, text) without the source, as in the notebook."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def buscar_keyword(self, consulta: str, k: int = 3) -> list[tuple]:
        palabras = set(consulta.lower().split())
        puntajes = []
        for i, ch in enumerate(self.corpus.chunks):
            texto = ch.lower()
            score = sum(1 for p in palabras if p in texto)
            puntajes.append((score, i))
        puntajes.sort(reverse=True)
        return [(s, self.corpus.ids[i], self.corpus.chunks[i]) for s, i in puntajes[:k]]


class SemanticRetriever(HuggingFaceModel):
    """bge-small embeddings in an in-memory ChromaDB with cosine distance.
    NOTE(later, FastAPI): the index is rebuilt on every start; chromadb.PersistentClient would keep it on disk."""

    def __init__(self, corpus: Corpus, device: str | None = None):
        settings = config.settings()
        super().__init__(settings['models']['embedder'], device)
        settings_retrieval = settings['retrieval']
        self.corpus = corpus
        self.query_prefix = settings_retrieval['query_prefix']             # notebook: QUERY_PREFIX
        self.nombre_coleccion = settings_retrieval['chroma_collection']
        self.embed_batch = settings_retrieval['embed_batch']
        self.lote = settings_retrieval['chroma_lote']                      # notebook: LOTE
        self.coleccion = None

    def _cargar_tokenizer_y_modelo(self):
        st = SentenceTransformer(self.modelo_id, device=self.device)        # English, reads 512 tokens
        return None, st                                                    # sentence-transformers keeps its own tokenizer

    def indexar(self):
        """Offline phase of the pipeline: embed every chunk once and store it in Chroma."""
        self.cargar()
        cliente = chromadb.Client()
        try:
            cliente.delete_collection(self.nombre_coleccion)
        except Exception:
            pass
        coleccion = cliente.create_collection(self.nombre_coleccion, metadata={'hnsw:space': 'cosine'})
        chunks = self.corpus.chunks
        embeddings = self.model.encode(chunks, batch_size=self.embed_batch, normalize_embeddings=True,
                                       show_progress_bar=True)
        for inicio in range(0, len(chunks), self.lote):
            fin = min(inicio + self.lote, len(chunks))
            coleccion.add(ids=self.corpus.ids[inicio:fin], documents=chunks[inicio:fin],
                          metadatas=self.corpus.metadatos[inicio:fin], embeddings=embeddings[inicio:fin].tolist())
        self.coleccion = coleccion
        return self

    def buscar_semantica(self, consulta: str, k: int = 3) -> list[tuple]:
        if self.coleccion is None:
            self.indexar()
        embedding_consulta = self.model.encode([self.query_prefix + consulta], normalize_embeddings=True).tolist()
        respuesta_chroma = self.coleccion.query(query_embeddings=embedding_consulta, n_results=k)
        resultados = []
        for distancia, cid, doc, metadato in zip(respuesta_chroma['distances'][0], respuesta_chroma['ids'][0],
                                                 respuesta_chroma['documents'][0], respuesta_chroma['metadatas'][0]):
            resultados.append((round(1 - distancia, 3), cid, doc, metadato['fuente']))   # similarity = 1 - distance
        return resultados

    def liberar(self) -> None:
        self.coleccion = None
        super().liberar()


class BM25Retriever:
    """Lexical search with BM25 over the chunks, ignoring stopwords from settings.yaml."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.stopwords = set(config.settings()['retrieval']['stopwords'].split())   # notebook: STOP
        self.bm25 = BM25Okapi([self._tok(ch) for ch in corpus.chunks])

    def _tok(self, texto: str) -> list[str]:
        return [w for w in re.findall(r'\w+', texto.lower()) if w not in self.stopwords]

    def buscar_bm25(self, consulta: str, k: int = 10) -> list[tuple]:
        scores = self.bm25.get_scores(self._tok(consulta))
        top = np.argsort(scores)[::-1][:k]
        return [(round(float(scores[i]), 3), self.corpus.ids[i], self.corpus.chunks[i], self.corpus.metadatos[i]['fuente'])
                for i in top]


def rrf_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. rankings: lists of ids, each already ordered from best to worst."""
    puntajes = {}
    for ranking in rankings:
        for posicion, cid in enumerate(ranking):
            puntajes[cid] = puntajes.get(cid, 0.0) + 1.0 / (k + posicion + 1)
    return sorted(puntajes, key=puntajes.get, reverse=True)


class HybridRetriever:
    """Semantic + BM25 fused with RRF. Returns score None because RRF ranks, it does not score."""

    def __init__(self, semantic: SemanticRetriever, bm25: BM25Retriever):
        self.semantic = semantic
        self.bm25 = bm25
        self.corpus = semantic.corpus
        self.rrf_k = config.settings()['retrieval']['rrf_k']

    def buscar_hibrida(self, consulta: str, k: int = 10, n: int = 20) -> list[tuple]:
        """Gets n candidates with EACH method, fuses them with RRF and keeps the top k."""
        dense_ids = [cid for _, cid, _, _ in self.semantic.buscar_semantica(consulta, n)]
        bm25_ids = [cid for _, cid, _, _ in self.bm25.buscar_bm25(consulta, n)]
        fusionado = rrf_fusion([dense_ids, bm25_ids], self.rrf_k)[:k]
        return [(None, cid, self.corpus.by_id[cid][0], self.corpus.by_id[cid][1]) for cid in fusionado]


class Reranker(HuggingFaceModel):
    """Cross-encoder that re-orders candidates from any retriever by reading query and chunk together."""

    def __init__(self, device: str | None = None):
        settings = config.settings()
        super().__init__(settings['models']['reranker'], device)                    # notebook: RERANKER_MODEL
        self.max_length = settings['retrieval']['reranker_max_length']

    def _cargar_tokenizer_y_modelo(self):
        reranker = CrossEncoder(self.modelo_id, max_length=self.max_length, device=self.device)
        return None, reranker                                                       # includes its own tokenizer

    def rerank(self, consulta: str, candidatos: list[tuple], k: int = 10) -> list[tuple]:
        self.cargar()
        pares = [[consulta, c[2]] for c in candidatos]
        scores = self.model.predict(pares)
        ordenados = sorted(zip(scores, candidatos), key=lambda t: t[0], reverse=True)
        return [(round(float(s), 3), c[1], c[2], c[3]) for s, c in ordenados[:k]]
