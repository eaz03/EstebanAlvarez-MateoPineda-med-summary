"""Corpus: the medical-terminology PDF, cleaned and split into chunks that keep their source (notebook cells 43 and 45)."""
import re

import pymupdf

from medsummary.core import config


def limpiar_texto(texto: str) -> str:
    texto = texto.replace("­", "")
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", texto)          # isolated numbers that look like page numbers
    texto = re.sub(r"(?<=\w)-\n(?=\w)", "", texto)             # join words split by a hyphen at the end of a line
    texto = re.sub(r"\s*([•●▪◦])\s*", r"\n\1 ", texto)         # keep bullets at the start of a line
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)             # join the remaining single line breaks
    texto = re.sub(r"[ \t]+", " ", texto)                      # normalize spaces without removing line breaks
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def partir_en_chunks(texto: str) -> list[str]:
    texto = texto.strip()
    if not texto:
        return []
    chunks = []
    bloques = re.split(r"\n\s*\n", texto)                      # split into paragraphs
    patron_vineta = r"[•●▪◦]\s*"                                # bullets in any position
    # Medical term + pronunciation in parentheses + colon. Example: Abrasion (ă-BRĀ-zhŏn):
    patron_glosario = (
        r"(?<!\S)"
        r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’ -]*?)"
        r"\s+\("
        r"[^()\n]{2,50}"
        r"\):\s*"
    )
    for bloque in bloques:
        bloque = bloque.strip()
        if not bloque:
            continue

        if re.search(patron_vineta, bloque):                   # case 1: bulleted lists
            for parte in re.split(patron_vineta, bloque):
                parte = parte.strip()
                if parte:
                    chunks.append(parte)

        elif re.search(patron_glosario, bloque):               # case 2: glossaries without bullets
            inicios = list(re.finditer(patron_glosario, bloque))
            if inicios:
                introduccion = bloque[:inicios[0].start()].strip()
                if introduccion:                               # keep any introductory text too
                    chunks.append(introduccion)
                for i, match in enumerate(inicios):
                    inicio = match.start()
                    if i + 1 < len(inicios):
                        fin = inicios[i + 1].start()
                    else:
                        fin = len(bloque)
                    parte = bloque[inicio:fin].strip()
                    if parte:
                        chunks.append(parte)
            else:
                chunks.append(bloque)

        else:                                                  # case 3: normal text
            chunks.append(bloque)
    return chunks


def es_util(ch: str) -> bool:
    if re.search(r'https?://|doi\.org|licensed under', ch):    # references
        return False
    if len(ch) < 25 and not re.search(r'[:(]', ch):            # fragments like "complications."
        return False
    if ch[0].islower() and len(ch) < 60:                        # page-break fragments
        return False
    return True


class Corpus:
    """Pages of the PDF and their useful chunks. Each chunk keeps its source, which is what enables citing it."""

    def __init__(self, documentos: list[dict]):
        self.documentos = documentos          # notebook: corpus  ({'id': page number, 'fuente', 'texto'})
        self.chunks = []
        self.metadatos = []
        self.ids = []
        for doc in documentos:
            for j, ch in enumerate(partir_en_chunks(doc['texto'])):
                if not es_util(ch):
                    continue
                self.chunks.append(ch)
                self.metadatos.append({'fuente': doc['fuente'], 'doc_id': doc['id']})
                self.ids.append(f"{doc['id']}_chunk{j}")
        self.by_id = {}                       # chunk id -> (text, source)
        for cid, ch, metadato in zip(self.ids, self.chunks, self.metadatos):
            self.by_id[cid] = (ch, metadato['fuente'])

    @classmethod
    def desde_pdf(cls, pdf_path=None, fuente: str | None = None) -> 'Corpus':
        if pdf_path is None:
            pdf_path = config.path('corpus_pdf')            # notebook: PDF_PATH
        if fuente is None:
            fuente = config.settings()['corpus']['fuente']  # notebook: FUENTE
        if not pdf_path.is_file():
            raise FileNotFoundError(f'No existe {pdf_path}. Copia ahí Medical_terminology.pdf.')
        documentos = []
        with pymupdf.open(pdf_path) as doc:
            for numero, pagina in enumerate(doc, start=1):
                texto = limpiar_texto(pagina.get_text("text", sort=True))
                if texto:
                    documentos.append({"id": numero, "fuente": fuente, "texto": texto})
        return cls(documentos)
