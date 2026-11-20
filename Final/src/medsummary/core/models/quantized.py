"""QuantizedInstructModel: the same Qwen in GGUF (llama.cpp), quantized to Q8_0 (~int8) or Q4_K_M (~int4) (notebook
cell 92, QwenGGUF). It receives the same token ids as InstructModel (same tokenizer and chat template): only the weights
change, so generar() and the whole RAG work unchanged. On cuda or mps (Metal) every layer goes to the GPU."""
import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer

from medsummary.core import config
from medsummary.core.models.instruct import InstructModel


class QuantizedInstructModel(InstructModel):

    def __init__(self, variante: str, device: str | None = None):
        """variante: a key of models.gguf_archivos in settings.yaml ('q8_0' or 'q4_k_m')."""
        super().__init__(device)
        settings_modelos = config.settings()['models']
        self.variante = variante
        self.repo_gguf = settings_modelos['gguf_repo']
        self.archivo_gguf = settings_modelos['gguf_archivos'][variante]
        self.seed = config.settings()['seed']
        self.ruta_gguf = None
        self.paradas = set()

    def _cargar_tokenizer_y_modelo(self):
        tokenizer = AutoTokenizer.from_pretrained(self.modelo_id)          # the tokenizer comes from the normal Qwen
        self.ruta_gguf = hf_hub_download(self.repo_gguf, self.archivo_gguf)
        en_gpu = self.device in ('cuda', 'mps')
        llm = Llama(model_path=self.ruta_gguf, n_ctx=8192, n_threads=os.cpu_count(),
                    n_gpu_layers=-1 if en_gpu else 0, seed=self.seed, verbose=False)
        marcas = tokenizer.convert_tokens_to_ids(['<|im_end|>', '<|endoftext|>'])
        self.paradas = {t for t in [tokenizer.eos_token_id, *marcas] if t is not None}
        return tokenizer, llm

    def _generar_texto_nuevo(self, entradas, max_new_tokens: int) -> str:
        """Greedy decoding with llama.cpp, like do_sample=False; returns only the new text."""
        ids = entradas['input_ids'][0].tolist()
        self.model.reset()                       # no prompt cache from previous calls: same as the normal model
        nuevos = []
        for token in self.model.generate(ids, top_k=1, top_p=1.0, temp=0.0):
            if token in self.paradas:
                break
            nuevos.append(token)
            if len(nuevos) >= max_new_tokens:
                break
        return self.tokenizer.decode(nuevos, skip_special_tokens=True)

    def memoria_gb(self) -> float:
        """For GGUF, the size of the file (notebook: get_memory_footprint of QwenGGUF)."""
        return os.path.getsize(self.ruta_gguf) / 1e9

    def formato(self) -> str:
        return self.variante

    def liberar(self) -> None:
        if self.model is not None:
            self.model.close()
        super().liberar()
