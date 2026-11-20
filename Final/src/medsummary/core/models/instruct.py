"""InstructModel: a causal chat LLM, Qwen2.5-1.5B-Instruct (notebook: gen_model, gen_tok). Judge inherits from it."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from medsummary.core import config
from medsummary.core.models.hardware import DTYPES
from medsummary.core.models.huggingface import HuggingFaceModel


class InstructModel(HuggingFaceModel):
    seccion_settings = 'instruct'   # reads models.instruct and the instruct: block of settings.yaml

    def __init__(self, device: str | None = None):
        settings = config.settings()
        super().__init__(settings['models'][self.seccion_settings], device)   # notebook: GEN_MODEL
        settings_modelo = settings[self.seccion_settings]
        self.dtype = DTYPES[settings_modelo['dtype']]
        self.max_new_tokens = settings_modelo['max_new_tokens']

    def _argumentos_de_carga(self) -> dict:
        return {'torch_dtype': self.dtype}

    def _cargar_tokenizer_y_modelo(self):
        tokenizer = AutoTokenizer.from_pretrained(self.modelo_id)
        model = AutoModelForCausalLM.from_pretrained(self.modelo_id, **self._argumentos_de_carga())
        model.to(self.device)
        return tokenizer, model.eval()

    def _generar_texto_nuevo(self, entradas, max_new_tokens: int) -> str:
        """Greedy decoding; returns only the text generated after the prompt."""
        with torch.inference_mode():
            salida = self.model.generate(**entradas, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        tokens_nuevos = salida[0][entradas['input_ids'].shape[1]:]
        return self.tokenizer.decode(tokens_nuevos, skip_special_tokens=True)

    def generar(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        """System + user turn (notebook cell 53): chat template as text, then tokenized."""
        self.cargar()
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        entradas = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        return self._generar_texto_nuevo(entradas, max_new_tokens).strip()

    def generar_desde_mensajes(self, mensajes: list[dict], max_new_tokens: int | None = None) -> str:
        """A whole message list tokenized by the chat template (notebook cells 33 and 85). Not stripped."""
        self.cargar()
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        entradas = self.tokenizer.apply_chat_template(mensajes, add_generation_prompt=True, return_dict=True,
                                                      return_tensors='pt').to(self.device)
        return self._generar_texto_nuevo(entradas, max_new_tokens)

    def memoria_gb(self) -> float:
        """Size of the weights in memory (notebook: get_memory_footprint)."""
        return self.model.get_memory_footprint() / 1e9

    def formato(self) -> str:
        """Precision of the weights, e.g. 'torch.bfloat16'."""
        return str(self.model.dtype)
