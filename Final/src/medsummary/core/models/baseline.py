"""BaselineModel: Flan-T5 exactly as downloaded, with no fine-tuning (notebook sections 2 and 3)."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from medsummary.core import config
from medsummary.core.models.huggingface import HuggingFaceModel


class BaselineModel(HuggingFaceModel):

    def __init__(self, device: str | None = None):
        settings = config.settings()
        super().__init__(settings['models']['summarizer'], device)            # notebook: MODELO
        settings_summarizer = settings['summarizer']
        self.max_input_len = settings_summarizer['max_input_len']             # notebook: MAX_INPUT_LEN
        self.max_target_len = settings_summarizer['max_target_len']           # notebook: MAX_TARGET_LEN
        self.num_beams = settings_summarizer['num_beams']
        self.no_repeat_ngram_size = settings_summarizer['no_repeat_ngram_size']
        self.batch = settings_summarizer['batch']                             # notebook: BATCH
        self.prompt = config.prompt('summarizer.txt')                         # notebook: PROMPT

    def _cargar_tokenizer_y_modelo(self):
        tokenizer = AutoTokenizer.from_pretrained(self.modelo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.modelo_id)
        model.to(self.device)
        model.generation_config.no_repeat_ngram_size = self.no_repeat_ngram_size   # blocks repeating any 4-token sequence
        return tokenizer, model.eval()

    def construir_prompt(self, dialogo: str) -> str:
        """notebook: PROMPT.format(dialogue=dialogo)"""
        return config.rellenar_plantilla(self.prompt, dialogue=dialogo)

    def _generar_notas(self, dialogos: list[str]) -> list[str]:
        self.cargar()
        prompts = [self.construir_prompt(dialogo) for dialogo in dialogos]
        entradas = self.tokenizer(prompts, return_tensors='pt', truncation=True,
                                  max_length=self.max_input_len, padding=True).to(self.device)
        with torch.no_grad():
            salida = self.model.generate(**entradas, max_new_tokens=self.max_target_len, num_beams=self.num_beams)
        return self.tokenizer.batch_decode(salida, skip_special_tokens=True)

    def resumir(self, dialogo: str) -> str:
        """One dialogue -> one clinical note (notebook cell 34, _generar)."""
        notas = self._generar_notas([dialogo])
        return notas[0].strip()

    def resumir_lote(self, dialogos: list[str]) -> list[str]:
        """Many dialogues in batches of self.batch, for the full validation/test sets (notebook cells 10, 16, 20)."""
        notas = []
        for inicio in range(0, len(dialogos), self.batch):
            notas.extend(self._generar_notas(dialogos[inicio:inicio + self.batch]))
        return notas
