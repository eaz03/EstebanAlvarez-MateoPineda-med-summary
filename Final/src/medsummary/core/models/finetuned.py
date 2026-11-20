"""FineTunedModel: BaselineModel + the LoRA adapter trained in M1. resumir() and resumir_lote() are inherited.

NOTE(efficiency): baseline and fine-tuned are two separate objects, so loading both at once keeps two copies of
Flan-T5 in memory (~1 GB in float32). Combining them in one PeftModel and running the baseline inside
model.disable_adapter() (notebook cell 34) would keep a single copy. Until then, call baseline.liberar() before
loading the fine-tuned model.
"""
from peft import PeftModel

from medsummary.core import config
from medsummary.core.models.baseline import BaselineModel


class FineTunedModel(BaselineModel):

    def __init__(self, device: str | None = None):
        super().__init__(device)
        self.adapter = config.settings()['models']['adapter_name']   # notebook: ADAPTER
        self.adapter_dir = config.path('adapter_dir')                # notebook: ADAPTER_DIR

    def _cargar_tokenizer_y_modelo(self):
        tokenizer, model = super()._cargar_tokenizer_y_modelo()      # plain Flan-T5, as in the baseline
        if not self.adapter_dir.is_dir():
            raise FileNotFoundError(f'No existe {self.adapter_dir}. Copia ahí m1_lora_adapter (M1, sección 9).')
        model = PeftModel.from_pretrained(model, str(self.adapter_dir), adapter_name=self.adapter)
        model.set_adapter(self.adapter)
        return tokenizer, model.eval()

    def norma_lora_b(self) -> float:
        """Sum of the lora_B norms. 0 means the adapter was never trained (check from notebook cell 34)."""
        self.cargar()
        norma = 0.0
        for nombre, parametro in self.model.named_parameters():
            if f'lora_B.{self.adapter}.' in nombre:
                norma += parametro.float().norm().item()
        return norma
