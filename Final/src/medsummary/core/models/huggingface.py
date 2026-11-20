"""HuggingFaceModel: parent of every model class. Holds tokenizer + model, the device, loading and freeing. Not runnable alone."""
from medsummary.core import config
from medsummary.core.models.hardware import detectar_dispositivo, liberar_memoria


class HuggingFaceModel:

    def __init__(self, modelo_id: str, device: str | None = None):
        self.modelo_id = modelo_id
        if device is None:
            device = detectar_dispositivo(config.settings()['device'])
        self.device = device
        self.tokenizer = None
        self.model = None

    def _cargar_tokenizer_y_modelo(self):
        """Each subclass returns (tokenizer, model) with the model already on self.device."""
        raise NotImplementedError

    @property
    def esta_cargado(self) -> bool:
        return self.model is not None

    def cargar(self):
        """Downloads/loads the weights the first time; later calls do nothing. Returns self."""
        if not self.esta_cargado:
            self.tokenizer, self.model = self._cargar_tokenizer_y_modelo()
        return self

    def liberar(self) -> None:
        """Drops the weights and returns their GPU memory (notebook: liberar('model', ...))."""
        self.tokenizer = None
        self.model = None
        liberar_memoria()
