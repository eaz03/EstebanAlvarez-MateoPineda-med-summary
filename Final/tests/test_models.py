"""Checks the model classes against the notebook without downloading weights (torch/transformers/peft are faked).
Run: python tests/test_models.py"""
import json, os, sys, tempfile, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src'))

# ---------- fakes ----------
estado = {'cuda': False, 'mps': True}; llamadas = []
torch = types.ModuleType('torch'); torch.float16, torch.bfloat16, torch.float32 = 'f16', 'bf16', 'f32'
torch.cuda = types.SimpleNamespace(is_available=lambda: estado['cuda'], empty_cache=lambda: llamadas.append('cuda_empty'))
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: estado['mps']))
torch.mps = types.SimpleNamespace(empty_cache=lambda: llamadas.append('mps_empty'))
class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): return False
torch.no_grad = torch.inference_mode = _NoGrad
class Entradas(dict):
    def to(self, d): llamadas.append(('inputs_to', d)); return self
class FakeTokenizer:
    def __call__(self, prompts, **kw): llamadas.append(('tokenize', prompts, kw)); return Entradas(n=len(prompts))
    def batch_decode(self, out, skip_special_tokens): return [f'  nota {i}  ' for i in range(out)]
class FakeModel:
    def __init__(self): self.generation_config = types.SimpleNamespace()
    def to(self, d): llamadas.append(('model_to', d)); return self
    def eval(self): return self
    def generate(self, **kw): llamadas.append(('generate', {k: v for k, v in kw.items() if k != 'n'})); return kw['n']
tr = types.ModuleType('transformers'); tr.set_seed = lambda s: None
tr.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda m: FakeTokenizer())
tr.AutoModelForSeq2SeqLM = types.SimpleNamespace(from_pretrained=lambda m: llamadas.append(('load', m)) or FakeModel())
tr.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda m, **kw: FakeModel())
peft = types.ModuleType('peft')
class PeftModel(FakeModel):
    @classmethod
    def from_pretrained(cls, base, ruta, adapter_name): llamadas.append(('peft', ruta, adapter_name)); return cls()
    def set_adapter(self, n): llamadas.append(('set_adapter', n))
peft.PeftModel = PeftModel
sys.modules.update(torch=torch, transformers=tr, peft=peft)

from medsummary.core import config
from medsummary.core.models import BaselineModel, FineTunedModel, InstructModel, Judge, detectar_dispositivo

nb = json.load(open('notebooks/Entrega3_TopicosIA_EAZ_MPA.ipynb'))
cell = lambda i: ''.join(nb['cells'][i]['source'])
fallos = []
def ok(nombre, condicion):
    print(('OK   ' if condicion else 'FAIL ') + nombre)
    if not condicion: fallos.append(nombre)

# ---------- device ----------
ok('MPS fallback env set', os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1')
ok('auto -> mps', detectar_dispositivo() == 'mps'); estado['cuda'] = True
ok('auto -> cuda first', detectar_dispositivo() == 'cuda'); estado.update(cuda=False, mps=False)
ok('auto -> cpu', detectar_dispositivo() == 'cpu'); estado['mps'] = True

# ---------- notebook originals ----------
g = {'PROMPT': 'Summarize the following doctor-patient dialogue into a clinical note:\n\n{dialogue}'}
ok('PROMPT literal matches cell 7', "PROMPT = 'Summarize the following doctor-patient dialogue into a clinical note:\\n\\n{dialogue}'" in cell(7))
c33 = cell(33)
exec(c33[c33.index('PLANTILLA_FLOW ='):c33.index('def liberar_juez_anterior')], g)
exec(c33[c33.index('def construir_prompt_flow'):c33.index('def parsear_flow')], g)
exec(cell(84), g)

# ---------- BaselineModel (runs alone, no adapter needed) ----------
dialogo = 'Doctor: Hi\nPatient: My {head} hurts [OUTPUT]?'
baseline = BaselineModel()
ok('baseline prompt == PROMPT.format', baseline.construir_prompt(dialogo) == g['PROMPT'].format(dialogue=dialogo))
llamadas.clear(); nota = baseline.resumir(dialogo)
ok('baseline.resumir runs and strips', nota == 'nota 0')
ok('baseline loads plain Flan-T5 only', ('load', 'google/flan-t5-base') in llamadas and not any(c[0] == 'peft' for c in llamadas if isinstance(c, tuple)))
gen = [c[1] for c in llamadas if isinstance(c, tuple) and c[0] == 'generate'][0]
ok('generate params = cell 34 (MAX_TARGET_LEN, num_beams=4)', gen == {'max_new_tokens': 200, 'num_beams': 4})
tok = [c for c in llamadas if isinstance(c, tuple) and c[0] == 'tokenize'][0][2]
ok('tokenize params (MAX_INPUT_LEN=512, truncation)', tok == {'return_tensors': 'pt', 'truncation': True, 'max_length': 512, 'padding': True})
ok('no_repeat_ngram_size=4', baseline.model.generation_config.no_repeat_ngram_size == 4)
ok('model on detected device (mps)', ('model_to', 'mps') in llamadas)
llamadas.clear(); notas = baseline.resumir_lote([dialogo] * 19)
ok('resumir_lote: 19 dialogues -> batches 8,8,3', [c[2] if False else len(c[1]) for c in llamadas if isinstance(c, tuple) and c[0] == 'tokenize'] == [8, 8, 3] and len(notas) == 19)
llamadas.clear(); baseline.liberar()
ok('liberar frees weights + mps cache', not baseline.esta_cargado and 'mps_empty' in llamadas)

# ---------- FineTunedModel (inherits resumir) ----------
ok('FineTunedModel inherits resumir from BaselineModel', FineTunedModel.resumir is BaselineModel.resumir)
try:
    FineTunedModel().resumir(dialogo); ok('missing adapter raises FileNotFoundError', False)
except FileNotFoundError:
    ok('missing adapter raises FileNotFoundError', True)
with tempfile.TemporaryDirectory() as carpeta:
    finetuned = FineTunedModel(); finetuned.adapter_dir = Path(carpeta)
    llamadas.clear(); finetuned.resumir(dialogo)
    ok('fine-tuned = Flan-T5 + PeftModel adapter m1', ('load', 'google/flan-t5-base') in llamadas and ('peft', carpeta, 'm1') in llamadas and ('set_adapter', 'm1') in llamadas)

# ---------- InstructModel / Judge ----------
judge, instruct = Judge(), InstructModel()
ok('model ids', (judge.modelo_id, instruct.modelo_id) == ('flowaicom/Flow-Judge-v0.1', 'Qwen/Qwen2.5-1.5B-Instruct'))
ok('judge load args (float16 kept, sdpa)', judge._argumentos_de_carga() == {'torch_dtype': 'f16', 'low_cpu_mem_usage': True, 'attn_implementation': 'sdpa'})
ok('instruct load args (auto)', instruct._argumentos_de_carga() == {'torch_dtype': 'auto'})
ok('max_new_tokens judge=192, ragas=512, instruct=220', (judge.max_new_tokens, judge.max_new_tokens_ragas, instruct.max_new_tokens) == (192, 512, 220))
nota = 'The patient reports headache.'
for juez, rubrica in judge.rubricas.items():
    entradas = {'conversation': dialogo} if rubrica['usa_dialogo'] else None
    ok(f'flow prompt {juez} == cell 33', judge.construir_prompt_flow(entradas, nota, rubrica['criterio'], rubrica['rubrica'])
       == g['construir_prompt_flow'](dialogo, nota, g['RUBRICAS_FLOW'][juez]))
def original_85(entradas, salida, metrica):
    s = '\n# INPUT\nBelow are the inputs required for performing the task:\n<inputs>\n'
    for e, t in entradas.items(): s += f'<{e}>\n{t}\n</{e}>\n'
    s += '</inputs>\n'
    p = g['PLANTILLA_FLOW'].replace('[SECCION_INPUT]', s).replace('[OUTPUT]', salida)
    return p.replace('[CRITERIO]', g['RUBRICAS_RAGAS'][metrica]['criterio']).replace('[RUBRICA]', g['RUBRICAS_RAGAS'][metrica]['rubrica'])
for metrica, rubrica in judge.rubricas_ragas.items():
    entradas = {'original_note': nota, 'retrieved_passages': 'p1\n\np2'}
    ok(f'ragas prompt {metrica} == cell 85', judge.construir_prompt_flow(entradas, 'x', rubrica['criterio'], rubrica['rubrica']) == original_85(entradas, 'x', metrica))
ok('parsear_flow valid', Judge.parsear_flow('<feedback> good </feedback>\n<score>4</score>') == {'score': 4, 'explanation': 'good', 'valido': True})
r = Judge.parsear_flow('garbage'); ok('parsear_flow invalid -> 3, flagged', r['score'] == 3 and not r['valido'])
msg = 'Input should be 0 or 1'
ok('ragas retry text == cell 85', config.rellenar_plantilla(judge.mensaje_reintento_ragas, error=msg) == (f'Your answer was not valid ({msg}). Write the feedback inside '
   '<feedback></feedback> tags, then 0 or 1 inside <score></score> tags.'))
puntajes = iter([2, 5, 5]); judge.generar_desde_mensajes = lambda m, n=None: f'<feedback>x</feedback><score>{next(puntajes)}</score>'
final, _ = judge.juez_completo(dialogo, nota)
ok('juez_completo: grounding 2 caps final at 2.0', final == round(min(0.45 * 2 + 0.35 * 5 + 0.20 * 5, 2.0), 2))
respuestas = iter(['<feedback>a</feedback><score>7</score>', '<feedback>b</feedback><score>1</score>'])
judge.generar_desde_mensajes = lambda m, n=None: next(respuestas)
ok('juzgar_binario retries once after invalid score', judge.juzgar_binario({'clinical_draft': nota}, 'x', 'context_recall') == (1, 'b'))

# ---------- QuantizedInstructModel (GGUF, cell 92 QwenGGUF) with a fake llama.cpp ----------
import tempfile as _tempfile
eventos = []
class FakeLlama:
    def __init__(self, model_path, n_ctx, n_threads, n_gpu_layers, seed, verbose): eventos.append(('llama', n_gpu_layers, seed))
    def reset(self): eventos.append('reset')
    def generate(self, ids, top_k, top_p, temp):
        eventos.append(('generate', list(ids), top_k, temp))
        yield from [5, 6, 7, 99, 8]                       # 99 = <|im_end|>: generation must stop there
    def close(self): eventos.append('close')
archivo_gguf = _tempfile.NamedTemporaryFile(delete=False); archivo_gguf.write(b'x' * 2000); archivo_gguf.close()
llama_cpp = types.ModuleType('llama_cpp'); llama_cpp.Llama = FakeLlama
hub = types.ModuleType('huggingface_hub'); hub.hf_hub_download = lambda repo, archivo: archivo_gguf.name
sys.modules.update(llama_cpp=llama_cpp, huggingface_hub=hub)
class FakeTensorIds:
    def __getitem__(self, i): return types.SimpleNamespace(tolist=lambda: [1, 2, 3])
class FakeQwenTokenizer:
    eos_token_id = 98
    def convert_tokens_to_ids(self, tokens): return [99, None]
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt): return 'PROMPT'
    def __call__(self, prompt, return_tensors): return Entradas(input_ids=FakeTensorIds())
    def decode(self, ids, skip_special_tokens): return ' '.join(map(str, ids))
from medsummary.core.models import quantized
quantized.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda m: FakeQwenTokenizer())
estado.update(cuda=False, mps=True)
q4 = quantized.QuantizedInstructModel('q4_k_m')
ok('GGUF: Qwen tokenizer + q4_k_m file from settings', (q4.modelo_id, q4.archivo_gguf) == ('Qwen/Qwen2.5-1.5B-Instruct', 'qwen2.5-1.5b-instruct-q4_k_m.gguf'))
ok('GGUF: stops at <|im_end|>, returns only new tokens', q4.generar('sys', 'user') == '5 6 7')
ok('GGUF: all layers on the GPU (mps), seed 42, greedy', ('llama', -1, 42) in eventos and ('generate', [1, 2, 3], 1, 0.0) in eventos and 'reset' in eventos)
ok('GGUF: respects max_new_tokens', q4.generar('sys', 'user', max_new_tokens=2) == '5 6')
ok('GGUF: memory = file size, format = variant', abs(q4.memoria_gb() - 2e-6) < 1e-12 and q4.formato() == 'q4_k_m')
q4.liberar(); ok('GGUF: liberar closes llama.cpp', 'close' in eventos and not q4.esta_cargado)

print('\nALL PASSED' if not fallos else f'\n{len(fallos)} FAILED: {fallos}')
sys.exit(1 if fallos else 0)
