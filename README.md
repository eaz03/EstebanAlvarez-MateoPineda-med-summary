Tópicos Especiales y Aplicaciones en Inteligencia Artificial - SI4006 
# Proyecto Semestral
## Esteban Álvarez Zuluaga, Mateo Pineda Álvarez

Sistema que toma la conversación entre un médico y su paciente y genera un borrador de la historia clínica, para ahorrarle tiempo de documentación al personal de salud.

## Contenido
- **Entrega 1:** carpeta con toda la información relativa a la entrega 1. Lo pedido por la entrega se encuentra en [README_M1](Entrega1/README_M1.md)
- **Entrega 2:** carpeta con toda la información relativa a la entrega 2. Lo pedido por la entrega se encuentra en [README_M2](Entrega2/README_M2.md)
- **Entrega 3:** carpeta con toda la información relativa a la entrega 3. Lo pedido por la entrega se encuentra en [README_M3](Entrega3/README_M3.md)
- **Final:** carpeta con toda la información relativa a la entrega final. Lo pedido por la entrega se encuentra en [reporte.md](Final/reporte.md)

## Demo desplegada y WandB

**Link de despliegue:** Se utilizó un tunel de Cloudflared local para el despliegue, por lo que la url cambia cada vez que se despliega. Para la demo https://asian-vegetables-asylum-databases.trycloudflare.com 

**Link de Weights and Biases:** [https://wandb.ai/mpinedaa4-eafit/medsummary/1e5dhxqe](https://forge.coreweave.com/wandb/mpinedaa4-eafit/medsummary?nw=nwusermpinedaa4)

## Instrucciones de ejecución

Se requiere Python 3.12 o superior y [uv](https://docs.astral.sh/uv/).

Para correr el sistema localmente, desde la carpeta Final:

```bash
uv sync                                   # instala dependencias
cp .env.example .env                      # opcional: WANDB_API_KEY para ver las peticiones en W&B
uv run uvicorn medsummary.api.main:app --host 127.0.0.1 --port 8000
```
Abrir http://127.0.0.1:8000.
