Tópicos Especiales y Aplicaciones en Inteligencia Artificial — SI4006 
# Entrega M1 — Fine-tuning baseline
## Esteban Álvarez Zuluaga, Mateo Pineda Álvarez

Sistema que toma la conversación entre un médico y su paciente y genera un borrador de la historia clínica, para ahorrarle tiempo de documentación al personal de salud.

## Contenido
- README.md: Información general, modelo y familia escogidos y resultados
- [data.md](data.md): Información detallada sobre el dataset
- [Selección del Modelo](model_selection.md): Detalle sobre selección del modelo
-  : Cuaderno de colab con fine-tuning LoRA y pruebas contra baseline


## Modelo Base y Familia Escogidos

__Escogimos la familia encoder-decoder, por su capacidad de obtener información contextual del texto a resumir con el encoder, y de generar nuevo texto resumen que tome la información contextual con el decoder.__

Nuestra tarea es la transformación y resumen de texto, por lo que se espera que la familia de modelo que pueda servir es un encoder-decoder. Basado en esto, la familia de modelos T5 era la opción más utilizada y efectiva para la tarea. De esta familia elegimos el modelo Flan-T5 (por encima de T5 o T5-small) porque este ya cuenta con cierto instruction-tuning, lo cual es una ventaja al no tener que enseñarle comportamiento de seguimiento de instrucciones, y que el baseline tenga mejores capacidades zero-shot.

Para llegar al modelo Flan-T5 usado, realizamos un proceso de busqueda basado en los criterios de nuestro problema, e inicialmente habíamos seleccionado el modelo medgemma-1.5-4b-it. Los detalles del proceso de selección se pueden encontrar en 

## Baseline de Comparación

El baseline para realizar la comparación es el mismo modelo sin fine-tuning (zero-shot) usando el mismo prompt de instrucción que usamos después del fine-tuning. Esta es una buena forma de comparación porque estamos tratando de asegurar que el menor número de variables sean las que influyan en los resultados del modelo y que la diferencia observada sea solamente por el fine-tuning y no por otros factores. Para compararlos, se realizan pruebas empiricas con unos pocos ejemplos, asi como métricas de similitud entre el texto generado por el modelo y el teórico como ROUGE y BERTScore.

## Tabla de Resultados

| Metric     | Baseline | Finetuned |       Δ |
| ---------- | -------: | --------: | ------: |
| ROUGE-1    |   0.2438 |    0.3613 | +0.1175 |
| ROUGE-2    |   0.0762 |    0.1289 | +0.0527 |
| ROUGE-L    |   0.2043 |    0.2987 | +0.0944 |
| ROUGE-Lsum |   0.2041 |    0.2976 | +0.0936 |
| BERT F1    |   0.3136 |    0.3937 | +0.0801 |


## Frase de lectura honesta

