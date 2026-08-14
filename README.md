Tópicos Especiales y Aplicaciones en Inteligencia Artificial — SI4006 
# Entrega M1 - Fine-tuning baseline
## Esteban Álvarez Zuluaga, Mateo Pineda Álvarez

Sistema que toma la conversación entre un médico y su paciente y genera un borrador de la historia clínica, para ahorrarle tiempo de documentación al personal de salud.

## Contenido
- README.md: Información general, modelo y familia escogidos y resultados
- [data.md](data.md): Información detallada sobre el dataset
- [Selección del Modelo](model_selection.md): Detalle sobre selección del modelo
-  [Fine tuning con LoRA](Entrega1_TopicosIA_EAZ_MPA.ipynb): Cuaderno de colab con fine-tuning LoRA y pruebas contra baseline


## Modelo Base y Familia Escogidos

__Escogimos la familia encoder-decoder, por su capacidad de obtener información contextual del texto a resumir con el encoder, y de generar nuevo texto resumen que tome la información contextual con el decoder.__

Nuestra tarea es la transformación y resumen de texto, por lo que se espera que la familia de modelo que pueda servir es un encoder-decoder. Basado en esto, la familia de modelos T5 era la opción más utilizada y efectiva para la tarea. De esta familia elegimos el modelo Flan-T5 (por encima de T5 o T5-small) porque este ya cuenta con cierto instruction-tuning, lo cual es una ventaja al no tener que enseñarle comportamiento de seguimiento de instrucciones, y que el baseline tenga mejores capacidades zero-shot.

Para llegar al modelo Flan-T5 usado, realizamos un proceso de busqueda basado en los criterios de nuestro problema, e inicialmente habíamos seleccionado el modelo medgemma-1.5-4b-it. Los detalles del proceso de selección se pueden encontrar en 

## Baseline de Comparación

El baseline para realizar la comparación es el mismo modelo sin fine-tuning (zero-shot) usando el mismo prompt de instrucción que usamos después del fine-tuning. Esta es una buena forma de comparación porque estamos tratando de asegurar que el menor número de variables sean las que influyan en los resultados del modelo y que la diferencia observada sea solamente por el fine-tuning y no por otros factores. Para compararlos, se realizan pruebas empiricas con unos pocos ejemplos, asi como métricas de similitud entre el texto generado por el modelo y el teórico como ROUGE y BERTScore.

## Tabla de Resultados

Como métricas cuantitativas se utilizaron las medidas ROUGE y BERTScore. Las medidas ROUGE miden similitud sintáctica de cadenas de texto, ROUGE-1 toma solo por palabra, ROUGE-2 por par de palabras y ROUGE-L por la subsecuencia común más grande. BERTScore corresponde a un cálculo de similitud de los embeddings generados por BERT. Siendo este el caso ROUGE permite cuantificar similitud sintáctica, y BERT semántica.

| Metric     | Baseline | Finetuned |       Δ |
| ---------- | -------: | --------: | ------: |
| ROUGE-1    |   0.2438 |    0.3613 | +0.1175 |
| ROUGE-2    |   0.0762 |    0.1289 | +0.0527 |
| ROUGE-L    |   0.2043 |    0.2987 | +0.0944 |
| ROUGE-Lsum |   0.2041 |    0.2976 | +0.0936 |
| BERT F1    |   0.3136 |    0.3937 | +0.0801 |

Vemos que en todas las métricas se logró mejora. Parece ser pequeña, pero se debe entender lo que representan. Para ROUGE-1 con un 11.7% de mejora, implica que casi el 40% de las palabras se comparten entre la predicción y el objetivo. Es importante tener en cuenta que dos resúmenes válidos pueden usar palabras completamente distintas, especialmente en el campo médico. Es por esto mismo que ROUGE-2 y ROUGE-L permanecen bajos, pero aún así mejoran. 

## Frase de lectura honesta

Cualitativamente, evidenciamos una mejora significativa, ya que el  modelo pasa de tratar de generar algo "plausible" pero incorrecto como una continuación del dialogo, a generar en todos los casos evaluados texto de resumen. Vemos que difiere un poco las palabras exactas, pero logra aprender el estilo y manera  del dataset, así como fundamentalmente no alterar, inventar o alucinar información. Cuantitativamente, los resultados indican mejora, pero se tienen en cuenta sus limitaciones.
