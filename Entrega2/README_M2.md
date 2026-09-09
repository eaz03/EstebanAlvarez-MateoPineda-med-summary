Tópicos Especiales y Aplicaciones en Inteligencia Artificial — SI4006 
# Entrega M2 - Harness de Modelo
## Esteban Álvarez Zuluaga, Mateo Pineda Álvarez

Sistema que toma la conversación entre un médico y su paciente y genera un borrador de la historia clínica, para ahorrarle tiempo de documentación al personal de salud.

Para esta entrega, se construyó un "harness" o esquema de evaluación para el modelo. Se consideraron métricas clásicas automaticas, jueces LLM, así como evaluación en ejemplos curados. 

## Contenido
- README_M2.md: Información general de la entrega, detalle del harness, resultados y conclusiones
- Versionamiento del Prompt: La carpeta '''PromptVersions''' contiene el versionamiento de los prompts del juez
- [Notebook](Entrega2_TopicosIA_EAZ_MPA.ipynb): El notebook de Colab con el código.
- [Conjunto de evaluación](eval_set.json): Los 10 ejemplos usados para el harness (incluyendo 2 ejemplos adversariales).
- [Scorecard](scorecard.csv): El scorecard final que muestra la comparación de los resultados del harness aplicados al modelo baseline (zero-shot) y al modelo fine-tuned.

Dentro de los directorios `baseline` y `fine-tuned` se encuentran mayores detalles de las evaluaciones del harness en cada uno de los ejemplos.

## Mitigación de sesgo LLM-as-ajudge
Para este ejercicio tomamos en cuenta el sesgo de auto-preferencia, el cual consiste en que un modelo va a preferir respuestas de sí mismo o de su familia de modelos. Nuestro modelo fine-tuned es un Flan-T5 base (modelo desarrollado por Google), mientras que el LLM-as-a-judge usado fue Qwen2.5-1.5B-Instruct, un modelo de pesos abiertos desarrollado por Alibaba Cloud. Al ser modelos de familias diferentes logramos mitigar el sesgo de auto-preferencia. Por otro lado, al utilizar una rúbrica por valor y no comparación entre dos opciones, se mitiga el sesgo de posición. También se consideró que el juez podría priorizar criterios que aparezcan antes en la rúbrica, y se evidenció este fenómeno. Se trató de corregir con iteraciones de la rúbrica, pero no fue totalmente exitoso.

## Scorecard del Baseline

| Métrica | Baseline | Finetuned |
| :--- | :---: | :---: |
| 1 · ROUGE-1 promedio | 0.235 | 0.377 |
| 1 · ROUGE-2 promedio | 0.100 | 0.134 |
| 1 · ROUGE-L promedio | 0.212 | 0.296 |
| 1 · BERTScore F1 promedio | 0.304 | 0.345 |
| 2 · LLM-juez promedio (1-5) | 2.90 | 3.20 |
| 3 · Aciertos de dominio | 6/10 | 6/10 |

## Evaluación Honesta
El harness permite medir un modelo de manera que se indica la calidad de sus respuestas en el dominio, pero no con un alto grado de confianza. Utilizando las métricas clásicas ROUGE y BERTScore se puede realizar un análisis de similitud sintáctica y semántica del texto. Se espera que estas métricas puntúen alto al buscar resumenes, pero se entiende que no consideran la factualidad, estructura, cohesión. Se utilizan como medida del cambio entre modelos, ver si ha empeorado la similitud de la entrada-respuesta, pero no permiten clasificar si una respuesta es o no es correcta por si mismas. Solo tuvimos en cuenta el BERTScore a la hora de calcular los aciertos, ya que en nuestro problema buscamos una transformación del texto que conserve su significado, aunque no necesariamente mantenga las mismas palabras o estructuras que el texto de referencia. A diferencia de ROUGE, que se basa principalmente en el solapamiento léxico entre ambos textos, BERTScore permite evaluar en mayor medida su similitud semántica.

Por su parte, el LLM-Judge permite evaluar aspectos como factualidad, cohesión y estructura asi como casos extremos por medio de una rúbrica. Vimos que la definición de esta rúbrica afecta completamente el resultado, el juez resulto o muy tajante o muy laxo en la evaluación. El propio juez alucina u olvida instrucciones, y en general no encontramos que siga fielmente el criterio que tratamos impartirle. En general, se logró que diferenciara entre las respuestas muy buenas y las muy malas, pero hay muchos falsos positivos, falsos negativos y respuestas "decentes" que clasifica en los extremos.

En cuanto a la dimensión más severa, pudimos notar que era el LLM. Las métricas clásicas se pueden optimizar fácilmente con los modelos haciendo una transformación casi que literal del diálogo de los pacientes y se obtendrían buenos resultados de ROUGE y BERTScore. Sin embargo, con el juez notamos que según la rúbrica podría ser muy estricto y, respuestas que pueden ser decentes o incluso muy buenas tendía a calificarlas mal. Esto se logró mitigar un poco cambiando el prompt de la rúbrica y encontrando un punto medio donde no fuera tan estricto pero tampoco laxo.

En cuanto a los casos adversariales, el LLM aparentemente es capaz de detectarlos correctamente y darles una calificación adecuada. Usamos dos casos adversariales: una conversación agendando una cita (que no es un diálogo del acto médico en sí) y otra en la que un paciente está solicitando un medicamento de control y altamente adictivo (opioides). Lo ideal es que el modelo en el primer caso detecte que no es un diálogo que merezca ser convertido en nota médica y tenga una válvula de escape; para el segundo caso también debería detectar un comportamiento ilegal y tener una válvula de escape. Al hacer el fine-tuning, no tuvimos en cuenta estos casos entonces nuestro modelo trata de hacer una nota clínica al respecto. El LLM-as-a-judge supo que este no era un comportamiento correcto y calificó negativamente estos dos casos.

Para finalizar con el tema del LLM-as-a-judge, parte del problema que tiene en nuestro caso es que estamos usando un modelo muy pequeño, que probablemente esté entrenado para conversaciones generales pero nuestro dominio es un nicho específico y con muchos tecnicismos, por lo cual esto puede afectar el desempeño del LLM a la hora de juzgar si una respuesta es correcta o no. Lo ideal es poder usar un modelo más especializado en el área médica, pero estos tienden a ser muy grandes y no cumple con el requisito de esta entrega.
