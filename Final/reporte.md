# El problema y el usuario
## ¿Quién es?
Médicos que deseen convertir transcripciones de diálogos médico-pacientes en resúmenes de historia clínica con terminología médica apropiada.

## ¿Qué decide con el sistema?
El usuario debe ingresar un diálogo médico-paciente, del cual recibe un resumen de historia clínica. A partir de esto, el usuario debe decidir si es una interpretación correcta según el contexto clínico del paciente y en qué sección de la historia clínica es la más apropiada para ingresar este resumen.

## ¿Qué métrica de éxito importa para esa decisión?
La métrica ideal en este contexto siempre será el juicio por un experto (médico). Sin embargo, la forma en la que más nos acercamos, y que está implementada en nuestro harness, es con un juez LLM que mida qué tan correcto es este resumen (en cuanto a no inventar datos que no están, que se tomen los datos que realmente son relevantes y que se use la terminología médica correcta).

# El sistema en un diagrama
![Diagrama del sistema](Diagrama_Sistema.png)

# Decisiones técnicas
## Modelo base
Nuestra tarea es la transformación y resumen de texto, por lo que se espera que la familia de modelo que pueda servir es un encoder-decoder. De esta familia elegimos el modelo Flan-T5 (por encima de T5 o T5-small) porque este ya cuenta con una base de instruction-tuning. Adicionalmente, debía ser un modelo lo suficientemente pequeño para que pudiera ejecutarse en Colab gratuito, lo cual fue la limitante para no escoger modelos encoder-decoder más grandes que proporcionaran un resultado más robusto.

La alternativa que se pensó fue MedGemma-1.5-4B-it, pero se descartó porque el fine-tuning se demoraba demasiado y Colab se desconectaba.

El proceso completo de selección de este modelo se puede encontrar en [Proceso.md](../Entrega1/Proceso.md).

## Recuperación
El corpus del RAG es el libro "Medical Terminology, 2nd edition" (Open RN, licencia CC BY 4.0), dividido en 6086 chunks sin overlap. Para los embeddings elegimos BAAI/bge-small-en-v1.5 en lugar de paraphrase-multilingual-MiniLM-L12-v2, porque todo nuestro pipeline trabaja en inglés (no necesitamos un modelo multilingüe) y porque acepta 512 tokens en vez de 128, lo que permite representar chunks de párrafos extensos. Los embeddings se guardan en ChromaDB como base de datos vectorial. Como modelo generador del RAG usamos Qwen2.5-1.5B-Instruct.

Sobre el RAG ingenuo construimos un RAG avanzado con tres técnicas: descomposición de la consulta con un LLM para extraer cada expresión que debe traducirse a terminología médica (máximo 6 por nota), hybrid search que combina la búsqueda dense con BM25 mediante RRF, y un reranker que reordena los chunks recuperados (k=3 por término y score mínimo mayor a 0 para evitar reemplazos que no sean coherentes). El proceso completo de construcción del RAG se puede encontrar en [README_M3.md](../Entrega3/README_M3.md).

## Patrón visual
Para este proyecto no se implementaron modelos visuales, debido a que el problema implica la conversión de conversaciones entre médicos y pacientes a resúmenes de historia clínica. Este proceso solo involucra texto y no requiere el uso de imágenes para llevar a cabo el objetivo y la resolución del problema. La implementación de un componente visual adicionalmente involucra modelos adicionales de peso elevado, y mayor procesamiento que empeoran la latencia y portabilidad del sistema. Se consideró mejor hacer un sistema más robusto que cumpla con sus necesidades clave de contexto, y las cumpla correctamente, a una adición que no añada tanto valor, y afecte los otros criterios de calidad.

## Técnica de eficiencia
Para la cuantización se empleó como técnica el GGUF usando llama-cpp. La razón para elegir GGUF en lugar de otras herramientas de cuantización como bitsandbytes es porque GGUF nos permite cuantizar y usar el modelo tanto en CPU como en GPU, lo cual representa una ventaja frente a los otros métodos de cuantización porque no teníamos claro inicialmente el entorno en el cual sería desplegado nuestro pipeline. Otros métodos exigen GPU e, incluso, algunos exigen que sea de la marca NVIDIA, lo cual representa una restricción grande a la hora de poder desplegar el código y, más aún, si a esto le sumamos que sea de manera gratuita.

La decisión de realizar la cuantización con int8 o int4 se redujo a las pruebas realizadas con ambas cuantizaciones. Si bien pudimos comprobar (como lo vimos en clase) que la diferencia de latencia entre int8 e int4 no es tan significativa como la que hay entre el modelo sin cuantizar e int8, elegimos quedarnos con int4 por varias razones. Los resultados del harness no fueron tan diferentes entre ambas cuantizaciones, lo que indica que con int4 la pérdida de calidad no es tan significativa como lo podría ser en otros escenarios. Inicialmente probamos el sistema con int8, el cual funcionaba adecuadamente (evidenciado en el harness); sin embargo, pudimos apreciar que con esta cuantización el agente no llamaba nunca la tool del RAG, sin importar el tipo de input que se le pasara. Al intentar los mismos ejemplos con int4, el agente sí pudo reconocer que debía llamar la tool del RAG cuando era pertinente y por este motivo decidimos elegir esta cuantización en lugar de int8.

Considerando el trade-off entre eficiencia y precisión, contemplamos perder un  máximo de un 10% del desempeño medido por el harness, para bajar la latencia en al menos un 30%. No consideramos un valor más alto de desempeño, ya que preferimos sacrificar latencia sobre calidad, considerado el contexto médico en que se trabaja, donde prima la factualidad, exactitud, y no es fundamental la velocidad.

| Versión | Dispositivo | ROUGE-L | BERTScore F1 | Juez grounding (1-5) | Juez coverage (1-5) | Juez form (1-5) | Juez final ponderado (1-5) | Aciertos de dominio | Latencia p50 (s) | Latencia p95 (s) | Memoria Qwen (GB) | VRAM extra (GB) | Memoria total del sistema (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Antes: Qwen sin cuantizar (bfloat16) | mps | 0.363 | 0.418 | 3.00 | 2.90 | 3.80 | 3.11 | 0.10 | 3.51 | 6.06 | 3.09 | – | 4.35 |
| Después: Qwen GGUF Q8_0 (~int8) | mps | 0.363 | 0.410 | 3.40 | 3.00 | 3.60 | 3.30 | 0.10 | 3.06 | 4.72 | 1.89 | – | 3.16 |
| Después: Qwen GGUF Q4_K_M (~int4) · desplegada | mps | 0.340 | 0.397 | 3.40 | 2.90 | 3.80 | 3.30 | 0.10 | 2.52 | 4.41 | 1.12 | – | 2.38 |

En la tabla se compara el RAG avanzado con Qwen sin cuantizar contra sus dos versiones cuantizadas con GGUF (int8 e int4), evaluadas con el mismo harness y en el mismo dispositivo. Se puede observar lo siguiente:

- **Métricas clásicas:** Con int8 el ROUGE-L se mantiene igual (0.363) y el BERTScore F1 baja levemente de 0.418 a 0.410 (-1.9%). Con int4 la reducción es un poco mayor: ROUGE-L baja a 0.340 (-6.3%) y BERTScore F1 a 0.397 (-5.0%). Estas métricas solo miden similitud con la nota de referencia, por lo que no permiten concluir si la nota es mejor o peor, solo que se parece un poco menos.

- **Juez LLM:** El juez final ponderado no empeora con la cuantización, sino que sube de 3.11 a 3.30 (+6.1%) tanto con int8 como con int4. Al desglosarlo por dimensión, la mejora viene casi toda del juez de grounding (3.00 a 3.40), es decir, las versiones cuantizadas inventan menos información. El coverage se mantiene prácticamente igual (2.90 - 3.00) y el juez de forma baja ligeramente con int8 (de 3.80 a 3.60), pero se recupera con int4 (3.80). Los aciertos de dominio se mantienen en 0.10 para las tres versiones. Como se explica en la tabla evolutiva, no se puede concluir el motivo de la mejora, pero se puede hacer una hipótesis de que se relaciona con un cambio en el uso del RAG: con menos reemplazos de términos, se dejan de cometer los peores errores (como el de "pulmonary embolism"). Esto indica que, al menos con nuestro harness de 10 ejemplos, la cuantización no tuvo un costo en la calidad, por lo que se cumple el umbral del 10%.

- **Latencia:** La latencia mediana (p50) bajó de 3.51s sin cuantizar a 3.06s con int8 (-12.8%) y a 2.52s con int4 (-28.2%). La latencia p95 tuvo un comportamiento similar: de 6.06s a 4.72s (-22.1%) y a 4.41s (-27.2%). En el p95 sí se confirma lo visto en clase: la mayor parte de la mejora ocurre al pasar del modelo sin cuantizar a int8 (una reducción de 1.34s), y el paso de int8 a int4 aporta mucho menos (0.31s). En la mediana, en cambio, ambos pasos aportan una mejora parecida (0.45s y 0.54s). Con int4 la reducción de la mediana queda muy cerca del 30% propuesto, y si se compara contra la mediana de 3.62s medida para el RAG avanzado en la tabla evolutiva, la reducción es de 30.4%.

- **Memoria:** Es la métrica en la que la cuantización tiene el efecto más claro. La memoria que ocupa Qwen baja de 3.09 GB a 1.89 GB con int8 (-38.8%) y a 1.12 GB con int4 (-63.8%). La memoria total del sistema baja de 4.35 GB a 3.16 GB (-27.4%) y a 2.38 GB (-45.3%). La diferencia entre la memoria total y la de Qwen se mantiene constante en aproximadamente 1.26 GB en las tres versiones, lo cual corresponde a los componentes que no se cuantizaron (Flan-T5 con LoRA, modelo de embeddings y reranker). Esto permite que el sistema completo quepa en su totalidad en entornos con recursos limitados, como Colab gratuito o una máquina virtual que no tenga GPU sino que solo cuente con CPU.

En conclusión, int4 ofrece el mejor balance: obtiene el mismo puntaje del juez que int8, tiene menor latencia y casi la mitad de la memoria total que el modelo sin cuantizar. A esto se suma la razón expuesta anteriormente, en la que establecimos que con int4 el agente sí llamaba la tool del RAG cuando era pertinente. Por estas razones es la versión desplegada.

# Tabla evolutiva

| Sistema | Dispositivo | ROUGE-L | BERTScore F1 | Juez grounding (1-5) | Juez coverage (1-5) | Juez form (1-5) | Juez final ponderado (1-5) | Aciertos de dominio | Latencia mediana p50 (s) |
|---|---|---|---|---|---|---|---|---|---|
| Modelo ajustado (Flan-T5 + LoRA) | mps | 0.381 | 0.444 | 3.40 | 3.00 | 4.00 | 3.38 | 0.10 | 0.98 |
| RAG ingenuo (Qwen sin cuantizar) | mps | 0.395 | 0.452 | 3.40 | 2.90 | 4.20 | 3.39 | 0.20 | 2.81 |
| RAG avanzado (Qwen sin cuantizar) | mps | 0.363 | 0.418 | 3.00 | 2.90 | 3.80 | 3.11 | 0.10 | 3.62 |
| RAG avanzado con Qwen Q4_K_M · desplegado | mps | 0.340 | 0.397 | 3.40 | 2.90 | 3.80 | 3.30 | 0.10 | 2.52 |

Según la tabla evolutiva podemos ver la evolución del sistema evaluado con el mismo harness, y con cada modificación. Se midió desde el modelo con fine-tuning, modelo con RAG ingenuo, RAG avanzado, y finalmente el modelo cuantizado desplegado. Se puede observar lo siguiente:

- **Métricas clásicas:** Se tomaron las métricas Rouge y BERT. Se evidencia una mejora en las métricas al utilizar el RAG ingenuo, pero se evidencia una ligera baja de las métricas para el RAG avanzado. Como estas métricas se basan exclusivamente en similitud sintactica, y el RAG tiene como objetivo una paráfrasis semántica a términos médicos, es de esperar que no se parezca exactamente a la nota inicial, pero mantenga el significado. Como tal, aunque no permite concluirlo, se puede conjeturar que el "empeoramiento" de las métricas clasicas, indicando diferencia sintáctica, refleja realmente una paráfrasis  (sin poder determinar si fue buena o mala, solo que ya no se parecen tanto, lo cual es de esperar). Con el RAG avanzado cuantizado, estas métricas también presentaron una disminución comparado con el RAG sin cuantizar, sin embargo cumple con el umbral establecido de 10% de reducción de las métricas.
- **Juez LLM:** El juez LLM (flowaicom/Flow-Judge-v0.1) se divide en tres categorías, cada una midiendo una dimensión de la historia clínica generada. El juez de grounding, mide que todo lo que este en la historia se haya visto reflejado en el dialogo (que el modelo no invente), el juez de coverage mide que todo lo importante del dialogo se vea reflejado en la historia (que el modelo no omita), y finalmente el juez de forma mide que la historia tenga el formato y la prosa sea coherente con una historia clínica. Cada juez genera un puntaje, y se realiza un promedio ponderado, dandole más importancia a la factualidad (0.45·grounding + 0.35·coverage + 0.20·form). Al comparar los resultados del juez ponderado entre las versiones del modelo, vemos que hay un aumento muy reducido (3.38 a 3.39, de modelo base a RAG ingenuo), casi insignificante, asi como una reducción más signitiva con el RAG avanzado (3.38 a 3.11 de modelo base a RAG avanzado). Analizando las respuestas del juez, se evidencia que esta reducción es probablemente por el comportamiento del RAG avanzado en el cual inyecta algunos síntomas no especificados por el paciente, o traduce a términos médicos términos que no lo ameritan. En un ejemplo cambia un "felt a pull" a "felt a pulmonary embolism", lo cual es penalizado por el juez. Se evidenció también que el juez es capaz de entender los terminos médicos, entonces cuando hay una cooreta como "chest pain" a "angina" no baja el puntaje. Como tal hay un pequeno empeoramiento en la calidad de las respuestas, por este problema conocido, pero en muchos otros casos mejora la calidad médica. Cuando el se cuantiza el RAG, se obtiene una mejora en el desempeño de 3.11 a 3.30. No se puede concluir un motivo para la mejora, pero se bevidenció en las pruebas que por ejemplo con cuantización int4 el modelo utilizaba mucho menos RAG. Se puede conjeturar sin buen sustento que los cambios entonces por cuantización probablemente fueron suficientes para que por ausencia de llamadas, se dejaran de realizar los peores errores. Al no empeorar, claramente supera el umbral establecido del 10%.
- **Latencia:** La latencia mediana del modelo con fine-tuning sin RAG, fue de 0.98s. Esto incrementó a 2.81s con el RAG ingenuo, y 3.62s con el RAG avanzado. Finalmente, disminuyó a 2.52s con la cuantización. Claramente se evidencia mejoría, y se logró el trade-off de una mejora en latencia de 30% con menos del 10% de pérdida de calidad. Sigue siendo mayor al modelo sin cuantizar, pero es incluso mejor que el RAG ingenuo, aunque utilice muchas más llamadas a otros modelos.

# Ablación

| Configuración | Juez final (1-5) | BERTScore F1 | Aciertos de dominio | RAGAS context recall | RAGAS context precision | RAGAS faithfulness * | RAGAS answer relevancy * | Δ juez vs. completo | p50 RAG (s) | p95 RAG (s) | Notas distintas (de 10) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sistema completo (hybrid + reranker) | 3.11 | 0.418 | 0.10 | 0.20 | 0.16 | 0.20 | 0.20 | 0.000 | 2.49 | 5.03 | 0 |
| Sin reranker (solo hybrid) | 2.94 | 0.389 | 0.10 | 0.10 | 0.14 | 0.00 | 0.17 | -0.180 | 2.94 | 5.33 | 5 |
| Sin hybrid (semántica + reranker) | 3.08 | 0.423 | 0.10 | 0.20 | 0.15 | 0.20 | 0.40 | -0.040 | 2.61 | 4.93 | 3 |
| Sin reranker ni hybrid (solo semántica) | 3.17 | 0.372 | 0.10 | 0.11 | 0.18 | 0.00 | 0.17 | 0.060 | 2.96 | 5.20 | 7 |
| Referencia: sin RAG (fine-tuned) | 3.38 | 0.444 | 0.10 | – | – | – | – | 0.265 | – | – | 5 |

Para la ablación tomamos dos elementos del RAG avanzado: el reranker y el hybrid search. Hicimos el ejercicio de quitar uno, luego el otro y finalmente ambos y medir cómo se comporta el sistema con estos cambios usando nuestro harness. Se puede observar lo siguiente:

- **Juez LLM:** Quitar el reranker es lo que más afecta al sistema: el juez final baja de 3.11 a 2.94 (-0.18). Quitar el hybrid search casi no tiene efecto (baja a 3.08). Sin embargo, al quitar ambos componentes el juez sube a 3.17 (+0.06), lo cual va en contra de lo esperado. Al revisar las notas, esta configuración comete errores graves, como cambiar "metformin" por "glucagon" o "temperature" por "extremes", y aun así el juez no la penaliza. Por esto, conjeturamos que con 10 ejemplos las diferencias tan pequeñas en el juez (menores a 0.1) pueden deberse al ruido y no a una mejora real, y que el juez no siempre detecta los reemplazos incorrectos. Ninguna configuración supera al modelo sin RAG (3.38), lo cual es coherente con lo visto en la tabla evolutiva.

- **Métricas clásicas:** El BERTScore F1 se mantiene similar entre el sistema completo (0.418) y la configuración sin hybrid (0.423), mientras que baja con la configuración sin reranker (0.389) y aún más con solo búsqueda semántica (0.372). Esta última es justamente la que más notas modifica (7 de 10), por lo que es de esperar que se parezca menos a la nota de referencia. Los aciertos de dominio se mantienen en 0.10 en todas las configuraciones, por lo que esta métrica no permite diferenciar el aporte de cada componente.

- **RAGAS:** Aquí se ve más claro el aporte del reranker. El context recall es de 0.20 en las dos configuraciones que tienen reranker y baja a 0.10 y 0.11 en las que no lo tienen, es decir, sin reranker se recuperan menos chunks relevantes. Lo mismo pasa con faithfulness, que es 0.20 con reranker y 0.00 sin él, lo que indica que sin reranker las notas reescritas no se apoyan en el contexto recuperado. El context precision es bajo en todas las configuraciones (va de 0.14 a 0.18), lo que muestra que la mayoría de chunks recuperados no son útiles, sin importar el componente. El answer relevancy es mayor en la configuración sin hybrid (0.40), pero al ser calculado solo sobre las notas que cambiaron, no permite concluir mucho.

- **Latencia:** El sistema completo es la configuración más rápida (p50 de 2.49s), y quitar el reranker la aumenta a 2.94s. Esto puede parecer contradictorio, ya que el reranker es un paso adicional. Se puede conjeturar que se debe al filtro de score mínimo mayor a 0 del reranker: sin él se toman siempre los 3 primeros chunks por expresión, lo que genera más reemplazos y más trabajo de reescritura para Qwen. El p95 es similar en todas las configuraciones (4.93s a 5.33s).

- **Estabilidad de las notas:** Quitar el hybrid search solo cambia 3 de las 10 notas, mientras que quitar el reranker cambia 5 y quitar ambos cambia 7. Al revisar los cambios, sin reranker aparecen reemplazos sin sentido, como "felt a pull" a "felt a intercostal retractions", "soft" a "SOFT PALATE", o "allergic to penicillin" a "hypersensitivity" (con lo que se pierde el nombre del medicamento). Esto confirma que el reranker es el que filtra los chunks que no son coherentes con la expresión que se debe traducir.

En conclusión, el reranker es el componente que más aporta al RAG avanzado: mejora el juez, el recall y la faithfulness, y reduce los reemplazos incorrectos. El hybrid search tiene un aporte marginal en nuestro corpus. Aun así, ninguna configuración del RAG supera al modelo sin RAG según el juez, lo cual es consistente con el problema ya identificado de que el RAG a veces reemplaza expresiones coloquiales por términos médicos incorrectos (como "pulmonary embolism"), error que se mantiene incluso en el sistema completo.

# Análisis de errores
Para el análisis de errores revisamos las notas generadas por cada sistema en los 10 ejemplos del harness, junto con las explicaciones del juez. Hay que tener en cuenta que todos los sistemas parten del mismo borrador generado por Flan-T5 con LoRA, por lo que los errores del borrador se heredan en todas las versiones, y el RAG solo puede agregar errores adicionales sobre él. Agrupamos los errores en cinco tipos:

## 1. Omisión de información clínica relevante
Aquí lo que falla es el modelo fine-tuneado (Flan-T5 + LoRA), que genera el borrador. Esto se puede ver en varios casos del eval_set, como en el ejemplo 0, el 1 y el 4.

El modelo tiende a quedarse con la primera frase del diálogo y producir notas muy cortas. Para intentar arreglarlo, probaríamos primero un ajuste de los parámetros de generación y entrenar con más épocas o con un modelo más grande (Flan-T5-large), si los recursos lo permiten. También se podría agregar un paso de verificación que compare las entidades clínicas del diálogo (medicamentos, dosis, síntomas) con las de la nota y regenere si falta alguna. También sería una buena idea iterar con diferentes tipos de prompt para encontrar el que mejores resultados produzca.

## 2. Conversión de antecedentes negativos a positivos y viceversa
Aquí lo que falla principalmente es el modelo fine-tuneado y en algunos casos la reescritura del RAG (Qwen).m Se puede ver en ejemplos del eval_set como el 2 y el 3.

Este es uno de los errores más graves para el usuario, porque una alergia o un síntoma inventado puede cambiar una decisión clínica. Para el borrador, agregaríamos al entrenamiento más ejemplos con antecedentes negados, y un post-procesamiento con un detector de negaciones que compare cada hallazgo entre el diálogo y la nota y verifique que realmente el antecedente es positivo o negativo. Para el RAG,se podría validar que las palabras de negación ("no", "denies") se mantengan y no sean reemplazadas al buscar en el corpus.

## 3. Reemplazos incorrectos de terminología médica
Lo que falla aquí es el RAG avanzado. Se puede observal en el eval_set en ejemplos como el 0, el 5 y el 7.

El problema suele empezar en la descomposición, que extrae palabras que no son síntomas ("pull", "soft", "high", "8.2"). Podríamos intentar restringir el prompt de descomposición para que no extraiga números ni adjetivos sueltos. Adicionalmente, podríamos subir el umbral del reranker porque es muy permisivo (acepta chunks que tenga un puntaje mayor a 0), y no permitiríamos reemplazar valores numéricos.

## 4. Diálogos fuera de alcance o con riesgo de seguridad
La válvula de escape del modelo fine-tuneado a veces no es capaz de verificar si el diálogo está en un contexto clínico o no, por lo que depende de la válvula implementada en el RAG. Esto se puede ver en los ejemplos 8 y 9.

Para arreglarlo, agregaríamos una válvula de escape más robusta desde el inicio del pipeline, es decir, desde el modelo fine-tuneado. SI se activa debería devuelver un mensaje en lugar de una nota. Para casos sensibles como solicitudes de sustancias controladas, agregaríamos ejemplos al entrenamiento en los que se documente explícitamente la decisión del médico ("No prescription was given"). El prompt del RAG ya indica que se devuelva el borrador sin cambios si no es una nota clínica, pero esto no es una medida robusta si el borrador ya viene mal.

## 5. Errores de atribución y de forma
Aquí falla el modelo fine-tuneado. El error de atribución se refiere a que el modelo pone las palabras de un sujeto como si las dijera otro (es decir, como si el médico dijera lo del paciente o viceversa). También ocurren errores en pronombres (por ejemplo usar segunda persona en vez de la tercera persona, que no corresponde al formato de historia clínica que se escribe en tercera persona). Esto se puede ver en los ejemplos 5, 6 y 7.

El dataset de entrenamiento (MTS-Dialog) tiene secciones en las que el médico habla casi todo el tiempo (evaluación, plan, examen físico), y el modelo no aprende a convertirlas a tercera persona. Se podría intentar agregar un post-procesamiento que detecte pronombres de primera y segunda persona y regenere la nota usando la tercera persona del singular. También se podría intentar incluir en el entrenamiento más ejemplos de esas secciones. En las ocasiones donde el modelo se inventa el sexo del paciente se puede reemplazar por "the patient" cuando el diálogo no lo menciona.

# Limitaciones

# Ética
## ¿A quién podría fallarle más?
Sospechamos que el modelo no funciona tan bien en ciertos diálogos pertenecientes a partes puntuales de la historia clínica. Por ejemplo, cuando se describen antecedentes del paciente (por ejemplo antecedentes personales o alérgicos), el modelo suele confundirse y toma antecedentes negativos y los transforma en antecedentes positivos (es decir, cuando un paciente niega tener un antecedente, el modelo lo transforma y afirma que el paciente sí tiene ese antecedente).
Adicionalmente, el RAG se suele equivocar transformando terminología médica muy especializada o aquellas cuya forma coloquial de decirse requiera más de una palabra (no siempre pasa esto, pero algunas veces sí falla cuando debe reconocer que el término coloquial se compone por más de una palabra).

Estos fallos no se midieron objetivamente sino que fueron producto de la experimentación que tuvimos a la hora de ensayar el modelo con varios tipos de ejemplos.

FInalmente, es claro que nuestro modelo deja por fuera a cualquier médico cuya práctica no sea en inglés, ya que el modelo solo fue entrenado con diálogos en este idioma.

## ¿Qué pasa con los datos?
El sistema recibe datos de conversaciones médico-paciente. Es responsabilidad del usuario que ingresa estos diálogos garantizar la privacidad, tanto del paciente como del médico, anonimizando todo dato que pueda llevar a la identificación de los sujetos. Sin embargo, nuestro sistema usa modelos locales, por lo que esta información no se comparte con modelos de empresas como OpenAI o Anthropic. Nuestro sistema lo único que recolecta son logs de hora, latencia, componentes utilizados en la petición y si se usó la válvula; además, enviamos métricas del harness y rendimiento a Weights & Biases (que es un tercero), pero ninguno de estos datos contiene información de los datos clínicos proporcionados al sistema. Adicionalmente, no guardamos ningún dato en nuestro sistema: el sistema no cuenta con una base de datos para almacenar los inputs del usuario; la única base de datos es ChromaDB pero esta solo guarda el corpus del RAG. 

## ¿Qué daño haría un error?
El peor daño que se puede cometer es que la nota clínica generada tenga imprecisiones suficientemente graves que puedan impactar en el diagnóstico y/o tratamiento de los pacientes, lo que implica un daño directo sobre pacientes (en su salud) y sobre el médico (en términos legales y de ética médica). El sistema genera una alerta al crear un resumen clínico advirtiendo al usuario de que este es un sistema de IA y que esta debe ser revisada por un profesional de la salud para verificar la veracidad y tomar la decisión final.

# Quién hizo qué

**Esteban:** Refactorización del notebook a proyecto desplegable en FastAPI basado en clases, consolidación de los avances de Mateo al proyecto desplegable, evaluación y realización del despliegue.


**Mateo:** Refactorización del notebook a proyecto desplegable en FastAPI basado en clases, implementación de la cuantización, tablas de comparación, página de despliegue, monitoreo en WandB

# Uso de IA

Durante el desarrollo de esta entrega se usó Claude como asistente de generación de código bajo supervisión humana y como apoyo para la adquisición de información para toma de decisiones. El análisis de resultados, las decisiones técnicas y las conclusiones del reporte son del equipo.

## Referencias

Abdin, M., Aneja, J., Awadalla, H., Awadallah, A., Awan, A. A., Bach, N., ... Zhou, X. (2024). *Phi-3 technical report: A highly capable language model locally on your phone* (arXiv:2404.14219). arXiv. https://doi.org/10.48550/arXiv.2404.14219

Anthropic. (2026). *Claude* (versión Opus 5.5) [Modelo de lenguaje grande]. https://claude.ai

Ben Abacha, A., Yim, W., Fan, Y. y Lin, T. (2023). An empirical study of clinical note generation from doctor-patient encounters. En *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics* (pp. 2291–2302). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.eacl-main.168

Biewald, L. (2020). *Experiment tracking with Weights and Biases* [Software]. https://wandb.ai

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., ... Wei, J. (2024). Scaling instruction-finetuned language models. *Journal of Machine Learning Research, 25*(70), 1–53. https://jmlr.org/papers/v25/23-0870.html

Cloudflare. (s. f.). *Cloudflare Tunnel* [Software]. https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/

Cormack, G. V., Clarke, C. L. A. y Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. En *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 758–759). ACM. https://doi.org/10.1145/1571941.1572114

Ernstmeyer, K. y Christman, E. (Eds.). (2024). *Medical terminology* (2.ª ed.). Open RN | WisTech Open; Chippewa Valley Technical College. https://wtcs.pressbooks.pub/medterm/

Es, S., James, J., Espinosa-Anke, L. y Schockaert, S. (2024). RAGAs: Automated evaluation of retrieval augmented generation. En *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations* (pp. 150–158). Association for Computational Linguistics. https://aclanthology.org/2024.eacl-demo.16

Flow AI. (2024). *Flow-Judge-v0.1* [Modelo de lenguaje]. Hugging Face. https://huggingface.co/flowaicom/Flow-Judge-v0.1

Gerganov, G. y colaboradores. (s. f.). *llama.cpp* [Software]. GitHub. https://github.com/ggml-org/llama.cpp

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. y Chen, W. (2022). LoRA: Low-rank adaptation of large language models. En *International Conference on Learning Representations (ICLR 2022)*. https://doi.org/10.48550/arXiv.2106.09685

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. En *Text Summarization Branches Out* (pp. 74–81). Association for Computational Linguistics. https://aclanthology.org/W04-1013

Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S. y Bossan, B. (2022). *PEFT: State-of-the-art parameter-efficient fine-tuning methods* [Software]. GitHub. https://github.com/huggingface/peft

Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R. y Deng, L. (2016). *MS MARCO: A human generated machine reading comprehension dataset* (arXiv:1611.09268). arXiv. https://doi.org/10.48550/arXiv.1611.09268

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. En *Advances in Neural Information Processing Systems 32* (pp. 8024–8035). Curran Associates.

Qwen Team. (2024). *Qwen2.5 technical report* (arXiv:2412.15115). arXiv. https://doi.org/10.48550/arXiv.2412.15115

Ramírez, S. (s. f.). *FastAPI* [Software]. https://fastapi.tiangolo.com

Reimers, N. y Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. En *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Robertson, S. y Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389. https://doi.org/10.1561/1500000019

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N. y Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. En *Advances in Neural Information Processing Systems 33* (pp. 5776–5788). Curran Associates.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. En *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 38–45). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-demos.6

Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D. y Nie, J.-Y. (2024). C-Pack: Packed resources for general Chinese embeddings. En *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 641–649). ACM. https://doi.org/10.1145/3626772.3657878

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q. y Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. En *International Conference on Learning Representations (ICLR 2020)*. https://doi.org/10.48550/arXiv.1904.09675

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E. y Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. En *Advances in Neural Information Processing Systems 36: Datasets and Benchmarks Track*. https://doi.org/10.48550/arXiv.2306.05685
