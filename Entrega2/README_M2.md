Tópicos Especiales y Aplicaciones en Inteligencia Artificial — SI4006 
# Entrega M2 - Harness de Modelo
## Esteban Álvarez Zuluaga, Mateo Pineda Álvarez

Sistema que toma la conversación entre un médico y su paciente y genera un borrador de la historia clínica, para ahorrarle tiempo de documentación al personal de salud.

Para esta entrega, se construyo un "harness" o esquema de evaluación para el modelo. Se consideraron métricas clásicas automaticas, jueces LLM, asi como evaluación en ejemplos curados. 

## Contenido
- README_M2.md: Información general de la entrega, detalle del harness, resultados y conclusiones
- Versionamiento del Prompt: La carpeta '''PromptVersions''' contiene el versionamiento de los prompts del juez
- [Experimentos con prompts](ExperimentosPrompts.md): Detalle sobre experimentos con los prompts (no pedido, pero se exponen pruebas empíricas realizadas)
-  [Experimentos con prompts](ExperimentosPrompts.md): Detalle sobre experimentos con los prompts (no pedido, pero se exponen pruebas empíricas realizadas)


## Scorecard del Baseline

==========================================================================
Métrica                                        Baseline      Finetuned
--------------------------------------------------------------------------
1 · ROUGE-1 promedio                              0.235          0.377
1 · ROUGE-2 promedio                              0.100          0.134
1 · ROUGE-L promedio                              0.212          0.296
1 · BERTScore F1 promedio                         0.304          0.345
2 · LLM-juez promedio (1-5)                        2.90           3.20
3 · Aciertos de dominio                            6/10           6/10
==========================================================================

## Evaluación Honesta
El harness permite medir un modelo de manera que se indica la calidad de sus respuestas en el dominio, pero no con un alto grado de confianza. Utilizando las métricas clásicas ROUGE y BERTScore se puede realizar un análisis de similitud sintáctica y semántica del texto. Se espera que estas métricas puntúen alto al buscar resumenes, pero se entiende que no consideran la factualidad, estructura, cohesión. Se utilizan como medida del cambio entre modelos, ver si ha empeorado la similitud de la entrada-respuesta, pero no permiten clasificar si una respuesta es o no es correcta por si mismas. 

Por su parte, el LLM-Judge permite evaluar aspectos como factualidad, cohesión y estructura asi como casos extremos por medio de una rúbrica. Vimos que la definición de esta rúbrica afecta completamente el resultado, o es muy tajante o muy laxo en la evaluación, el propio juez halucina u olvida instrucciones, y en general no encontramos que siga fielmente el criterio que tratamos impartirle. En general, se logró que diferenciara entre las respuestas muy buenas y las muy malas, pero hay muchos falsos positivos, falsos negativos y respuestas "decentes" que clasifica en los extremos.   
