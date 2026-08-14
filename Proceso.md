# Elección del Modelo

La tarea seleccionada corresponde a un *sequence-to-sequence* de diálogos de paciente-doctor a historia clínica preliminar. Como proceso de selección, se determinaron ciertos criterios que debía cumplir el modelo:

## **Rendimiento**

* El modelo debe ser entrenable en una GPU T4 Gratuita de Colab usando LoRA o QLoRA en tiempo razonable.
* Idealmente debe poder ser ejecutable localmente en un computador de consumidor con GPU Nvidia GTX 1650 y similar.

## **Conocimiento de Dominio**

* Poseer algún tipo de alineamiento a seguir instrucciones con RLHF o similar, de forma que se pueda mejorar su capacidad *zero-shot* solo con prompts. Este mismo alineamiento podría ayudar a evitar comportamientos fuera de lugar.
* Posee conocimiento de medicina porque:

  * Fue diseñado para aplicaciones médicas o bioclínicas.
  * Fue entrenado o se le realizó fine-tuning sobre información médica como libros, exámenes, etc.
* Haya demostrado buenos resultados en benchmarks médicos.

## **Tokenizador e Idioma**

* Su tokenizador debe funcionar bien en texto médico, usualmente comprendido de palabras largas con raíces griegas y en latín, así como términos descriptivos específicos.
* El tokenizador debe funcionar bien en español, considerando tildes y ñ. Idealmente debe ser bilingüe y desempeñarse bien en el idioma.

## **Licencia**

* Debe contar con una licencia que permita el uso académico. Se debe buscar la licencia más libre posible que incluso permita modificación, alteración y comercialización del modelo.

## **Familia del Modelo**

Antes de buscar modelos con estas características, se consideró respecto a la familia de modelo, que esta tarea es de resumen, lo cual recomendaría el uso de un encoder-decoder. Aun así, debido a que para los modelos de tipo "decoder" hay mayor disponibilidad de modelos con más parámetros y más especializados, se consideraron también, al tener buenos resultados en tareas de resumen.

## **Experimentos con MedGemma-1.5-4B-it**

MedGemma es un modelo de tipo decoder entrenado por Google cuyo objetivo es servir en aplicaciones médicas. Esto por medio de entrenamiento sobre conocimiento médico como corpus de texto e historias clínicas. El modelo cuenta con instruction tuning, de manera que responde bien a prompts detallando la tarea a realizar y amplificando las capacidades zero-shot para la tarea. Está entrenado en múltiples idiomas, y su tokenizador SentencePiece funciona adecuadamente en español.

Finalmente, cuenta con versiones de múltiples tamaños, siendo la más pequeña de 4B parámetros. Esta versión es ejecutable en Google T4, y es posible realizar fine-tuning en Colab.

Probándolo sobre el dataset, mostró capacidad zero shot muy buena, resumiendo la información de manera adecuada, pero sin seguir el formato presente en el dataset. Al hacer fine-tuning se demoró excesivamente, y la desconexión con el servicio no permitió su realización. Dado esto, se buscó otro modelo más pequeño donde pudiera completarse. El modelo escogido fue flanT5 por los motivos expuestos en el README.
