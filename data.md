# Descripción del Dataset

## Fuente

El dataset usado (**MTS-Dialog**) se obtuvo del siguiente repositorio de GitHub: <https://github.com/abachaa/MTS-Dialog>.

Este repositorio hace parte de un paper de una aproximación empírica para la toma de notas asistida por IA de la interacción médico-paciente:

> Asma Ben Abacha, Wen-wai Yim, Yadan Fan, and Thomas Lin. 2023. An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters. In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pages 2291–2302, Dubrovnik, Croatia. Association for Computational Linguistics.

El dataset de este repositorio viene, a su vez, de otro dataset llamado **MTsamples** (<https://www.mtsamples.com/>), una colección de historias clínicas anonimizadas.

Para evitar el riesgo de violación de privacidad, los autores del paper tomaron estas historias clínicas y generaron conversaciones simuladas a partir de ellas (que es el dataset que usamos para este proyecto). Estos diálogos simulados fueron realizados por 8 anotadores entrenados, con experiencia en el sector médico.

Se tomaron las 6 especialidades de consulta más frecuentes: medicina general, neurología, ortopedia, dermatología, alergología e inmunología.

El formato de las notas clínicas es el formato **SOAP**, que es el más usado, el cual divide una historia clínica en 4 secciones: subjetivo, objetivo, análisis y plan.

## Tamaño

El dataset final incluye **1700 diálogos** médico-paciente junto con su resumen para la historia clínica. Cada diálogo, junto con su resumen, se dividió posteriormente en diferentes secciones de historia clínica, para crear un resumen de acuerdo con lo que se espera de esa parte de la historia clínica:

| # | Sección | Descripción |
|---|---|---|
| 1 | `fam/sochx` | Antecedentes familiares y sociales |
| 2 | `genhx` | Enfermedad actual |
| 3 | `pastmedicalhx` | Antecedentes patológicos |
| 4 | `cc` | Motivo de consulta |
| 5 | `pastsurgical` | Antecedentes quirúrgicos |
| 6 | `allergy` | Antecedentes alérgicos |
| 7 | `ros` | Revisión por sistemas |
| 8 | `medications` | Antecedentes farmacológicos |
| 9 | `assessment` | Análisis |
| 10 | `exam` | Examen físico |
| 11 | `diagnosis` | Diagnóstico |
| 12 | `disposition` | Disposiciones al final de la consulta |
| 13 | `plan` | Plan de tratamiento |
| 14 | `edcourse` | Departamento de emergencias |
| 15 | `immunizations` | Antecedentes de inmunizaciones |
| 16 | `imaging` | Imágenes diagnósticas |
| 17 | `gynhx` | Antecedentes ginecoobstétricos |
| 18 | `procedures` | Procedimientos realizados |
| 19 | `other_history` | Otros antecedentes |
| 20 | `labs` | Paraclínicos |

Esta división hace que la unidad de análisis del dataset sean secciones específicas de la historia clínica y no el diálogo completo. Esto hace que se expanda el dataset y se obtenga un total de **15982 registros**.

## División de los datos

El repositorio de GitHub ya trae el dataset dividido en el conjunto de train, validation y dos de test. Tomamos la decisión de respetar esta división por defecto, pero juntamos los dos archivos de test en un solo conjunto:

| Conjunto | Registros | Porcentaje |
|---|---|---|
| Train | 11454 | 71.67% |
| Test | 3714 | 23.24% |
| Validación | 814 | 5.09% |

## Idioma

El idioma del dataset es **inglés**.

## Licencia

El repositorio tiene un `LICENSE.txt` que expresa que el conjunto de datos se encuentra bajo una licencia **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Esta es una licencia bastante laxa que permite copiar, redistribuir, modificar y utilizar comercialmente el material, siempre que cumplas las condiciones de atribución (lo cual realizamos en la sección de fuente de este README).

Una aclaración pertinente de esta licencia es que no protege el uso de datos personales por completo. Sin embargo, el dataset original (MTsamples) contiene información anonimizada y, adicionalmente, los autores del dataset MTS-Dialog usaron diálogos simulados, por lo que el dataset no cuenta con información de datos personales que puedan identificar a los participantes y este riesgo se atenúa.

## Tarea

El dataset contiene 4 columnas: `ID`, `section_header`, `section_text` y `dialogue`. Para nuestro proyecto no nos interesa el `ID` ni el `section_header` (que es la sección a la cual pertenece ese fragmento dentro de la historia clínica).

| Columna | Rol |
|---|---|
| `dialogue` | **Input**: conversación entre médico y paciente |
| `section_text` | **Output**: resumen con terminología médica del diálogo para ser registrado en la historia clínica |

Lo anterior termina siendo una doble tarea de **resumen y transformación** del texto. Se debe sintetizar la información, pero a su vez transformarla en un texto con jerga médica.

## Sesgos o limitaciones conocidas

- **Diálogos simulados**: la limitación principal del dataset es que los diálogos son simulados y puede no corresponder a una conversación natural y espontánea entre un médico y un paciente.
- **Idioma**: está en inglés y nosotros nos encontramos en un contexto donde hablamos español; la forma de comunicación y las expresiones pueden no traducir bien entre los idiomas si se deseara hacer el modelo para un público hispanohablante.
- **Cobertura de especialidades**: el dataset está acotado a 6 especialidades principales, por lo que si llega a recibir un diálogo de una consulta por otra especialidad no necesariamente va a tener un desempeño aceptable.
- **Granularidad**: las parejas diálogo-resumen corresponden cada una a un solo elemento de la historia clínica (antecedentes, medicamentos, etc.), por lo que es posible que al recibir el diálogo completo, o diálogos más largos, el comportamiento del modelo cambie.

A pesar de estas limitaciones, consideramos que es un dataset suficientemente robusto para crear una aproximación inicial a un modelo de IA para el problema que se quiere resolver.
