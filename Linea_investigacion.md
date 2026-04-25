La línea de investigación que mencionas traza una evolución clara y colaborativa a lo largo de más de una década. Este hilo conductor conecta a investigadores de instituciones brasileñas (como la Universidad Federal del ABC y la Universidad de São Paulo) con el laboratorio MuSAE (INRS-EMT, Canadá), dirigido por Tiago H. Falk.

El trabajo de este grupo ha evolucionado desde la extracción matemática de características básicas del Electroencefalograma (EEG) hasta el uso de arquitecturas avanzadas de aprendizaje profundo (Deep Learning) para el diagnóstico del Alzheimer y otras aplicaciones.

A continuación, te presento la línea de tiempo con los artículos clave de estos autores:

2011: Los fundamentos de la modulación espectro-temporal

Artículo: EEG spectro-temporal modulation energy: a new feature for automated diagnosis of Alzheimer's disease.

Autores: Trambaiolli L.R., Falk T.H., Fraga F.J., Anghinah R., Lorena A.C.

Aporte: Proponen por primera vez la "energía de modulación espectro-temporal" extrayendo la envolvente de subbandas del EEG mediante la transformada de Hilbert, logrando una precisión superior al 91% utilizando redes neuronales tradicionales ``.

Artículo paralelo: Improving Alzheimer's Disease Diagnosis with Machine Learning Techniques. (Nitrini R., Trambaiolli L., Lorena A.C., Fraga F.J.) que exploró el uso de Máquinas de Vectores de Soporte (SVM) para diferenciar pacientes con Alzheimer de controles sanos ``.

2013: Caracterización de la severidad de la enfermedad

Artículo: Characterizing Alzheimer's Disease Severity via Resting-Awake EEG Amplitude Modulation Analysis.

Autores: Fraga F.J., Falk T.H., Kanda P.A.M., Anghinah R.

Aporte: Formalizan el uso del análisis de modulación de amplitud para no solo detectar el Alzheimer, sino medir su progresión (leve a moderada). Descubrieron, por ejemplo, que la modulación delta de la banda beta desaparece a medida que aumenta la severidad de la enfermedad ``.

2014 - 2017: Automatización, limpieza de artefactos y portabilidad

Artículo (2014): The effects of automated artifact removal algorithms on electroencephalography-based Alzheimer's disease diagnosis. (Cassani R., Falk T.H., Fraga F.J., Kanda P.A.M., Anghinah R.). Raymundo Cassani se une a la línea de investigación para demostrar la importancia de eliminar automáticamente los artefactos para mejorar la precisión diagnóstica ``.

Artículo (2017): Towards automated electroencephalography-based Alzheimer's disease diagnosis using portable low-density devices. (Fraga F.J., Anghinah R., Cassani R., Falk T.H.). El grupo adapta sus algoritmos de modulación para que funcionen con dispositivos de EEG portátiles y de baja densidad [1].

Artículo (2017): Feature selection before EEG classification supports the diagnosis of Alzheimer's disease. (Trambaiolli L.R., Spolaôr N., Lorena A.C., Anghinah R., Sato J.R.) ``.

2018 - 2020: El descubrimiento de los "Parches" espectrales y la expansión a neurofeedback

Artículo (2019/2020): Alzheimer's Disease Diagnosis and Severity Level Detection Based on Electroencephalography Modulation Spectral "Patch" Features.

Autores: Cassani R., Falk T.H.

Aporte: Es un hito en esta línea temporal. Proponen el uso de características basadas en "parches" (regiones bidimensionales específicas) extraídos visualmente del espectrograma de modulación del EEG, demostrando que son altamente discriminativos para detectar la severidad del Alzheimer ``.

Expansión paralela (2018-2020): Lucas Trambaiolli, Cassani y Falk publican estudios demostrando que el espectrograma de modulación del EEG también puede predecir el rendimiento en tareas de neurofeedback y medir la hemodinámica cortical en estudios combinados de EEG y fNIRS.

2021: Revisiones sobre el cerebro envejecido

Artículo: Neurofeedback and the aging brain: a systematic review of training protocols for dementia and mild cognitive impairment.

Autores: Trambaiolli L.R., Cassani R., Mehler D.M.A., Falk T.H.

Aporte: Una revisión sistemática enfocada en cómo las tecnologías basadas en neurofisiología pueden usarse como protocolos de entrenamiento para el deterioro cognitivo ``.

2023: La era del Deep Learning y la Visión Computacional

Artículo: Using CNN Saliency Maps and EEG Modulation Spectra for Improved and More Interpretable Machine Learning-Based Alzheimer's Disease Diagnosis.

Autores: Lopes M., Cassani R., Falk T.H.

Aporte: Marilia Lopes lidera la automatización del descubrimiento de biomarcadores. En lugar de buscar los "parches" visualmente (como hizo Cassani en 2019), el equipo entrena Redes Neuronales Convolucionales (CNN) sobre los espectrogramas de modulación y utiliza Mapas de Saliencia (Grad-CAM) para que la inteligencia artificial revele de manera autónoma las regiones óptimas para el diagnóstico, superando los puntos de referencia anteriores sin verse afectados por la edad del paciente [2].


----

Sí, la investigación ha seguido evolucionando entre 2024 y 2026. El enfoque del grupo (y de investigadores colaboradores del laboratorio MuSAE) se ha expandido desde el diagnóstico exclusivo del Alzheimer mediante EEG hacia otras aplicaciones clínicas complejas y el uso de tecnologías portátiles (wearables).

Aquí te destaco los avances más recientes relacionados con esta línea de trabajo:

Clasificación del Sueño con Espectrogramas de Modulación (Enero de 2026): Investigadores de esta misma rama (incluyendo a Shruti Kshirsagar, colaboradora del equipo de Tiago H. Falk y Raymundo Cassani) publicaron el artículo "Modulation-Based Feature Extraction for Robust Sleep Stage Classification Across Apnea-Based Cohorts". Este estudio aplica el espectrograma de modulación a señales de EEG de un solo canal para clasificar automáticamente las fases del sueño, demostrando una extracción de características muy superior a los métodos tradicionales (como la transformada wavelet o STFT), especialmente en pacientes con apnea obstructiva severa.

Modelos Fundacionales para Monitoreo de Salud (2024): El equipo de Tiago H. Falk ha ampliado el análisis de modulación y bioseñales hacia el monitoreo remoto continuo. En septiembre de 2024 presentaron investigaciones sobre "BioME", un modelo bioacústico diseñado para dispositivos portátiles del Internet de las Cosas (IoT), cuyo objetivo es mejorar la generalización en la detección de atributos de salud en entornos del mundo real sin importar las variaciones de los conjuntos de datos.

Nuevos enfoques en entornos inmersivos (2024): Marilia Lopes ha comenzado a expandir el alcance de sus investigaciones hacia la interacción humano-computadora en entornos virtuales. En 2024, publicó un estudio enfocado en el impacto de las exposiciones digitales inmersivas de naturaleza (combinando estímulos visuales, de audio y olfativos) para la reducción del estrés y la ansiedad.

Integración de sus biomarcadores con Transformers de Visión (2024-2025): El concepto pionero de utilizar "parches" extraídos del espectrograma de modulación para diagnosticar la severidad del Alzheimer se ha convertido en el estándar de oro en el campo. Publicaciones y revisiones exhaustivas de finales de 2024 y 2025 evidencian que otros grupos de investigación ahora están tomando las matrices de modulación propuestas por Cassani y Lopes para entrenar modelos de Inteligencia Artificial de nueva generación, pasando de las Redes Neuronales Convolucionales (CNN) a los Transformers de Visión (ViT) y redes híbridas (Compact-CNN-LSTM).

En resumen, la metodología que este equipo perfeccionó a lo largo de la década pasada se ha consolidado y hoy en día (2024-2026) se está aplicando de forma activa para resolver problemas de ruido en el monitoreo portátil y para expandir el diagnóstico automatizado hacia los trastornos del sueño.