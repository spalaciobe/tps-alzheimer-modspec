# Informe de evaluación del TFM contra el estado del arte

**Trabajo evaluado:** Replicación de Lopes et al. (2023) — CNN sobre espectrograma de modulación de EEG + saliencia + SVM — sobre OpenNeuro ds004504 (AD vs HC, LOSO-CV, 3 seeds), extendido con la comparación STFT vs CWT-Morlet y la introducción del control **CWT-fair**.
**Marco:** TFM, Maestría UNAL Medellín, curso de Procesamiento Digital de Señales.
**Fuentes de verdad:** `docs/INFORME_TFM.md`, `results/fair_cwt_analysis.json`.
**Método de evaluación:** 6 lentes adversariales web-capaces (sota-eeg-ad · sota-tf-fairness · pertinencia · nuevas-conclusiones · presentación · redacción) + síntesis. Julio 2026.

---

## 1. Veredicto ejecutivo

Es un TFM **sólido, honesto y metodológicamente por encima de la media** de la literatura EEG-AD sobre ds004504: rigor estadístico (DeLong por seed + Stouffer + BH-FDR + poder), LOSO estricta anti-leakage y reproducibilidad total lo ubican en el cuartil superior de rigor del campo, aunque su desempeño (Acc 0.764 / AUC 0.856) quede ~7 puntos por debajo del SOTA directamente comparable (DICE-net, 83.28% LOSO, mismo dataset) y lejos de los foundation models de EEG (LEAD ~91% F1). Su valor **no está en la carrera de accuracy** —el linaje modspec+CNN-saliency+SVM es de una generación anterior— sino en dos cosas: (a) una replicación *leakage-free* creíble y (b) una **contribución genuina de DSP**: identificar que la comparación STFT vs CWT en un espectro de *modulación* está confundida por el eje de modulación y controlarla con CWT-fair. Sirve como **nota metodológica / cautionary tale publicable** (EMBC/EUSIPCO/workshop), no como paper de resultados SOTA; para consolidarlo faltan cerrar la fairness bidireccional, un test de equivalencia formal y el reencuadre explícito del aporte.

---

## 2. Posicionamiento vs estado del arte

### 2.1 EEG-AD sobre ds004504 — dónde cae el 0.764/0.856
El dataset está intensamente *benchmarkeado* y sus cifras dependen críticamente del protocolo de validación:

| Régimen de validación | Rango típico de accuracy | Ejemplos |
|---|---|---|
| Ventana/época sin agrupar por sujeto (**con leakage**) | ~88–99.8% | STEADYNet 88–98%; CNN hasta 99.8% por leakage |
| Media honesta subject-level LOSO | **~71–82%** | Revisión de generalizabilidad 2025: ~71.4% (SD 2.0); AHEPA "riguroso" ~82.5% |
| SOTA comparable subject-level | ~83–91% | **DICE-net 83.28%** (LOSO, AD-CN); LEAD ~91% F1 (foundation model) |
| **Este TFM (LOSO leakage-free)** | **0.764** (0.831 con ensemble multi-seed) | AUC 0.856 (0.900 ensemble) |

**Lectura:** el TFM está **por encima de la media LOSO honesta (~71%)** y **por debajo del mejor SOTA comparable (~83%)**. La brecha de ~7 puntos con DICE-net es real (protocolo equivalente) y el trabajo **no la discute pese a que DICE-net es del propio creador del dataset**. Frente a las cifras de 88–98%, la diferencia es en buena parte metodológica (leakage documentado), lo que convierte el 0.764 en una cifra *más creíble*, no peor.

### 2.2 El método está una generación por detrás
El SOTA 2024–2026 lo forman conv-transformers (DICE-net), GNN sobre conectividad funcional y sobre todo **foundation models de EEG** (LaBraM, EEGPT, CBraMod, LEAD) con pre-entrenamiento auto-supervisado. Saliency-guided patches + SVM es interpretable y barato pero ya no es frontera de desempeño. El TFM apenas menciona este ecosistema.

### 2.3 La novedad del aporte del eje de modulación — ¿tiene precedentes?

- **Ya conocido (no es novedad):**
  - El *principio* "para comparar dos representaciones T-F hay que igualar resolución/preprocesamiento y aislar la transformada" está establecido en audio-DL (Huzaifah 2017; Choi 2017), en fCWT ("fair comparison" entre resoluciones; Arts 2022) y en la resolución temporal aprendible (DiffRes, Liu 2023).
  - La noción de que el **eje de modulación tiene su propio ancho de banda/Nyquist** es *fundacional* en el análisis de amplitud-modulación de EEG (Fraga 2012; Cassani 2013/2020) y en los *modulation filterbanks* auditivos (Chi-Shamma 2005).

- **Genuinamente nuevo (no localizado en la literatura):** el **diagnóstico puntual** de que, en un espectrograma de *modulación*, comparar STFT vs CWT introduce el confound en el **segundo eje** (Nyquist de modulación 1.56 Hz por submuestreo fs/hop en STFT vs 100 Hz en CWT nativa); que ese artefacto DSP —no la transformada— explica el grueso de la diferencia aparente e incluso la inversión del ranking según el método de saliencia; y su **operacionalización como control replicable** (CWT-fair: decimar la envolvente antes de la FFT de modulación).

**Conclusión de posicionamiento:** aporte **incremental pero real**, entre "aplicación cuidadosa de metodología conocida" y "hallazgo nuevo". Ortogonal a la carrera de accuracy, apropiado y hasta sobresaliente para un TFM de DSP; publicable como *short paper* / nota de resultados negativos, no como contribución de alto impacto. Hoy está **sub-encuadrado** frente a los precedentes.

---

## 3. Pertinencia de las conclusiones

**Balance general: las conclusiones están bien soportadas para el alcance declarado y, si acaso, el trabajo *infra-vende* su aporte.** No hay sobre-afirmaciones graves: los NS se reportan como NS, los post-hoc se marcan, y el propio informe expone en su contra la inestabilidad de patches (Jaccard ≈0.05) y las correlaciones de saliencia cercanas a cero.

**Soporte verificado en `fair_cwt_analysis.json`:** SVM vainilla STFT AUC 0.856 / CWT-fair 0.828 / CWT nativa 0.778; DeLong Stouffer STFT vs CWT-fair p=0.318, CWT nativa vs CWT-fair p=0.283; Grad-CAM p=0.587 / 0.527; CNN p=0.679; BH-FDR q≈0.68 en las tres. ✓

**Matiz crítico (compartido por 4 lentes):** "STFT y CWT-fair son estadísticamente indistinguibles" es un **fallo-en-rechazar con potencia baja** (3 seeds, n=65, q≈0.68), no una **equivalencia demostrada**. La formulación estrictamente correcta es *"no se detecta diferencia"*, no *"son equivalentes"*. Falta un **test de equivalencia (TOST)** sobre ΔAUC con margen δ preespecificado, o un IC bootstrap por sujeto. El informe lo reconoce, pero es la brecha de rigor más importante.

**Condicionalidad del nulo:**
- **Fairness unidireccional:** CWT-fair solo "iguala hacia abajo" (descarta modulaciones >1.56 Hz). Sin la condición simétrica (STFT con hop fino "hacia arriba") no se distingue *"la transformada no importa"* de *"el rango de modulación compartido no importa"*.
- **Aislamiento imperfecto:** STFT y CWT-fair siguen difiriendo en el eje **portador** (Fourier lineal vs 32 escalas log de Morlet); conviene explicitarlo.
- **CWT subexplorada:** un solo `cmor1.5-1.0`, sin tuning ni cone-of-influence.

**Sub-afirmación / hallazgos infravalorados:** la **reducción de varianza entre seeds al igualar el eje** (CWT nativa acc ±0.081 → CWT-fair ±0.046; STFT ±0.009) es una conclusión propia ya presente en los datos y apenas explotada. La descomposición tipo mediación (el eje explica ~64% de la brecha en vainilla, ~36% en Grad-CAM) es una estructura reutilizable bien fundamentada.

---

## 4. Nuevas conclusiones posibles + ángulo publicable

**Ángulo publicable central:** reencuadrar el trabajo como **nota metodológica** ("El eje de modulación es un confound al comparar representaciones T-F en espectros de modulación EEG"), con AD como caso de estudio — no como paper de diagnóstico de AD. **Venue realista:** IEEE EMBC (4 pág.) o EUSIPCO / workshop; para journal (*J. Neuroscience Methods* short communication, *Biomedical Signal Processing and Control*, *Frontiers in Neuroinformatics*) conviene sumar antes fusion o validación externa.

**Análisis adicionales de coste casi nulo (máximo ROI, re-análisis puro sobre los `results/*perfold*.json` ya guardados):**

1. **Late-fusion STFT + CWT-fair** a partir de los scores por sujeto. Dado r≈−0.09 entre saliency maps, hay probable complementariedad; si el AUC supera 0.856, el resultado nulo se convierte en un **hallazgo positivo** ("individualmente equivalentes pero complementarios"). Coste casi cero, sin re-entrenar.
2. **TOST sobre ΔAUC** (por seed, margen ±0.05) + efecto mínimo detectable: convierte "no rechazamos" en "equivalentes dentro de δ" o, honestamente, "no concluyente por potencia".
3. **Ablation SVM barato (minutos):** patches **fijos** en bandas canónicas alpha/theta occipito-temporales vs patches saliency-guided. Si empatan → hallazgo fuerte (la maquinaria CNN/saliency no aporta sobre features clásicos); si gana saliency → se justifica el pipeline. Publicable en ambos casos.
4. **Reporte de estabilidad como resultado propio:** la reducción de varianza inter-seed al igualar el eje.
5. **Mini-hallazgo XAI (coste cero):** documentar la concentración anómala de Grad-CAM en gamma (99.3%) vs alpha-theta de vainilla ("cuidado con Grad-CAM sobre espectros de modulación").
6. **Calibración (ECE/Brier)** de los scores existentes, alineándose con la tendencia 2025 de evaluación leakage-free calibrada.

**Coste alto / menor ROI (trabajo futuro declarado):** igualar "hacia arriba" (STFT hop fino → re-entrena CNN ~14h); validación cross-dataset; AD vs FTD / multiclase; segunda parametrización de CWT.

---

## 5. Presentación de la información

**Narrativa: sobresaliente.** Estructura IMRaD completa, resumen ejecutivo con tabla, guion de 7 slides de calidad de conferencia. Tablas rigurosas: media ± SD sobre 3 seeds, siempre las tres condiciones, DeLong con delta por seed.

**Brecha crítica: la contribución central no tiene figura.** La convergencia CWT-nativa → CWT-fair → STFT vive **solo en tablas de texto**. Peor: **todas las figuras embebidas muestran la comparación confundida STFT vs CWT-nativa** que el propio trabajo desaconseja mirar. El lector nunca *ve* el resultado que da nombre al trabajo.

**Qué añadir (prioridad de figuras):**
- **Figura protagonista del hallazgo:** dumbbell/slope o barras agrupadas con las 3 condiciones (STFT/nativa/fair) × pipelines, AUC ± SD, mostrando cómo CWT-fair se desplaza hacia STFT. Reemplaza la figura confundida de §4.3 y va en el slide-clímax (slide 6). Datos en `results/fair_cwt_analysis.json`.
- **Forest plot de DeLong:** ΔAUC por seed + combinado (Stouffer) con IC y línea en 0, para las 3 comparaciones — convención del área para "no hay diferencia".
- **Esquema del eje de modulación:** STFT ~13 bins (→1.56 Hz) | CWT nativa ~180 bins (→22.5 Hz) | CWT-fair ~13 bins, todos colapsando a 45 px tras el resize.
- **Regenerar `auc_master_summary.png`** incluyendo la barra CWT-fair.
- **Panel `modspec_means_cwt_fair.png`** junto a STFT y CWT-nativa.
- **Homogeneizar `saliency_compare.png`:** misma escala de color divergente en todos los paneles.

---

## 6. Redacción del informe

**Fortalezas (nivel casi publicable):**
- **Manejo de incertidumbre ejemplar:** distingue "ausencia de evidencia" de "evidencia de ausencia", acota cada afirmación a su alcance, reporta nulos sin maquillar. Alineado con TRIPOD+AI (2024), REFORMS (2023), NERVE-ML (2025).
- **Transparencia metodológica excepcional:** documenta desviaciones del paper (ICLabel vs wICA, batch size, selección heurística de patches) y hasta qué función *no* se invoca, citando archivo y líneas.
- **Autocrítica robusta:** 16 limitaciones específicas.
- Español académico correcto, terminología DSP usada con propiedad.

**Qué mejorar:**
- **Falta una sección formal de Estado del Arte / trabajo relacionado** que sitúe el 0.764/0.856 frente a los benchmarks 2023–2026 (DICE-net, foundation models) marcando cuáles tienen leakage. Es lo primero que un jurado echará en falta.
- **Cero figuras embebidas** en el cuerpo (solo rutas a .png) — ver §5.
- **Bibliografía delgada (9 refs) y sin nada de 2024–2026** ni guías de reporte.
- **Pasajes de bitácora en la versión final** ("en versiones anteriores de este informe...") → mover a un CHANGELOG/apéndice.
- **Redundancia:** la narrativa del artefacto DSP se repite en resumen, §3.3, §4.3, §5.1 y §6; consolidar las 16 limitaciones en ~8–10 agrupadas.
- **Inconsistencia terminológica:** EA/AD, saliency/saliencia, vainilla/vanilla → fijar un término + tabla de siglas.
- **Exceso de negrita** y callout con emoji, poco formales.
- Convertir el disclaimer clínico en **sección de "Consideraciones éticas"**.
- Elevar la contribución real (identificar y controlar el confound) a **objetivo explícito** en §1.4.

---

## 7. Recomendaciones priorizadas

| Prioridad | Acción | Coste |
|---|---|---|
| **ALTA** | **Reencuadrar el aporte** en abstract/título/objetivos: "benchmark honesto (replicación leakage-free) + hallazgo metodológico (CWT-fair)", separándolo del resultado AD vs HC. | Redacción |
| **ALTA** | **Tabla de benchmark ds004504** subject-level: situar 0.764/0.856 frente a DICE-net (83.28% LOSO) y a las cifras 88–98% marcando el leakage. | Redacción |
| **ALTA** | **Test de equivalencia (TOST)** sobre ΔAUC con margen δ + IC bootstrap por sujeto; reformular "indistinguibles" como equivalencia dentro de δ (o "no concluyente por potencia"). Subir a ≥10 seeds si es viable. | Re-análisis |
| **ALTA** | **Figura protagonista del hallazgo** (3 condiciones × pipelines) + **forest plot de DeLong**, reemplazando la figura confundida de §4.3. | Bajo (datos listos) |
| **ALTA** | **Sección formal de Estado del Arte** con foundation models (LaBraM, EEGPT, LEAD) y conv-transformers; argumentar el nicho del modspec interpretable. | Redacción |
| **MEDIA** | **Late-fusion STFT + CWT-fair** sobre scores guardados: podría transformar el nulo en un hallazgo positivo de complementariedad. | Bajo (sin re-entrenar) |
| **MEDIA** | **Ablation SVM con patches fijos** (bandas canónicas) vs saliency-guided. | Bajo (minutos) |
| **MEDIA** | **Embeber figuras clave** en el cuerpo; homogeneizar escalas de color. | Bajo |
| **MEDIA** | Explicitar que CWT-fair **no aísla perfectamente** la transformada (eje portador log vs lineal); anclar la novedad citando Huzaifah/Choi/fCWT/Fraga/Cassani/Chi-Shamma. | Redacción |
| **MEDIA** | Limpiar meta-comentario de bitácora, reducir redundancia, fijar terminología + glosario, anclar a una guía de reporte (TRIPOD+AI / REFORMS / NERVE-ML), añadir calibración (ECE/Brier). | Redacción / re-análisis |
| **BAJA** | **Igualar "hacia arriba"** (STFT hop fino → Nyquist de la CWT) para cerrar la fairness bidireccional. | Alto (~14h, re-entrena CNN) |
| **BAJA** | **Validación cross-dataset** mínima (2º cohorte EEG-AD público). | Alto |
| **BAJA** | Extender a **AD vs FTD / multiclase**; 2ª parametrización de CWT; **nested CV** para eliminar el selection bias de patches. | Medio |

---

## 8. Referencias relevantes

**SOTA EEG-AD y ds004504**
- Miltiadous et al. (2023). *DICE-Net: A Convolution-Transformer Architecture for Alzheimer Detection in EEG.* IEEE. — SOTA comparable (LOSO AD-CN, 83.28%). https://ieeexplore.ieee.org/document/10179900/
- Wang et al. (2025). *LEAD: Large Foundation Model for EEG-Based Alzheimer's Disease Detection.* arXiv:2502.01678. https://arxiv.org/abs/2502.01678
- *EEG Foundation Models: A Critical Review* (2025), arXiv:2507.11783; *EEG-FM-Bench* (2025), arXiv:2508.17742.
- *A Novel CNN-Based Framework for AD Detection Using EEG Spectrogram Representations* (2025). J. Pers. Med. 15(1):27. https://doi.org/10.3390/jpm15010027
- *EEG-based neurodegenerative disease diagnosis: conventional vs deep learning* (2025). Sci. Rep. https://www.nature.com/articles/s41598-025-00292-z

**Leakage, generalizabilidad y benchmarking**
- *Data leakage in deep learning studies of translational EEG* (2024). Frontiers in Neuroscience. https://doi.org/10.3389/fnins.2024.1373515
- *Evaluating the Generalizability of EEG-Based AI Models in Alzheimer's and Dementia Diagnosis* (2025). medRxiv 2025.09.10.25334048.
- *The AHEPA EEG benchmark* (2026). Cognitive Neurodynamics. https://doi.org/10.1007/s11571-026-10464-w
- *Alzheimer's diagnosis from EEG with reliable probabilities: subject-wise, leakage-free evaluation and isotonic calibration* (2025). J. Eng. Appl. Sci. https://doi.org/10.1186/s44147-025-00821-7

**Comparación justa de representaciones T-F y dominio de la modulación**
- Lopes, Cassani, Falk (2023). *Using CNN Saliency Maps and EEG Modulation Spectra…* Comput. Intell. Neurosci. 2023:3198066 (paper replicado). https://doi.org/10.1155/2023/3198066
- Huzaifah (2017). *Comparison of Time-Frequency Representations for Environmental Sound Classification using CNNs.* arXiv:1706.07156.
- Choi et al. (2017). *A Comparison of Audio Signal Preprocessing Methods for DNNs on Music Tagging.* arXiv:1709.01922.
- Arts & van den Broek (2022). *fCWT.* Nature Computational Science. https://doi.org/10.1038/s43588-021-00183-z
- Liu et al. (2023). *Learning the Spectrogram Temporal Resolution (DiffRes).* arXiv:2210.01719.
- Fraga et al. (2012). *EEG amplitude modulation analysis for semi-automated AD diagnosis.* EURASIP JASP 2012:192. https://doi.org/10.1186/1687-6180-2012-192
- Cassani et al. (2013). *Characterizing AD Severity via Resting-Awake EEG Amplitude Modulation.* PLOS ONE. https://doi.org/10.1371/journal.pone.0072240
- Cassani & Falk (2020). *AD diagnosis based on EEG modulation spectral 'patch' features.* IEEE JBHI 24(7):1982-1993. https://doi.org/10.1109/JBHI.2019.2953475
- Chi, Ru, Shamma (2005). *Multiresolution spectrotemporal analysis of complex sounds.* JASA. https://doi.org/10.1121/1.1945807

**Guías de reporte / reproducibilidad**
- Collins et al. *TRIPOD+AI statement.* BMJ 2024;385:e078378.
- Kapoor et al. *REFORMS: Reporting Standards for ML-Based Science* (2023).
- *NERVE-ML checklist* (2025).

---

*Nota de calibración: evaluado como TFM de maestría en DSP. En ese marco el trabajo es notable en rigor y honestidad (por encima de la media del campo); las brechas señaladas son de posicionamiento, potencia estadística, fairness bidireccional y presentación visual — todas subsanables sin re-correr los experimentos caros. Con el reencuadre + TOST + late-fusion + la figura protagonista, pasa de "muy buen TFM" a "material publicable como nota metodológica".*

*(Generado por evaluación adversarial multi-agente con búsqueda web, workflow `eval-sota-tfm`.)*
