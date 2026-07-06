# Guion de presentación — 5 minutos

**TFM: ¿Importa cómo descomponemos el EEG para detectar Alzheimer?**
Sebastián Palacio Betancur · UNAL Medellín · TPS

> **Cómo usar este guion**: cada slide tiene su tiempo objetivo, la idea
> que debe quedar, y un texto sugerido (léelo natural, no de memoria). El
> arco narrativo es: *quisimos probar que CWT gana → obtuvimos un titular
> → el rigor nos frenó dos veces → y encontramos algo más interesante*.
> Total: ~5:00. Si vas apurado, el corazón es la **slide 5**.

---

## Slide 1 — Título · (0:00–0:15) · 15 s

**Idea**: presentarte y plantar la pregunta en una frase.

> "Buenos días. Mi proyecto se pregunta algo simple: para detectar
> Alzheimer con EEG, **¿importa cómo descomponemos la señal** en
> tiempo y frecuencia? Repliqué un pipeline del estado del arte y comparé
> dos formas de hacerlo."

---

## Slide 2 — La pregunta · (0:15–1:00) · 45 s

**Idea**: por qué EEG, qué es el espectrograma de modulación, y la hipótesis.

> "El Alzheimer es la demencia más común, y detectarlo temprano importa,
> pero resonancia y PET son caras. El **EEG es barato y portátil**.
>
> La representación que uso se llama **espectrograma de modulación**: no
> mira la frecuencia directamente, sino *a qué ritmo cambia* la energía
> de cada banda en el tiempo. Y todo eso depende del primer paso —cómo
> hago la descomposición tiempo-frecuencia—, ese bloque naranja del
> diagrama.
>
> Mi **hipótesis** era que la **CWT**, la transformada wavelet, con su
> mejor resolución en bajas frecuencias, debería ganarle a la STFT.
> El plan: replicar el pipeline de Lopes 2023, pero sobre **datos
> públicos**, y ver si la CWT de verdad gana."

*Transición*: "Primero, lo que construí."

---

## Slide 3 — Lo que construí · (1:00–1:40) · 40 s

**Idea**: el pipeline en un vistazo + el truco del método + mi ablation.

> "El pipeline va de izquierda a derecha: preproceso el EEG, calculo el
> espectrograma —con STFT **o** CWT—, y entreno una CNN.
>
> El truco del método de Lopes es elegante: la CNN **no clasifica**; se
> usa para *descubrir* qué regiones del espectrograma separan enfermos de
> sanos, vía mapas de saliencia. Un SVM final usa esas regiones como
> *features*.
>
> Yo monté la comparación completa: dos transformadas, por dos métodos de
> saliencia, por tres semillas. Y todo con **anti-leakage estricto**:
> cada cosa se calcula dentro de cada fold, nunca con datos de test."

*Transición*: "¿Y qué salió?"

---

## Slide 4 — El titular · (1:40–2:25) · 45 s

**Idea**: la replicación funciona; STFT tiene mayor AUC media — pero falta probar que sea real.

> "Esto es lo primero, y es una buena noticia: la **replicación
> funciona**. La mejor configuración —SVM vainilla con STFT, que es la
> fiel al paper— llega a un **AUC de 0.856**, en el mismo rango que el
> paper original, pero sobre datos públicos independientes. Eso ya
> justifica el trabajo.
>
> Y fíjense: la STFT tiene **mayor AUC media** que la CWT, 0.856 contra
> 0.778. La pregunta natural es: ¿esa diferencia es **real**, o es solo
> ruido entre semillas?
>
> Antes de afirmar que una transformada gana, hay que probarlo bien."

*Transición (clave, baja el tono)*: "Y ahí vino el giro."

---

## Slide 5 — El giro · (2:25–3:35) · 70 s · ⭐ CORAZÓN

**Idea**: dos frenos independientes. Es la slide más importante — no la apures.

> "El rigor me obligó a frenar dos veces.
>
> **Freno uno, la estadística.** La inferencia correcta es el test DeLong
> **por semilla** —preservando la variabilidad entre inicializaciones—, y
> luego combinando los p-valores. Cuando lo hago así y corrijo por
> comparaciones múltiples… **ninguna comparación sobrevive**. Ninguna. El
> resultado real es 'no concluyente'.
>
> **Freno dos, y este es de procesamiento de señales.** Las dos imágenes
> son 45 por 45, se ven iguales… pero su **eje de modulación no mide lo
> mismo**: en la STFT el Nyquist es 1.5 Hz, en la CWT es 100. O sea que
> ni siquiera estaba comparando peras con peras: la comparación está
> **confundida por DSP**.
>
> Así que la conclusión honesta es: **no puedo afirmar que una transformada
> le gane a la otra**. Ni CWT, ni STFT."

*Transición*: "Pero al frenar, apareció lo interesante."

---

## Slide 6 — Lo que de verdad encontré · (3:35–4:25) · 50 s

**Idea**: los dos hallazgos que valen más que la hipótesis original.

> "Dos cosas.
>
> **Una**: el ranking depende del **método de saliencia**. Con Grad-CAM
> parece ganar la CWT; con vainilla, la STFT. Los mapas de las dos
> transformadas son **estadísticamente ortogonales** —miran regiones
> distintas—, lo que sugiere que podrían ser **complementarias**: un
> ensemble de las dos es una idea natural a futuro.
>
> **Y dos, lo que más me gustó**: el modelo **redescubre biología
> conocida**. La mejor configuración concentra casi el 90% de su atención
> en las bandas alfa y theta, y en los canales occipito-temporales —O1,
> O2, T5, T6—, que es justo donde la clínica localiza el Alzheimer. Nadie
> se lo dijo; lo encontró solo.
>
> El *qué* descubre el modelo depende del pipeline completo, no solo de
> la transformada. Ese fue el hallazgo más valioso."

*Transición*: "Para cerrar."

---

## Slide 7 — Lo que me llevo + Gracias · (4:25–5:00) · 35 s

**Idea**: tres takeaways, reproducibilidad, cierre.

> "Me llevo tres cosas. **Uno**: la replicación funciona sobre datos
> públicos —el método es reproducible—. **Dos**: mi hipótesis no se
> sostuvo… pero la contraria tampoco; el rigor manda. Y **tres**, lo más
> importante: las **lecciones metodológicas** valen más que el resultado
> —reportar varianza, elegir bien el estadístico, y comparar transformadas
> solo con ejes equivalentes—.
>
> Todo está en un **repo público**, con auditoría externa incluida. Lo
> primero que haría a futuro es unificar ese eje de modulación antes de
> volver a comparar.
>
> **Gracias** —quedo atento a sus preguntas."

---

## Preguntas probables (Q&A) — respuestas de bolsillo

- **"¿Entonces tu hipótesis falló?"**
  No exactamente: no encontré evidencia a favor de CWT, pero tampoco
  probé que STFT sea mejor. El aporte real es *demostrar que la
  comparación, como se hace usualmente, está confundida* y mostrar cómo
  hacerla bien.

- **"¿Por qué STFT tiene Nyquist de modulación tan bajo?"**
  Porque el paso temporal de la STFT (con `noverlap=64` a 200 Hz) es
  0.32 s. Eso fija el Nyquist del eje de modulación en 1.56 Hz. La CWT
  conserva los 200 Hz, así que su Nyquist es 100. Al forzar ambas a
  45×45, quedan midiendo rangos físicos distintos.

- **"¿Por qué solo 3 seeds?"**
  Costo de cómputo: ~48 h en una RTX 3050 de 4 GB. Con 3 ya se ve la
  varianza (CWT vainilla tiene SD altísima). Idealmente 10+, y lo dejo
  como trabajo futuro.

- **"¿Por qué el análisis de bandas es solo para STFT?"**
  El eje portador de la CWT es logarítmico (`geomspace`); asignar
  píxel→banda requiere otro mapeo. Reportarlo con el eje lineal de la
  STFT habría sesgado la CWT, así que lo restringí honestamente.

- **"¿Es usable clínicamente?"**
  No. Es una replicación académica sobre datos públicos, sin validación
  externa ni aprobación regulatoria. Los patches además son inestables
  entre folds (Jaccard ≈ 0.05).

---

## Notas de entrega

- **Ritmo**: ~110–120 palabras/min. El guion tiene ~720 palabras → ~5:00.
- **Dónde respirar**: las transiciones en *cursiva* son pausas — úsalas.
- **Si te sobra tiempo**: expande slide 6 (biología redescubierta).
- **Si te falta tiempo**: en slide 5 di solo el freno 1 (estadística) y
  menciona el DSP en una frase.
- **Slide que no puedes saltar**: la 5. Es el argumento intelectual del
  trabajo.
