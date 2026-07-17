# Guion de presentación — 5 minutos

**TFM: ¿Importa cómo descomponemos el EEG para detectar Alzheimer?**
Sebastián Palacio Betancur · UNAL Medellín · TPS

> **Cómo usar este guion**: cada slide tiene su tiempo objetivo, la idea
> que debe quedar, y un texto sugerido (léelo natural, no de memoria). El
> arco narrativo es: *quise comparar STFT vs CWT → apareció una diferencia
> confusa → descubrí que no comparaba peras con peras (el eje de modulación)
> → lo controlé → las transformadas empatan; la diferencia era un artefacto
> de DSP.* Dos versiones según tu tiempo:
> **(1) Núcleo narrativo** (slides 1–3, 4–7, sin los zooms 3a–3c ni los
> ejemplos 3d–3e): ~5:00 a ritmo ágil, ~8:00 pausado.
> **(2) Versión completa** (con los 3 *zooms* de DSP y los 2 ejemplos de
> gráficas): **~11–13 min a ritmo pausado**, ideal para un público de
> señales o una defensa de ~15 min. El corazón sigue siendo las **slides 5
> y 6**; los zooms/ejemplos son profundidad que puedes comprimir o saltar.

---

## Slide 1 — Título · (0:00–0:15) · 15 s

**Idea**: presentarte y plantar la pregunta en una frase.

> "Buenos días. Mi proyecto pregunta algo simple: para detectar Alzheimer
> con EEG, **¿importa cómo descomponemos la señal** en tiempo y frecuencia?
> Repliqué un pipeline del estado del arte y comparé dos formas de hacerlo."

---

## Slide 2 — La pregunta · (0:15–1:00) · 45 s

**Idea**: por qué EEG, qué es el espectrograma de modulación, y la hipótesis.

> "El Alzheimer es la demencia más común, y detectarlo temprano importa,
> pero resonancia y tomografía por emisión de positrones son caras. El **EEG es barato y portátil**.
>
> La representación que uso se llama **espectrograma de modulación**: no
> mira la frecuencia directamente, sino *a qué ritmo* cambia la energía de
> cada banda. Y todo eso depende del primer paso —cómo hago la
> descomposición tiempo-frecuencia—, ese bloque naranja del diagrama.
>
> El trabajo que replico —**Lopes y colegas, 2023**— hace algo ingenioso
> con esa representación: en vez de entrenar un clasificador directo, usan
> una **red convolucional para *descubrir*** qué zonas del espectro
> distinguen a los pacientes, mediante mapas de saliencia; y luego un
> **SVM** clasifica usando solo esas zonas. Reportan alrededor de un **71%
> de acierto**.
>
> Mi **hipótesis** era que la **CWT**, la transformada wavelet, con mejor
> resolución en bajas frecuencias, debería ganarle a la STFT. El plan:
> replicar ese pipeline sobre **datos públicos** y ver si la CWT de verdad
> gana."

*Transición*: "Primero, lo que construí."

---

## Slide 3 — Lo que construí · (1:00–1:40) · 40 s

**Idea**: el pipeline en un vistazo + el truco del método + las 3 representaciones.

> "El pipeline va de izquierda a derecha: preproceso el EEG, calculo el
> espectrograma, y entreno una CNN.
>
> El truco del método de Lopes es elegante: la CNN **no clasifica**; se usa
> para *descubrir* qué regiones del espectrograma separan enfermos de sanos,
> vía mapas de saliencia. Un SVM final usa esas regiones como *features*.
>
> Comparé **tres representaciones**: STFT, CWT, y una tercera —CWT-fair—
> que es la clave y explico en un momento. Por dos métodos de saliencia,
> por tres semillas. Y todo con **anti-leakage estricto**: cada cosa se
> calcula dentro de cada fold."

*Transición*: "Antes de los resultados, y como es un curso de señales, un
acercamiento a las tres decisiones de DSP clave."

---

## Slide 3a — Zoom DSP: limpieza (FIR + ICA) · (~45 s)

**Idea**: por qué FIR de fase lineal, qué es ICA y por qué el etiquetado automático.

> "Primero, **limpieza**. Filtro pasa-banda de 0.5 a 45 Hz. El 0.5 quita la
> deriva lenta y el DC; el 45 corta la línea eléctrica y el músculo. Y —esto
> importa— uso un **FIR de fase lineal**: no quiero que el filtro distorsione
> el *tiempo*, porque justo después voy a medir *cómo cambia la energía en el
> tiempo*. Luego aplico **ICA** (Análisis de Componentes Independientes):
> separa la mezcla que capta cada electrodo en sus **fuentes
> independientes** —como aislar cada voz en una fiesta—, y así el parpadeo,
> el músculo o la línea salen como componentes propias; un clasificador
> automático, **ICLabel**, marca las no cerebrales para descartarlas. Si no,
> un parpadeo es un pico de potencia que el espectrograma podría confundir
> con un biomarcador."

---

## Slide 3b — Zoom DSP: acondicionamiento (200 Hz + 8 s) · (~35 s)

**Idea**: por qué 200 Hz (aliasing) y por qué 8 s (resolución de modulación).

> "Segundo, **acondicionamiento**. Bajo a **200 Hz**: como ya filtré a 45,
> con un Nyquist de 100 sobra y evito aliasing, con menos datos. Y corto en
> **épocas de 8 segundos**. El 8 no es arbitrario: la resolución del eje de
> modulación es 1 sobre la duración, o sea **0.125 Hz** —fina para las
> modulaciones lentas, por debajo de 2 Hz, que son las que importan en
> Alzheimer—. Más corto perdería esa resolución; más largo rompería la
> estacionariedad del EEG."

---

## Slide 3c — Zoom DSP: representación T-F + modspec · (~40 s)

**Idea**: STFT vs CWT es la variable en estudio; y siembra la sutileza del eje.

> "Y tercero, la **representación**, el corazón del proyecto. Descompongo con
> STFT o con CWT: la STFT usa una **ventana fija**; la wavelet **ajusta la
> ventana por escala**, dando mejor resolución en bajas frecuencias —de ahí
> mi hipótesis—. De ahí saco el espectro de modulación y lo llevo a una
> imagen de **45×45** para la CNN. Y dejo sembrada una sutileza: forzar las
> dos a 45×45 **oculta que su eje vertical no mide lo mismo**. Justo eso es
> lo que voy a destapar en un momento."

*Transición*: "¿Y qué imagen sale de todo esto?"

---

## Slide 3d — Ejemplo real: el espectro de modulación · (~45 s)

**Idea**: mostrar la gráfica real que se obtiene y señalar el biomarcador.

> "Y esto es lo que sale: un **ejemplo real** del espectro de modulación.
> En el eje vertical, la frecuencia de la señal, con las bandas clásicas
> —delta, theta, alfa, beta—; en el horizontal, la frecuencia de
> **modulación**, o sea *a qué ritmo* cambia la energía. Cada sujeto produce
> una imagen así.
>
> Lo interesante es el panel de la derecha: la **diferencia entre pacientes
> y sanos**. En Alzheimer sube la actividad **lenta** —el delta, en rojo— y
> baja el alfa —en azul—. Eso, en una sola imagen, es el biomarcador que el
> modelo va a aprender a separar."

*Transición*: "¿Y cómo lo aprende?"

---

## Slide 3e — Ejemplo real: mapas de saliencia · (~40 s)

**Idea**: el truco de Lopes en imágenes; y sembrar que STFT y CWT difieren.

> "Aquí está el **truco de Lopes en imágenes**. La red no da un diagnóstico:
> genera un **mapa de saliencia** que resalta las regiones del espectro que
> separan enfermos de sanos —las zonas más claras—. Y un SVM final clasifica
> usando solo esas regiones.
>
> Fíjense en un detalle: la **STFT y la CWT resaltan patrones distintos**.
> Esa pista es justo la que me llevó al giro que viene."

*Transición*: "Con eso montado, ¿qué salió en números?"

---

## Slide 4 — El titular · (1:40–2:25) · 45 s

**Idea**: la replicación funciona; y aparece una diferencia que cambia de signo.

> "La buena noticia: la **replicación funciona**. La mejor configuración
> —SVM vainilla con STFT, la fiel al paper— llega a una **exactitud de
> 0.764, en el rango del 0.71 del paper**, sobre datos públicos (y un AUC
> de 0.856, que el paper no reporta). Eso ya justifica el trabajo.
>
> Pero apareció algo curioso: la comparación STFT vs CWT **cambia de signo**
> según el método de saliencia. Con vainilla, la STFT parece mejor; con
> Grad-CAM, la CWT parece mejor. Eso es sospechoso —si una transformada
> fuera realmente mejor, no debería depender de ese detalle—.
>
> Así que antes de concluir nada, miré el primer paso con lupa."

*Transición (clave, baja el tono)*: "Y ahí vino el giro."

---

## Slide 5 — El giro · (2:25–3:25) · 60 s · ⭐ CORAZÓN

**Idea**: las dos imágenes 45×45 no miden lo mismo en el eje de modulación; la solución es CWT-fair.

> "Resulta que **no estaba comparando peras con peras**. Las dos imágenes
> son 45 por 45 y se ven equivalentes, pero su **eje vertical —la frecuencia
> de modulación— no mide lo mismo**. Por cómo funciona cada transformada, la
> STFT llega hasta 1.56 Hz de modulación con unos 13 valores reales; la CWT
> llega hasta 100 Hz con unos 180. Ambas se estiran o encogen a 45 píxeles,
> pero el contenido físico es distinto.
>
> Entonces una comparación STFT vs CWT mezcla **dos cosas**: la transformada
> en sí, y la resolución de ese eje de modulación.
>
> Para separarlas construí una tercera condición: **CWT-fair**. Es la
> **misma CWT**, pero le igualo el eje de modulación al de la STFT. Así,
> comparar STFT contra CWT-fair aísla el efecto de la **transformada**; y
> comparar la CWT nativa contra CWT-fair mide el efecto del **eje** —o sea,
> el artefacto—."

*Transición*: "¿Y qué pasó al igualar el eje?"

---

## Slide 6 — El resultado · (3:25–4:25) · 60 s · ⭐ CORAZÓN

**Idea**: al igualar el eje, la CWT se mueve hacia la STFT en ambas direcciones; empatan.

> "Esto es lo importante. Al igualar el eje de modulación, la CWT **se mueve
> hacia la STFT en las dos direcciones**.
>
> Con vainilla, la CWT estaba 0.078 de AUC por debajo de la STFT; con el eje
> igualado, queda a solo 0.028 —una reducción de ~64%—. Con Grad-CAM, estaba
> 0.087 por encima; queda a 0.056 —~36%—. En los dos casos, igualar el eje
> mueve la CWT hacia la STFT (más fuerte en vainilla que en Grad-CAM).
>
> Y estadísticamente: el test de DeLong entre STFT y CWT-fair da p de 0.32 y
> 0.59 —no significativo—, y nada sobrevive la corrección por comparaciones
> múltiples.
>
> O sea: **a igualdad de eje de modulación, la STFT y la CWT-Morlet son
> indistinguibles**. La diferencia que veía —en ambos sentidos— era **en
> gran parte un artefacto de procesamiento de señales**, no la transformada.
> Ese fue el hallazgo central."

*Transición*: "Para cerrar."

---

## Slide 7 — Lo que me llevo + Gracias · (4:25–5:00) · 35 s

**Idea**: tres takeaways, reproducibilidad, cierre.

> "Me llevo tres cosas. **Uno**: la replicación funciona sobre datos
> públicos. **Dos**, la lección metodológica: **comparar transformadas
> tiempo-frecuencia exige igualar el eje de modulación**; sin ese control,
> un artefacto puede hasta invertir el ranking. Y **tres**: el aporte real
> no es un número, sino identificar el confound, diseñar el control, y
> demostrar que la diferencia era un artefacto.
>
> Todo está en un repo público, con el informe y el código documentados. Lo
> primero a futuro es igualar el eje 'hacia arriba' y ver si las
> modulaciones rápidas de la CWT aportan algo.
>
> **Gracias** —quedo atento a sus preguntas."

---

## Preguntas probables (Q&A) — respuestas de bolsillo

- **"¿Entonces la CWT no sirve?"**
  No es eso: sirve *igual* que la STFT una vez comparadas de forma justa. Lo
  que muestro es que la diferencia aparente era un artefacto del eje de
  modulación, no una ventaja/desventaja real de la transformada.

- **"¿Qué es exactamente CWT-fair?"**
  La misma CWT-Morlet (misma resolución de frecuencia portadora), pero
  decimando la envolvente de potencia a 3.125 Hz antes de la FFT de
  modulación, para que el eje de modulación tenga el mismo Nyquist (1.56 Hz)
  y el mismo número de bins reales que la STFT.

- **"¿Por qué la STFT tiene ese Nyquist tan bajo?"**
  Porque su paso temporal es 0.32 s (hop de 64 muestras a 200 Hz). La CWT
  conserva los 200 Hz. Al forzar ambas a 45×45, quedan midiendo rangos
  físicos distintos.

- **"CWT-fair descarta información de la CWT, ¿no la perjudica?"**
  Iguala 'hacia abajo', sí. Es la elección conservadora y fisiológicamente
  razonable (la modulación diagnóstica en Alzheimer es lenta, <2 Hz). La
  alternativa —subir la STFT con hop fino— la dejo como trabajo futuro.

- **"¿Por qué solo 3 seeds?"**
  Costo de cómputo (GPU de 4 GB). Con 3 ya se ve la varianza; la inferencia
  es DeLong por seed + combinación. Idealmente 10+, queda como futuro.

- **"¿Es usable clínicamente?"**
  No. Replicación académica sobre datos públicos, sin validación externa ni
  aprobación regulatoria.

---

## Notas de entrega

- **Ritmo y duración**: el discurso completo son ~1290 palabras. A ritmo **pausado** (~110 wpm) son ~11:45 de lectura pura, o **~13:30 en vivo** (con pausas al cambiar de slide, señalar figuras y respirar). El núcleo narrativo (sin 3a–3e) son ~830 palabras: ~7:30 pausado, ~5:00 ágil. Ensaya con cronómetro y decide qué versión usas.
- **Zooms de DSP (3a–3c)**: suman ~2:00. Son para el público de señales de la materia. Si el tiempo aprieta, compríme a un solo *zoom* de ~40 s ("filtro FIR de fase lineal, 200 Hz por aliasing, 8 s por resolución de modulación") o sáltalos; el arco no depende de ellos.
- **Ejemplos de gráficas (3d–3e)**: suman ~1:25. La 3d (espectro de modulación) es la más valiosa para un curso de señales —muestra la entrada real y el biomarcador—; consérvala si puedes. La 3e (saliencia) es opcional: si vas justo, resúmela en una frase ("la CNN resalta las regiones que separan AD de sanos y el SVM las usa") apoyándote en el diagrama de la slide 3.
- **Dónde respirar**: las transiciones en *cursiva* son pausas — úsalas.
- **Slides que no puedes saltar**: la 5 (el confound + CWT-fair) y la 6 (la
  convergencia). Son el argumento central.
- **Si te falta tiempo**: en slide 4 di solo "la replicación funciona" y
  salta el detalle del cambio de signo; recupéralo en la slide 6.
- **Si te sobra tiempo**: en slide 6 menciona que también la CNN sola
  muestra el mismo patrón (CWT nativa 0.590 → CWT-fair 0.667, hacia STFT
  0.695).
