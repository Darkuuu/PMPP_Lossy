Also nach meinem Verständnis müssen wir folgende Schritte/Funktionen in CUDA übertragen

- generell:
    - evtl später das Bilden der Blöcke auf Grafikkarte parallelisieren (sonst limitiert Prozessor bei der Erstellung der Blöcke die Geschwindigkeit)
        - Zeile 186 ff. in RendererRBUC8x8.cu
        - Aufruf der Compress Funktion in Zeile 298

- für Compress-Funktion (Zeile 1224 in RendererRBUC8x8.cu)
    - compress_internal1 (Zeile 1079, bestimmt Minimum und Maximum der Daten)
        - getMin und getMax-Funktionen
    - setMinMax (Zeile 608 - 642, setzt Minimum und Maximum (für 4D Vektordaten besonders))
    - compress_internal2 (Zeile 1090 ff., berechnet alle Transformationen für jeden Pixel und bestimmt bestes Ergebnis)
        - Transformationsfunktionen (siehe project_structure.md für Zeilenangaben)
            - Gradient: getAvg, predict, encode
            - Haar: getAvg
            - SubMin
            - SubMax
            - Swizzle-Funktionen

- für Decompress-Funktion (Zeile ?)
    - ?