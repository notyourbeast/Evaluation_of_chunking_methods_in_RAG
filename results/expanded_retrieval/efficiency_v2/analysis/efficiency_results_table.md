# Expanded efficiency benchmark

Values are observed descriptive summaries on one machine. Preparation uses three complete trials; warm-query latency uses 300 observations (100 deterministic sampled queries × 3 randomised orders); fresh-process latency uses three observations. Fresh-process timing is not hardware-cold timing.

| dataset | strategy | chunks_median | chunks_p10 | chunks_p90 | preparation_median_seconds | preparation_p10_seconds | preparation_p90_seconds | serialised_faiss_index_bytes_median | warm_total_median_seconds | warm_total_p10_seconds | warm_total_p90_seconds | fresh_process_total_median_seconds | fresh_process_total_p10_seconds | fresh_process_total_p90_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arguana | fixed | 12074.0 | 12074.0 | 12074.0 | 68.064579 | 67.460282 | 68.794178 | 18545709.0 | 0.017284 | 0.012206 | 0.025467 | 0.576667 | 0.527499 | 1.341582 |
| arguana | semantic | 18910.0 | 18910.0 | 18910.0 | 347.378382 | 346.939536 | 349.160650 | 29045805.0 | 0.019191 | 0.013681 | 0.029950 | 0.627794 | 0.604490 | 1.082995 |
| arguana | sentence | 11124.0 | 11124.0 | 11124.0 | 288.299504 | 286.883550 | 290.535255 | 17086509.0 | 0.019824 | 0.012941 | 0.070066 | 0.554278 | 0.463165 | 1.448338 |
| arguana | topic | 17204.0 | 17184.0 | 17226.4 | 387.233259 | 386.532398 | 387.320262 | 26425389.0 | 0.019157 | 0.013953 | 0.027457 | 0.546377 | 0.524684 | 1.365459 |
| nfcorpus | fixed | 7512.0 | 7512.0 | 7512.0 | 49.759882 | 49.638566 | 49.833621 | 11538477.0 | 0.009096 | 0.007605 | 0.011091 | 0.575201 | 0.539110 | 1.278728 |
| nfcorpus | semantic | 11888.0 | 11888.0 | 11888.0 | 241.833139 | 241.051703 | 241.892999 | 18260013.0 | 0.010340 | 0.009358 | 0.012610 | 0.607226 | 0.550162 | 1.129565 |
| nfcorpus | sentence | 7034.0 | 7034.0 | 7034.0 | 183.895207 | 183.562549 | 188.437488 | 10804269.0 | 0.009316 | 0.008185 | 0.011647 | 0.555802 | 0.495500 | 1.296561 |
| nfcorpus | topic | 10507.0 | 10488.6 | 10511.0 | 282.191543 | 282.016176 | 285.404916 | 16138797.0 | 0.010072 | 0.009032 | 0.012197 | 0.631476 | 0.539862 | 1.333270 |
| scifact | fixed | 10400.0 | 10400.0 | 10400.0 | 69.291900 | 69.063193 | 70.133334 | 15974445.0 | 0.009820 | 0.009140 | 0.016770 | 0.621477 | 0.529879 | 1.014766 |
| scifact | semantic | 15864.0 | 15864.0 | 15864.0 | 307.135542 | 306.281391 | 307.293913 | 24367149.0 | 0.010796 | 0.010218 | 0.014538 | 0.508251 | 0.503624 | 1.052551 |
| scifact | sentence | 9550.0 | 9550.0 | 9550.0 | 241.816998 | 241.802761 | 249.446850 | 14668845.0 | 0.009698 | 0.008864 | 0.069481 | 0.530789 | 0.499410 | 0.944123 |
| scifact | topic | 13147.0 | 13147.0 | 13147.0 | 332.480426 | 332.274968 | 355.755047 | 20193837.0 | 0.010308 | 0.009591 | 0.014462 | 0.554315 | 0.492513 | 1.202812 |
