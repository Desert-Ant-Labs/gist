# Third-party notices - Emo

The `emo` model is trained on synthetic and open data under licenses permitting
commercial use and derivative works. The semantic stream is built on an open,
distilled multilingual embedding.

## Model components
- **potion-multilingual-128M** (`minishlab/potion-multilingual-128M`) - a
  distilled static multilingual embedding (model2vec), used to initialize the
  pruned semantic table. **MIT**.
- Synthetic and LLM-labeled phrase/emoji data generated in the private
  `emo-training` repository. Emoji labels follow the Unicode emoji data.

No non-commercial or unlicensed data is used.

## Android platform libraries

Android Unicode NFKC normalization uses the platform's ICU (`libicu`, API 31+)
through desert-ant-core's `TextNormalization`. JSON parsing uses the Kotlin
host's native JSON through the JNI host, and on-demand model download uses the
host's HTTP stack. On-device inference uses LiteRT (`libLiteRt.so`). No JSON,
HTTP, or ICU library is vendored or hand-rolled in the native layer.
