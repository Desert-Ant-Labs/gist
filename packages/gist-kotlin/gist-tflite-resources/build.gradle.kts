// Optional bundled model for Gist on Android; the ai.desertant.model-resources
// convention plugin packages the LiteRT files staged by `mise run android-natives`.
plugins { id("ai.desertant.model-resources") }
version = "2.1.0"
desertAntResources {
    tfliteFiles = listOf(
        "gist.tflite", "gist_config.json", "gist_embedding.i8",
        "gist_embedding.json", "gist_tokenizer.bin", "taxonomy.json",
    )
}
