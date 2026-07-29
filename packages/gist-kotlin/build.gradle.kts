// Android library (AAR) for gist. The AGP/Kotlin/publish boilerplate and the Swift
// native build wiring live in the shared ai.desertant.model-sdk convention plugin
// (published from desert-ant-core); this file supplies only gist's version and
// description. `mise run build-android` -> `mise run android-natives` builds the
// prebuilt Swift JNI into src/main/jniLibs before packaging.
plugins { id("ai.desertant.model-sdk") version "0.4.2" }
version = "2.0.1"
desertAntSdk {
    description = "On-device multilingual content topic tagging for Android: turns a title, post, or " +
        "description into ranked topics from a 36-topic taxonomy across 101 languages, fully on device."
}
