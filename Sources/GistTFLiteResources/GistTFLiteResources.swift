import Foundation

/// Bundle accessor for the LiteRT (`.tflite`) head plus the shared sidecars.
/// Used by Linux and Windows builds when the `BundledModel` trait is enabled
/// (Android bundles the optional `:gist-tflite-resources` Gradle artifact, and
/// wasm always downloads); Apple platforms use `GistCoreMLResources` instead.
///
/// ```swift
/// import GistTFLiteResources
/// let gist = Gist(bundle: GistTFLiteResourcesBundle.bundle)
/// ```
public enum GistTFLiteResourcesBundle {
    public static var bundle: Bundle { Bundle.module }
}
