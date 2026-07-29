import Foundation

/// Bundle accessor for the Core ML (`.mlmodelc`) head plus the shared sidecars.
/// Used by Apple builds when the `BundledModel` trait is enabled; other
/// platforms use `GistTFLiteResources` (Linux/Windows) or download (Android/web).
///
/// ```swift
/// import GistCoreMLResources
/// let gist = Gist(bundle: GistCoreMLResourcesBundle.bundle)
/// ```
public enum GistCoreMLResourcesBundle {
    public static var bundle: Bundle { Bundle.module }
}
