// swift-tools-version: 6.1
import PackageDescription
import Foundation

// gist: on-device content topic tagging for every platform.
//
//   desert-ant-core            reusable primitives (JSON, ModelStore,
//                              TextNormalization, Inference sessions + factory)
//   Sources/Gist               shared pipeline (pure Swift; platform variation is
//                              data: which artifact ships where, no tensor branching)
//   Sources/GistCoreMLResources  Apple/Core ML head + shared sidecars (optional bundle)
//   Sources/GistTFLiteResources  LiteRT (.tflite) head + shared sidecars for Linux/Windows
//   Sources/GistAndroid        C ABI + Swift JNI -> packages/gist-kotlin (+ Node native)
//   Sources/GistWeb            wasm entry point -> packages/gist-node
//
// The two-stream pipeline (potion embedding lookup + hashed n-grams -> MLP head)
// runs the head through the shared InferenceSession; tokenization, embedding
// pooling, and n-gram hashing are pure Swift. gist is a ~74 MB model, so unlike
// emo it downloads on demand by default; enable the BundledModel trait to ship
// the resources for a fully offline app.
let appleResourcePlatforms: [Platform] = [.macOS, .macCatalyst, .iOS, .tvOS, .watchOS, .visionOS]
let bundledModelTrait = Trait(
    name: "BundledModel",
    description: "Bundle the gist model into the Swift package product for fully offline apps. Off by default (gist is ~74 MB); the default downloads on demand."
)
let packageTraits: Set<Trait> = [
    .default(enabledTraits: []),   // download on demand by default
    bundledModelTrait,
]

// The Android static-stdlib link needs no macros in the build graph, so this
// flag (set by the shared catalog's android-natives task) drops JavaScriptKit
// and the wasm entry point. The wasm/JS code is all `#if os(WASI)`, so it is
// absent off-wasm anyway.
let noJavaScriptKit = ProcessInfo.processInfo.environment["SWIFT_ANDROID_STATIC_BUILD"] != nil

let jsDependencies: [Package.Dependency] = noJavaScriptKit ? [] : [
    .package(url: "https://github.com/swiftwasm/JavaScriptKit", from: "0.56.1"),
]
let packageDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/Desert-Ant-Labs/desert-ant-core.git", from: "0.5.3"),
    .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
] + jsDependencies

let wasmProducts: [Product] = noJavaScriptKit ? [] : [
    .executable(name: "GistWeb", targets: ["GistWeb"]),
]
let packageProducts: [Product] = [
    .library(name: "Gist", targets: ["Gist"]),
    // Optional resource bundles for offline apps (BundledModel trait).
    .library(name: "GistCoreMLResources", targets: ["GistCoreMLResources"]),
    .library(name: "GistTFLiteResources", targets: ["GistTFLiteResources"]),
    // Android JNI library (built by the catalog's android-natives task).
    .library(name: "GistAndroid", type: .dynamic, targets: ["GistAndroid"]),
    // Native library for the Node.js server-side backend. Shares the GistAndroid
    // target: on a host (Linux/macOS) triple only the C ABI in `CABI.swift`
    // compiles (`AndroidJNI.swift` is `#if os(Android)`); koffi in
    // packages/gist-node binds the `gist_*` C ABI over the resulting libGistNode.
    .library(name: "GistNode", type: .dynamic, targets: ["GistAndroid"]),
] + wasmProducts

let gistDependencies: [Target.Dependency] = [
    .product(name: "JSON", package: "desert-ant-core"),
    .product(name: "ModelStore", package: "desert-ant-core"),
    .product(name: "TextNormalization", package: "desert-ant-core"),
    .product(name: "PlatformSupport", package: "desert-ant-core"),
    .product(name: "ModelResources", package: "desert-ant-core"),
    .product(name: "Inference", package: "desert-ant-core"),
    .product(name: "RealModule", package: "swift-numerics"),
    .target(name: "GistCoreMLResources", condition: .when(platforms: appleResourcePlatforms, traits: ["BundledModel"])),
    .target(name: "GistTFLiteResources", condition: .when(platforms: [.linux, .windows], traits: ["BundledModel"])),
]

let gistTarget: Target = .target(
    name: "Gist",
    dependencies: gistDependencies,
    swiftSettings: [.define("GIST_BUNDLED_MODEL", .when(traits: ["BundledModel"]))]
)

let resourceTargets: [Target] = [
    .target(name: "GistCoreMLResources", resources: [
        .copy("Resources/gist.mlmodelc"), .copy("Resources/gist_embedding.i8"),
        .copy("Resources/gist_embedding.json"), .copy("Resources/gist_config.json"),
        .copy("Resources/gist_tokenizer.bin"), .copy("Resources/taxonomy.json"),
    ]),
    .target(name: "GistTFLiteResources", resources: [
        .copy("Resources/gist.tflite"), .copy("Resources/gist_embedding.i8"),
        .copy("Resources/gist_embedding.json"), .copy("Resources/gist_config.json"),
        .copy("Resources/gist_tokenizer.bin"), .copy("Resources/taxonomy.json"),
    ]),
]

let androidTarget: Target = .target(
    name: "GistAndroid",
    dependencies: [
        "Gist",
        .product(name: "FFIBuffer", package: "desert-ant-core"),
        .product(name: "HostBridge", package: "desert-ant-core", condition: .when(platforms: [.android])),
        .product(name: "ModelStore", package: "desert-ant-core", condition: .when(platforms: [.android])),
        .product(name: "PlatformSupport", package: "desert-ant-core"),
    ]
)

let testTarget: Target = .testTarget(
    name: "GistTests",
    dependencies: [
        "Gist",
        .target(name: "GistTFLiteResources", condition: .when(platforms: [.linux, .windows])),
    ],
    resources: [
        .copy("Resources/gist_tokenizer.bin"),
        .copy("Resources/gist_embedding.i8"),
        .copy("Resources/gist_embedding.json"),
        .copy("Resources/gist-sdk-oracle.json"),
        .copy("Resources/gist-feature-oracle.json"),
    ]
)

let wasmTargets: [Target] = noJavaScriptKit ? [] : [
    .executableTarget(
        name: "GistWeb",
        dependencies: [
            "Gist",
            .product(name: "JavaScriptKit", package: "JavaScriptKit", condition: .when(platforms: [.wasi])),
            .product(name: "JavaScriptEventLoop", package: "JavaScriptKit", condition: .when(platforms: [.wasi])),
        ],
        swiftSettings: [.swiftLanguageMode(.v5)]
    ),
]

let package = Package(
    name: "Gist",
    platforms: [.iOS(.v16), .macOS(.v13), .macCatalyst(.v16), .tvOS(.v16), .watchOS(.v9), .visionOS(.v1)],
    products: packageProducts,
    traits: packageTraits,
    dependencies: packageDependencies,
    targets: [gistTarget] + resourceTargets + [androidTarget, testTarget] + wasmTargets
)
