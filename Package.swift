// swift-tools-version: 5.9
import PackageDescription
import Foundation

// gist: on-device content topic tagging for every platform.
//
//   desert-ant-core          reusable primitives (TextNormalization, JSON,
//                            ModelStore, Inference sessions + platform factory)
//   Sources/Gist             shared pipeline (pure Swift; platform variation is
//                            data: which model artifact ships where)
//   Sources/GistWeb          wasm entry point -> packages/gist-node
//
// The two-stream pipeline (potion embedding lookup + hashed n-grams -> MLP head)
// runs the head through the shared InferenceSession (LiteRT / Core ML / JS host);
// tokenization, embedding pooling, and n-gram hashing are pure Swift.
let noJavaScriptKit = ProcessInfo.processInfo.environment["SWIFT_ANDROID_STATIC_BUILD"] != nil

let jsDependencies: [Package.Dependency] = noJavaScriptKit ? [] : [
    .package(url: "https://github.com/swiftwasm/JavaScriptKit", from: "0.56.1"),
]
let wasmProducts: [Product] = noJavaScriptKit ? [] : [
    .executable(name: "GistWeb", targets: ["GistWeb"]),
]
let wasmTargets: [Target] = noJavaScriptKit ? [] : [
    .executableTarget(name: "GistWeb", dependencies: ["Gist"] + [
        .product(name: "JavaScriptKit", package: "JavaScriptKit", condition: .when(platforms: [.wasi])),
        .product(name: "JavaScriptEventLoop", package: "JavaScriptKit", condition: .when(platforms: [.wasi])),
    ]),
]

let package = Package(
    name: "Gist",
    platforms: [.iOS(.v16), .macOS(.v13), .tvOS(.v16), .visionOS(.v1)],
    products: [
        .library(name: "Gist", targets: ["Gist"]),
    ] + wasmProducts,
    dependencies: [
        .package(url: "https://github.com/Desert-Ant-Labs/desert-ant-core.git", from: "0.2.4"),
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
    ] + jsDependencies,
    targets: [
        .target(
            name: "Gist",
            dependencies: [
                .product(name: "TextNormalization", package: "desert-ant-core"),
                .product(name: "JSON", package: "desert-ant-core"),
                .product(name: "ModelStore", package: "desert-ant-core"),
                .product(name: "PlatformSupport", package: "desert-ant-core"),
                .product(name: "Inference", package: "desert-ant-core"),
                .product(name: "RealModule", package: "swift-numerics"),
            ]
        ),
        .testTarget(
            name: "GistTests",
            dependencies: ["Gist"],
            resources: [
                .copy("Resources/gist_tokenizer.bin"),
                .copy("Resources/gist-sdk-oracle.json"),
                .copy("Resources/gist_embedding.i8"),
                .copy("Resources/gist_embedding.json"),
                .copy("Resources/gist-feature-oracle.json"),
            ]
        ),
    ] + wasmTargets
)
