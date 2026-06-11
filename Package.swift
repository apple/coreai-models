// swift-tools-version: 6.0

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import PackageDescription

let package = Package(
    name: "coreai-models",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(
            name: "CoreAILM",
            targets: [
                "CoreAILanguageModels"
            ]
        ),
        .library(
            name: "CoreAIDiffusion",
            targets: [
                "CoreAIDiffusionPipeline"
            ]
        ),
        .library(
            name: "CoreAISegmentation",
            targets: [
                "CoreAIImageSegmenter"
            ]
        ),
        .library(
            name: "CoreAIObjectDetection",
            targets: [
                "CoreAIObjectDetector"
            ]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
        .package(url: "https://github.com/mlc-ai/xgrammar", branch: "main"),
    ],
    targets: [
        .target(
            name: "CoreAILanguageModels",
            dependencies: [
                "CoreAIShared",
                "CXGrammar",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Sources/CoreAILanguageModels",
            swiftSettings: [
                .define("CXGRAMMAR_IMPORT"),
                .enableUpcomingFeature("MemberImportVisibility"),
            ],
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        ),
        .target(
            name: "CoreAIImageSegmenter",
            dependencies: ["CoreAIShared"],
            path: "swift/Sources/CoreAIImageSegmenter",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),
        .target(
            name: "CoreAIObjectDetector",
            dependencies: ["CoreAIShared"],
            path: "swift/Sources/CoreAIObjectDetector",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),

        // Shared utilities
        .target(
            name: "CoreAIShared",
            dependencies: [],
            path: "swift/Sources/CoreAIShared",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),

        // Diffusion Pipeline
        .target(
            name: "CoreAIDiffusionPipeline",
            dependencies: [
                "CoreAIShared",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Sources/CoreAIDiffusionPipeline",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),

        // CXGrammar C bridge
        .target(
            name: "CXGrammar",
            dependencies: [
                .product(name: "XGrammar", package: "xgrammar"),
            ],
            path: "swift/Sources/lib/CXGrammar",
            publicHeadersPath: "include"
        ),

        // MARK: Executable targets

        .executableTarget(
            name: "llm-runner",
            dependencies: [
                "CoreAILanguageModels",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "swift/Sources/Tools/llm-runner",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),
        .executableTarget(
            name: "image-segmenter",
            dependencies: [
                "CoreAIImageSegmenter",
                "CoreAIShared",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "swift/Sources/Tools/image-segmenter",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),
        .executableTarget(
            name: "object-detector",
            dependencies: [
                "CoreAIObjectDetector",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "swift/Sources/Tools/object-detector",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),
        .executableTarget(
            name: "diffusion-runner",
            dependencies: [
                "CoreAIDiffusionPipeline",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "swift/Sources/Tools/diffusion-runner",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),

        // Public LLM Benchmark CLI (based on mlx-lm benchmark)
        .executableTarget(
            name: "llm-benchmark",
            dependencies: [
                "CoreAILanguageModels",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "swift/Sources/Tools/benchmark",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),

        // MARK: Test targets

        .target(
            name: "TestUtilities",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "swift/Tests/TestUtilities",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),
        .testTarget(
            name: "LanguageModelsTests",
            dependencies: [
                "CoreAILanguageModels",
                "TestUtilities",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Tests/LanguageModelsTests",
            resources: [
                .copy("Resources/MinimalTokenizer")
            ],
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        ),
        .testTarget(
            name: "ImageSegmenterTests",
            dependencies: [
                "CoreAIImageSegmenter",
                "TestUtilities",
            ],
            path: "swift/Tests/ImageSegmenterTests"
        ),
        .testTarget(
            name: "DiffusionPipelineTests",
            dependencies: [
                "CoreAIDiffusionPipeline",
                "TestUtilities",
            ],
            path: "swift/Tests/DiffusionPipelineTests"
        ),
        .testTarget(
            name: "ObjectDetectorTests",
            dependencies: ["CoreAIObjectDetector"],
            path: "swift/Tests/ObjectDetectorTests"
        ),
        .testTarget(
            name: "CoreAISharedTests",
            dependencies: ["CoreAIShared", "TestUtilities"],
            path: "swift/Tests/CoreAISharedTests"
        ),
        .testTarget(
            name: "GuidedGenerationTests",
            dependencies: [
                "CoreAILanguageModels",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/Tests/GuidedGenerationTests",
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        ),
    ],
    swiftLanguageModes: [.v6],
    cxxLanguageStandard: .cxx17
)
