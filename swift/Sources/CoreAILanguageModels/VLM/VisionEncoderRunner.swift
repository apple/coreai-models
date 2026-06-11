// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Vision Encoder Runner for Qwen3-VL.
// Loads a vision .aimodel and runs inference to transform pixel patches into visual features.

import CoreAI
import CoreAIShared
import Foundation



// MARK: - Vision Encoder Runner

/// Runs a vision encoder model to transform pixel patches into visual features.
///
/// ## Model Signature
/// - Input: `pixel_values: f32[784, 1536]`
/// - Output: `image_features: f32[196, 2048]` (converted to f16)
#if canImport(CoreAI)
@available(macOS 27, iOS 27, *)
public class VisionEncoderRunner: @unchecked Sendable {
    private let function: InferenceFunction
    private let inputName: String
    private let outputName: String
    private let inputDescriptor: NDArrayDescriptor
    private let outputDescriptor: NDArrayDescriptor

    public init(assetURL: URL) async throws {
        let model = try await AIModel(contentsOf: assetURL, options: .default)

        guard let descriptor = model.functionDescriptor(for: "main") else {
            throw InferenceRuntimeError.functionNotFound("main in vision encoder")
        }
        guard descriptor.inputNames.count >= 1, descriptor.outputNames.count >= 1 else {
            throw InferenceRuntimeError.genericError("vision encoder needs 1+ input and output")
        }

        self.inputName = descriptor.inputNames[0]
        self.outputName = descriptor.outputNames[0]

        guard case .ndArray(let inDesc) = descriptor.inputDescriptor(of: inputName) else {
            throw InferenceRuntimeError.genericError("Cannot get input descriptor")
        }
        self.inputDescriptor = inDesc

        guard case .ndArray(let outDesc) = descriptor.outputDescriptor(of: outputName) else {
            throw InferenceRuntimeError.genericError("Cannot get output descriptor")
        }
        self.outputDescriptor = outDesc

        guard let fn = try model.loadFunction(named: "main") else {
            throw InferenceRuntimeError.functionNotFound("main in vision encoder")
        }
        self.function = fn
    }

    /// Run vision encoder on preprocessed pixel values [784, 1536] float32.
    /// Returns [numVisualTokens * hiddenSize] Float16 values.
    public func run(
        pixelValues: Data,
        numVisualTokens: Int = 196,
        hiddenSize: Int = 2048
    ) async throws -> [LogitsScalarType] {
        let inputShape = inputDescriptor.shape
        let resolvedInputDesc = inputDescriptor.resolvingDynamicDimensions(inputShape)
        var inputArray = NDArray(descriptor: resolvedInputDesc)

        pixelValues.withUnsafeBytes { src in
            var view = inputArray.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                let count = pixelValues.count / MemoryLayout<Float>.size
                src.baseAddress!.assumingMemoryBound(to: Float.self)
                    .withMemoryRebound(to: Float.self, capacity: count) { srcPtr in
                        ptr.update(from: srcPtr, count: count)
                    }
            }
        }

        let outputShape = outputDescriptor.shape
        let resolvedOutputDesc = outputDescriptor.resolvingDynamicDimensions(outputShape)
        var outputArray = NDArray(descriptor: resolvedOutputDesc)

        var outputBackings = InferenceFunction.MutableViews()
        outputBackings.insert(&outputArray, for: outputName)

        _ = try await function.run(
            inputs: [inputName: inputArray],
            states: InferenceFunction.MutableViews(),
            outputViews: consume outputBackings
        )

        let totalElements = numVisualTokens * hiddenSize
        var features = [LogitsScalarType](repeating: 0, count: totalElements)
        outputArray.view(as: Float.self).withUnsafePointer { ptr, _, _ in
            for i in 0..<totalElements {
                features[i] = LogitsScalarType(ptr[i])
            }
        }
        return features
    }
}



#endif
