// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Core AI diffusion model function — manages a single InferenceFunction
/// for stateless model evaluation (text encoder, UNet, VAE).
public actor CoreAIDiffusionModelFunction {
    private let modelURL: URL
    private var model: AIModel?
    private var function: InferenceFunction?
    private var isLoaded = false

    public init(modelURL: URL) {
        self.modelURL = modelURL
    }

    // MARK: - ResourceManaging

    public func loadResources() async throws {
        guard !isLoaded else { return }

        let options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        let loadedModel = try await AIModel(contentsOf: modelURL, options: options)
        guard let fn = try loadedModel.loadFunction(named: "main") else {
            throw CoreAIDiffusionError.functionNotFound("main", modelURL)
        }

        self.model = loadedModel
        self.function = fn
        self.isLoaded = true
    }

    public func unloadResources() {
        function = nil
        model = nil
        isLoaded = false
    }

    // MARK: - [Float]-based API

    /// Each input is either a pre-built NDArray (for cached constants) or a (Float, shape) pair.
    public enum ModelInput: Sendable {
        case floats([Float], [Int])
        case cached(NDArray)
    }

    /// Pre-pack a Float array into an NDArray matching the model's input descriptor at a given index.
    public func prepackInput(index: Int, data: [Float], shape: [Int]) async throws -> NDArray {
        let fn = try await ensureLoaded()
        let names = fn.descriptor.inputNames
        guard index < names.count else {
            throw CoreAIDiffusionError.functionNotFound("index \(index)", modelURL)
        }
        let name = names[index]
        guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else {
            throw CoreAIDiffusionError.functionNotFound(name, modelURL)
        }
        let resolved = nd.resolvingDynamicDimensions(shape)
        var array = NDArray(descriptor: resolved)
        Self.packFloatsIntoNDArray(&array, data: data, scalarType: resolved.scalarType)
        return array
    }

    /// Reject a supplied buffer whose element count doesn't match its descriptor.
    ///
    /// The fills below copy `data.count` elements into an array sized by the
    /// descriptor, so a short buffer would leave the tail zeroed and produce silently
    /// wrong output rather than an error.
    private static func requireExactCount(
        _ count: Int, _ descriptor: NDArrayDescriptor, input name: String
    ) throws {
        // Skip when dimensions are still dynamic — there is no expected count yet.
        guard descriptor.shape.allSatisfy({ $0 > 0 }) else { return }
        let expected = descriptor.shape.reduce(1, *)
        if count != expected {
            throw CoreAIDiffusionError.inputCountMismatch(
                name: name, shape: descriptor.shape, expected: expected, got: count)
        }
    }

    public func run(floatInputs: [([Float], [Int])]) async throws -> [Float] {
        try await run(inputs: floatInputs.map { .floats($0.0, $0.1) })
    }

    public func run(inputs: [ModelInput]) async throws -> [Float] {
        let fn = try await ensureLoaded()

        var namedInputs: [String: NDArray] = [:]
        for (i, name) in fn.descriptor.inputNames.enumerated() where i < inputs.count {
            switch inputs[i] {
            case .cached(let array):
                namedInputs[name] = array
            case .floats(let data, let shape):
                guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else { continue }
                let resolved = nd.resolvingDynamicDimensions(shape)
                var array = NDArray(descriptor: resolved)
                Self.packFloatsIntoNDArray(&array, data: data, scalarType: resolved.scalarType)
                namedInputs[name] = array
            }
        }

        return try await encodeAndSync(fn: fn, inputs: namedInputs)
    }

    private static func packFloatsIntoNDArray(
        _ array: inout NDArray, data: [Float], scalarType: NDArray.ScalarType
    ) {
        switch scalarType {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            let view = array.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr, shape, strides in
                fillStrided(ptr, data: data, shape: shape, strides: strides) { Float16($0) }
            }
        case .bfloat16:
            array.mutableRawView().withUnsafeMutableBytes { ptr, shape, strides in
                let dst = ptr.assumingMemoryBound(to: UInt16.self)
                fillStrided(dst, data: data, shape: shape, strides: strides) { val in
                    let bits = val.bitPattern
                    let lsb = (bits >> 16) & 1
                    let rounded = bits &+ 0x7FFF &+ lsb
                    return UInt16(truncatingIfNeeded: rounded >> 16)
                }
            }
        #endif
        case .float32:
            let view = array.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { ptr, shape, strides in
                fillStrided(ptr, data: data, shape: shape, strides: strides) { $0 }
            }
        default:
            break
        }
    }

    public func run(intInputs: [([Int32], [Int])]) async throws -> [Float] {
        let fn = try await ensureLoaded()

        var namedInputs: [String: NDArray] = [:]
        for (i, name) in fn.descriptor.inputNames.enumerated() where i < intInputs.count {
            let (data, shape) = intInputs[i]
            guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else { continue }
            let resolved = nd.resolvingDynamicDimensions(shape)
            try Self.requireExactCount(data.count, resolved, input: name)
            var array = NDArray(descriptor: resolved)
            fillNDArray(&array, as: Int32.self, count: data.count) { data[$0] }
            namedInputs[name] = array
        }

        return try await encodeAndSync(fn: fn, inputs: namedInputs)
    }

    // MARK: - NDArray-based API (for parity tests)

    public func predict(inputs: [String: NDArray]) async throws -> [String: [Float]] {
        let fn = try await ensureLoaded()
        try Self.requireSingleOutput(fn)
        let floats = try await encodeAndSync(fn: fn, inputs: inputs)
        return [fn.descriptor.outputNames[0]: floats]
    }

    public func predictAllOutputs(inputs: [String: NDArray]) async throws -> [String: [Float]] {
        let fn = try await ensureLoaded()
        return try await encodeAndSyncAll(fn: fn, inputs: inputs)
    }

    public func predictAutoNamed(inputs: [NDArray]) async throws -> [String: [Float]] {
        let fn = try await ensureLoaded()
        try Self.requireSingleOutput(fn)
        var namedInputs: [String: NDArray] = [:]
        for (i, name) in fn.descriptor.inputNames.enumerated() where i < inputs.count {
            namedInputs[name] = inputs[i]
        }
        let floats = try await encodeAndSync(fn: fn, inputs: namedInputs)
        return [fn.descriptor.outputNames[0]: floats]
    }

    private static func requireSingleOutput(_ fn: InferenceFunction) throws {
        let names = fn.descriptor.outputNames
        if names.count != 1 {
            throw CoreAIDiffusionError.expectedSingleOutput(got: names)
        }
    }

    // MARK: - Core inference

    private func ensureLoaded() async throws -> InferenceFunction {
        if function == nil { try await loadResources() }
        guard let fn = function else { throw CoreAIDiffusionError.notLoaded }
        return fn
    }

    /// Run inference and return the output as an NDArray (no CPU readback).
    public func runReturningNDArray(inputs: [ModelInput]) async throws -> NDArray {
        let fn = try await ensureLoaded()

        var namedInputs: [String: NDArray] = [:]
        for (i, name) in fn.descriptor.inputNames.enumerated() where i < inputs.count {
            switch inputs[i] {
            case .cached(let array):
                namedInputs[name] = array
            case .floats(let data, let shape):
                guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else { continue }
                let resolved = nd.resolvingDynamicDimensions(shape)
                var array = NDArray(descriptor: resolved)
                Self.packFloatsIntoNDArray(&array, data: data, scalarType: resolved.scalarType)
                namedInputs[name] = array
            }
        }

        var outputs = try await fn.run(inputs: namedInputs)
        guard let outputName = fn.descriptor.outputNames.first,
            let srcArray = outputs.remove(outputName)?.ndArray
        else {
            throw CoreAIDiffusionError.notLoaded
        }
        return srcArray
    }

    private func encodeAndSync(fn: InferenceFunction, inputs: [String: NDArray]) async throws -> [Float] {
        var outputs = try await fn.run(inputs: inputs)

        guard let outputName = fn.descriptor.outputNames.first,
            let srcArray = outputs.remove(outputName)?.ndArray
        else {
            return []
        }

        return try ndArrayToFloats(srcArray)
    }

    private func encodeAndSyncAll(fn: InferenceFunction, inputs: [String: NDArray]) async throws -> [String: [Float]] {
        var outputs = try await fn.run(inputs: inputs)

        var result: [String: [Float]] = [:]
        for name in fn.descriptor.outputNames {
            guard let srcArray = outputs.remove(name)?.ndArray else { continue }
            result[name] = try ndArrayToFloats(srcArray)
        }
        return result
    }

    /// Read an output NDArray into a dense `[Float]`.
    ///
    /// Goes through `readNDArray` so a padded output buffer is indexed by stride
    /// rather than read linearly, matching how inputs are written. The element count
    /// comes from a same-typed view — `view(as:)` traps when the type disagrees with
    /// the array's scalar type, so it can't be probed generically.
    private func ndArrayToFloats(_ array: NDArray) throws -> [Float] {
        switch array.scalarType {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16, .bfloat16:
            let count = array.view(as: Float16.self).withUnsafePointer { _, shape, _ in
                (0..<shape.count).reduce(1) { $0 * shape[$1] }
            }
            return readNDArray(array, as: Float16.self, count: count).map { Float($0) }
        #endif
        case .float32:
            let count = array.view(as: Float.self).withUnsafePointer { _, shape, _ in
                (0..<shape.count).reduce(1) { $0 * shape[$1] }
            }
            return readNDArray(array, as: Float.self, count: count)
        default:
            throw CoreAIDiffusionError.unsupportedOutputScalarType(array.scalarType)
        }
    }

    public var inputDescriptors: [String: NDArrayDescriptor] {
        get async throws {
            let fn = try await ensureLoaded()
            var result: [String: NDArrayDescriptor] = [:]
            for name in fn.descriptor.inputNames {
                if case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: name) {
                    result[name] = desc
                }
            }
            return result
        }
    }

    public var outputDescriptors: [String: NDArrayDescriptor] {
        get async throws {
            let fn = try await ensureLoaded()
            var result: [String: NDArrayDescriptor] = [:]
            for name in fn.descriptor.outputNames {
                if case .ndArray(let desc) = fn.descriptor.outputDescriptor(of: name) {
                    result[name] = desc
                }
            }
            return result
        }
    }

    /// Infer the sequence length from the first input's shape (dim 1).
    /// Returns nil if the model isn't loaded or has no rank-2 input.
    public func inferSequenceLength() async throws -> Int? {
        let descs = try await inputDescriptors
        guard let desc = descs.values.first, desc.shape.count >= 2 else {
            return nil
        }
        let dim = desc.shape[1]
        return dim > 0 ? dim : nil
    }

    // MARK: - Stride-aware fill/read helpers

    /// Fill an NDArray buffer respecting non-contiguous strides.
    private static func fillStrided<T>(
        _ ptr: UnsafeMutablePointer<T>,
        data: [Float],
        shape: Span<Int>,
        strides: Span<Int>,
        convert: (Float) -> T
    ) {
        let ndim = shape.count
        if ndim <= 1 {
            for j in 0..<data.count { ptr[j] = convert(data[j]) }
            return
        }
        if isContiguousRowMajor(shape: shape, strides: strides) {
            for j in 0..<data.count { ptr[j] = convert(data[j]) }
            return
        }
        var indices = [Int](repeating: 0, count: ndim)
        for j in 0..<data.count {
            var offset = 0
            for d in 0..<ndim { offset += indices[d] * strides[d] }
            ptr[offset] = convert(data[j])
            var d = ndim - 1
            while d >= 0 {
                indices[d] += 1
                if indices[d] < shape[d] { break }
                indices[d] = 0
                d -= 1
            }
        }
    }
}

// MARK: - Errors

public enum CoreAIDiffusionError: Error, LocalizedError {
    case functionNotFound(String, URL)
    case notLoaded
    case unsupportedInputScalarType(NDArray.ScalarType)
    case unsupportedOutputScalarType(NDArray.ScalarType)
    case expectedSingleOutput(got: [String])
    case inputCountMismatch(name: String, shape: [Int], expected: Int, got: Int)

    public var errorDescription: String? {
        switch self {
        case .functionNotFound(let name, let url):
            return "Function '\(name)' not found in \(url.lastPathComponent)"
        case .notLoaded:
            return "Model not loaded. Call loadResources() first."
        case .unsupportedInputScalarType(let type):
            return "Unsupported model input scalar type: \(type) (expected float16 or float32)"
        case .unsupportedOutputScalarType(let type):
            return "Unsupported model output scalar type: \(type) (expected float16 or float32)"
        case .expectedSingleOutput(let names):
            return "Model declares \(names.count) outputs \(names); predict(...) expects exactly one. "
                + "Use predictAllOutputs(inputs:) for multi-output models."
        case .inputCountMismatch(let name, let shape, let expected, let got):
            return "Input '\(name)' expects \(expected) elements for shape \(shape), got \(got). "
                + "Check the order of the values passed to run(...) — binding is positional."
        }
    }
}
