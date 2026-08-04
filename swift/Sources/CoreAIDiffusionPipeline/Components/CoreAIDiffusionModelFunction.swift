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

    public func run(floatInputs: [([Float], [Int])]) async throws -> [Float] {
        let fn = try await ensureLoaded()

        var namedInputs: [String: NDArray] = [:]
        for (i, name) in fn.descriptor.inputNames.enumerated() where i < floatInputs.count {
            let (data, shape) = floatInputs[i]
            guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else { continue }
            let resolved = nd.resolvingDynamicDimensions(shape)
            var array = NDArray(descriptor: resolved)
            switch resolved.scalarType {
            #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
            case .float16:
                let view = array.mutableView(as: Float16.self)
                view.withUnsafeMutablePointer { ptr, shape, strides in
                    Self.fillStrided(ptr, data: data, shape: shape, strides: strides) { Float16($0) }
                }
            case .bfloat16:
                array.mutableRawView().withUnsafeMutableBytes { ptr, shape, strides in
                    let dst = ptr.assumingMemoryBound(to: UInt16.self)
                    Self.fillStrided(dst, data: data, shape: shape, strides: strides) { val in
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
                    Self.fillStrided(ptr, data: data, shape: shape, strides: strides) { $0 }
                }
            default:
                throw CoreAIDiffusionError.unsupportedInputScalarType(resolved.scalarType)
            }
            namedInputs[name] = array
        }

        return try await encodeAndSync(fn: fn, inputs: namedInputs)
    }

    public func run(intInputs: [([Int32], [Int])]) async throws -> [Float] {
        let fn = try await ensureLoaded()

        var namedInputs: [String: NDArray] = [:]
        for (i, name) in fn.descriptor.inputNames.enumerated() where i < intInputs.count {
            let (data, shape) = intInputs[i]
            guard case .ndArray(let nd) = fn.descriptor.inputDescriptor(of: name) else { continue }
            let resolved = nd.resolvingDynamicDimensions(shape)
            var array = NDArray(descriptor: resolved)
            let view = array.mutableView(as: Int32.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for j in 0..<data.count { ptr[j] = data[j] }
            }
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

    private func ndArrayToFloats(_ array: NDArray) throws -> [Float] {
        var result = [Float]()
        switch array.scalarType {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            array.view(as: Float16.self).withUnsafePointer { ptr, shape, strides in
                let count = (0..<shape.count).reduce(1) { $0 * shape[$1] }
                result.reserveCapacity(count)
                Self.readStrided(ptr, into: &result, shape: shape, strides: strides) { Float($0) }
            }
        case .bfloat16:
            array.rawView().withUnsafeBytes { ptr, shape, strides in
                let count = (0..<shape.count).reduce(1) { $0 * shape[$1] }
                result.reserveCapacity(count)
                let src = ptr.assumingMemoryBound(to: UInt16.self)
                Self.readStrided(src, into: &result, shape: shape, strides: strides) { bits in
                    Float(bitPattern: UInt32(bits) << 16)
                }
            }
        #endif
        case .float32:
            array.view(as: Float.self).withUnsafePointer { ptr, shape, strides in
                let count = (0..<shape.count).reduce(1) { $0 * shape[$1] }
                result.reserveCapacity(count)
                Self.readStrided(ptr, into: &result, shape: shape, strides: strides) { $0 }
            }
        default:
            throw CoreAIDiffusionError.unsupportedOutputScalarType(array.scalarType)
        }
        return result
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

    /// Read from an NDArray buffer respecting non-contiguous strides.
    private static func readStrided<T>(
        _ ptr: UnsafePointer<T>,
        into result: inout [Float],
        shape: Span<Int>,
        strides: Span<Int>,
        convert: (T) -> Float
    ) {
        let ndim = shape.count
        let count = (0..<ndim).reduce(1) { $0 * shape[$1] }
        if ndim <= 1 {
            for i in 0..<count { result.append(convert(ptr[i])) }
            return
        }
        var indices = [Int](repeating: 0, count: ndim)
        for _ in 0..<count {
            var offset = 0
            for d in 0..<ndim { offset += indices[d] * strides[d] }
            result.append(convert(ptr[offset]))
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
        }
    }
}
