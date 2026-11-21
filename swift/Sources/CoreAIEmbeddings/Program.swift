// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// A dense row-major tensor on the host.
struct HostTensor: Sendable, Equatable {
    enum Storage: Sendable, Equatable {
        case int32([Int32])
        case float32([Float])
    }

    var storage: Storage
    var shape: [Int]

    init(int32 values: [Int32], shape: [Int]) {
        precondition(values.count == shape.reduce(1, *), "HostTensor: \(values.count) values for shape \(shape)")
        storage = .int32(values)
        self.shape = shape
    }

    init(float32 values: [Float], shape: [Int]) {
        precondition(values.count == shape.reduce(1, *), "HostTensor: \(values.count) values for shape \(shape)")
        storage = .float32(values)
        self.shape = shape
    }

    /// The values as `Float`.
    var floats: [Float] {
        switch storage {
        case .int32(let v): return v.map(Float.init)
        case .float32(let v): return v
        }
    }

    /// The values as `Int32`; `nil` for float storage.
    var ints: [Int32]? {
        if case .int32(let v) = storage { return v }
        return nil
    }
}

/// A component's loaded asset. `CoreAIProgram` runs Core AI models; tests use fakes.
protocol Program: Sendable {
    /// Runs `function`. Inputs are converted to the function's declared scalar types; float
    /// outputs come back as `float32`.
    func run(function: String, inputs: [String: HostTensor]) async throws -> [String: HostTensor]
}

/// A `Program` over a Core AI model specialized for one compute unit. Functions load on first use.
actor CoreAIProgram: Program {
    let url: URL
    let unit: ComputeUnit
    private let model: AIModel
    private var functions: [String: InferenceFunction] = [:]

    /// Loads (and on first use specializes) the model at `url` for `unit`.
    init(url: URL, unit: ComputeUnit) async throws {
        self.url = url
        self.unit = unit
        var options: SpecializationOptions
        switch unit {
        case .gpu:
            options = SpecializationOptions(preferredComputeUnitKind: .gpu)
            options.expectFrequentReshapes = true
        case .neuralEngine: options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        }
        do {
            model = try await AIModel(contentsOf: url, options: options)
        } catch {
            throw EmbeddingError.programFailed(
                "\(url.lastPathComponent) failed on \(unit.rawValue): \(error.localizedDescription)")
        }
    }

    func run(function name: String, inputs: [String: HostTensor]) async throws -> [String: HostTensor] {
        let fn = try await function(named: name)
        let descriptor = fn.descriptor
        var arrays: [String: NDArray] = [:]
        for (key, tensor) in inputs {
            guard case .ndArray(let input) = descriptor.inputDescriptor(of: key) else {
                throw EmbeddingError.programFailed(
                    "cannot feed \(key) to \(url.lastPathComponent)/\(name): no such input (inputs: \(descriptor.inputNames))"
                )
            }
            var array = NDArray(descriptor: input.resolvingDynamicDimensions(tensor.shape))
            guard array.shape == tensor.shape else {
                throw EmbeddingError.programFailed(
                    "cannot feed \(key) to \(url.lastPathComponent)/\(name): it takes \(array.shape), got \(tensor.shape)"
                )
            }
            try Self.fill(&array, with: tensor, name: key)
            arrays[key] = array
        }
        var outputs: InferenceFunction.Outputs
        do {
            outputs = try await fn.run(inputs: arrays)
        } catch {
            throw failure(error, function: name)
        }
        var result: [String: HostTensor] = [:]
        for key in descriptor.outputNames {
            guard let array = outputs.remove(key)?.ndArray else { continue }
            result[key] = HostTensor(float32: flattenAsFloat(array), shape: array.shape)
        }
        return result
    }

    private func function(named name: String) async throws -> InferenceFunction {
        if let fn = functions[name] { return fn }
        let loaded: InferenceFunction?
        do {
            loaded = try model.loadFunction(named: name)
        } catch {
            throw failure(error, function: name)
        }
        guard let fn = loaded else {
            throw EmbeddingError.programFailed(
                "\(url.lastPathComponent) has no function '\(name)' (functions: \(model.functionNames.sorted()))"
            )
        }
        functions[name] = fn
        return fn
    }

    private func failure(_ error: Error, function: String) -> Error {
        if error is EmbeddingError { return error }
        return EmbeddingError.programFailed(
            "\(url.lastPathComponent)/\(function) failed on \(unit.rawValue): \(error.localizedDescription)")
    }

    /// Writes `tensor` into `array`, converting to the array's scalar type.
    static func fill(_ array: inout NDArray, with tensor: HostTensor, name: String) throws {
        switch (tensor.storage, array.scalarType) {
        case (.int32(let v), .int32):
            fillNDArray(&array, as: Int32.self, count: v.count) { v[$0] }
        case (_, .float32):
            // The stride-aware fill: a GPU array of a 4D input (CLIP `pixel_values`) can be padded.
            let v = tensor.floats
            fillNDArray(&array, as: Float.self, count: v.count) { v[$0] }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case (_, .float16):
            fillFloatNDArray(&array, with: tensor.floats)
        #endif
        default:
            throw EmbeddingError.programFailed(
                "cannot feed \(name): \(tensor.ints == nil ? "float32" : "int32") values to a \(array.scalarType) input"
            )
        }
    }
}

/// One `Program` per asset and compute unit, shared by the components that name it.
struct ProgramCache: Sendable {
    /// Makes and loads the program of a component's asset.
    typealias Factory = @Sendable (_ url: URL, _ component: Component) async throws -> any Program

    private let bundle: ModelBundle
    private let factory: Factory
    private var programs: [String: any Program] = [:]

    init(bundle: ModelBundle, factory: @escaping Factory) {
        self.bundle = bundle
        self.factory = factory
    }

    /// The program of `component`, loading its asset on first use.
    mutating func program(for component: Component) async throws -> any Program {
        let key = component.asset + "/" + component.computeUnits[0].rawValue
        if let program = programs[key] { return program }
        let program = try await factory(bundle.requireModelURL(for: component.asset), component)
        programs[key] = program
        return program
    }

    /// A factory of loaded `CoreAIProgram`s on the component's first compute unit.
    static let coreAI: Factory = { url, component in
        try await InstrumentsProfiler.profile(InstrumentsProfiler.beginModelLoad(name: component.asset)) {
            try await CoreAIProgram(url: url, unit: component.computeUnits[0])
        }
    }
}
