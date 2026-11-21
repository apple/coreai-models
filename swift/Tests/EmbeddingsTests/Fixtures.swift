// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation
import Testing

@testable import CoreAIEmbeddings

extension EmbeddingBlock {
    func tower(for modality: EmbeddingModality) -> Tower? { towers.first { $0.modality == modality } }
}

/// The `metadata.json` files of exported bundles, in `Resources/Bundles`: `clip-vit-b32`,
/// `embeddinggemma-2` and `embeddinggemma-2-ios`.
enum Fixtures {
    /// Fixture `name` after `edit` changes its `embedding` object.
    static func data(_ name: String, editing edit: (inout [String: Any]) -> Void = { _ in }) throws -> Data {
        let url = try #require(Bundle.module.url(forResource: "Bundles/\(name)", withExtension: "json"))
        var root = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var embedding = try #require(root["embedding"] as? [String: Any])
        edit(&embedding)
        root["embedding"] = embedding
        return try JSONSerialization.data(withJSONObject: root)
    }

    /// Decodes the `embedding` block of fixture `name` after `edit`.
    static func block(_ name: String, editing edit: (inout [String: Any]) -> Void = { _ in }) throws -> EmbeddingBlock {
        try EmbeddingBlock(metadata: data(name, editing: edit))
    }

    /// Makes `clip-vit-b32` text-only.
    static func clipTextOnly(_ embedding: inout [String: Any]) {
        embedding["towers"] = [["modality": "text", "pipeline": ["text_encoder"]]]
        embedding["image"] = nil
    }

    /// An RGB image with pixel `(x, y)` colored by `color`.
    static func image(width: Int, height: Int, color: (Int, Int) -> (UInt8, UInt8, UInt8)) throws -> CGImage {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let (r, g, b) = color(x, y)
                pixels.replaceSubrange((y * width + x) * 4..<(y * width + x) * 4 + 3, with: [r, g, b])
            }
        }
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        return try #require(
            CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }
}

extension Dictionary where Key == String, Value == Any {
    /// Applies `edit` to the object at `path` (keys separated by dots).
    mutating func edit(_ path: String, _ edit: (inout [String: Any]) -> Void) {
        let keys = path.split(separator: ".").map(String.init)
        guard let first = keys.first else { return edit(&self) }
        var child = self[first] as? [String: Any] ?? [:]
        child.edit(keys.dropFirst().joined(separator: "."), edit)
        self[first] = child
    }
}

/// One recorded call of a fake program.
struct FakeCall: Sendable {
    let component: String
    let function: String
    let inputs: [String: HostTensor]
}

/// Records the loads and calls of every fake program of a model, and holds the calls while `hold()`
/// is in effect.
actor CallLog {
    private(set) var calls: [FakeCall] = []
    /// The programs the model has loaded.
    private(set) var loads = 0
    /// The most calls running at once.
    private(set) var mostRunning = 0
    private var running = 0
    private var held: [CheckedContinuation<Void, Never>]?

    func calls(of component: String) -> [FakeCall] { calls.filter { $0.component == component } }

    func load() { loads += 1 }

    /// Makes later calls wait in `begin` until `release()`.
    func hold() { held = held ?? [] }

    func release() {
        let waiting = held ?? []
        held = nil
        for call in waiting { call.resume() }
    }

    func begin(_ call: FakeCall) async {
        calls.append(call)
        running += 1
        mostRunning = max(mostRunning, running)
        if held != nil { await withCheckedContinuation { held?.append($0) } }
    }

    func end() { running -= 1 }
}

/// A program of one asset that answers every function with outputs of the declared shapes. Each
/// output row `b` is filled by `value(component, output, b, inputs)`, or by its row index plus one.
struct FakeProgram: Program {
    let asset: String
    let block: EmbeddingBlock
    let log: CallLog
    let value: @Sendable (String, String, Int, [String: HostTensor]) -> [Float]?

    func run(function: String, inputs: [String: HostTensor]) async throws -> [String: HostTensor] {
        let (component, spec) = try #require(
            block.components.first {
                $0.value.asset == asset && $0.value.functions.contains { $0.function == function }
            })
        await log.begin(FakeCall(component: component, function: function, inputs: inputs))
        let batch = inputs.values.first?.shape.first ?? 1
        let length = inputs.values.lazy.compactMap { tensor -> Int? in tensor.shape.count > 1 ? tensor.shape[1] : nil }
            .first
        var outputs: [String: HostTensor] = [:]
        for (name, out) in spec.outputs {
            let rowShape = out.shape.dropFirst().map { axis -> Int in
                switch axis {
                case .length: return length ?? 1
                case .fixed(let n): return n
                default: return 1
                }
            }
            let rowCount = rowShape.reduce(1, *)
            var values: [Float] = []
            for b in 0..<batch {
                let row = value(component, name, b, inputs) ?? [Float](repeating: Float(b + 1), count: rowCount)
                values +=
                    row.count == rowCount
                    ? row : Array(row.prefix(rowCount)) + [Float](repeating: 0, count: max(0, rowCount - row.count))
            }
            outputs[name] = HostTensor(float32: values, shape: [batch] + rowShape)
        }
        await log.end()
        return outputs
    }
}

/// Fake models over the fixture metadata.
enum FakeModels {
    /// An `EmbeddingBundle` for fixture `name` after `edit`, rooted in a temporary directory.
    static func bundle(_ name: String, editing edit: (inout [String: Any]) -> Void = { _ in }) throws
        -> EmbeddingBundle
    {
        let raw = try Fixtures.data(name, editing: edit)
        return try EmbeddingBundle(
            bundle: ModelBundle(raw: raw, bundlePath: URL(fileURLWithPath: NSTemporaryDirectory())))
    }

    /// Token ids: BOS (when the bundle has one), one id per word (its length + 10), EOS.
    static func text(_ config: TextConfig) -> TextPreparer {
        TextPreparer(config: config) { string in
            (config.bosTokenId.map { [Int($0)] } ?? []) + string.split(separator: " ").map { $0.count + 10 }
                + [Int(config.eosTokenId)]
        }
    }

    /// A model over fixture `name` after `edit`, whose programs record into `log`.
    static func model(
        _ name: String, log: CallLog = CallLog(), text: TextPreparer? = nil,
        value: @escaping @Sendable (String, String, Int, [String: HostTensor]) -> [Float]? = { _, _, _, _ in nil },
        editing edit: (inout [String: Any]) -> Void = { _ in }
    ) throws -> CoreAIEmbeddingModel {
        let bundle = try bundle(name, editing: edit)
        let text = text ?? Self.text(bundle.block.text)
        return try CoreAIEmbeddingModel(
            bundle: bundle,
            programs: { _, component in
                await log.load()
                return FakeProgram(asset: component.asset, block: bundle.block, log: log, value: value)
            },
            text: text)
    }

    /// The one-letter words `a`-`z`, and `café` with CLIP's id, as a CLIP vocabulary. Byte-level BPE
    /// spells `é` as `Ã©`; `clipMerges` builds the word from its letters.
    static let clipVocab = Dictionary(
        uniqueKeysWithValues: (0..<26).map { ("\(UnicodeScalar(97 + $0)!)</w>", Int32(320 + $0)) }
            + [("cafÃ©</w>", 15304)])
    static let clipMerges = [["Ã", "©</w>"], ["f", "Ã©</w>"], ["a", "fÃ©</w>"], ["c", "afÃ©</w>"]]

    /// A `CLIPTokenizer` preparer for `config` over `clipVocab`.
    static func clipText(_ config: TextConfig) async throws -> TextPreparer {
        let folder = FileManager.default.temporaryDirectory.appending(path: "clip-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try JSONSerialization.data(withJSONObject: ["model": ["vocab": clipVocab, "merges": clipMerges]])
            .write(to: folder.appending(path: "tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["tokenizer_class": "CLIPTokenizer"])
            .write(to: folder.appending(path: "tokenizer_config.json"))
        return try await TextPreparer.load(from: folder, config: config)
    }
}
