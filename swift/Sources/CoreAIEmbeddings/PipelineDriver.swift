// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Runs one tower: its `pipeline` in order over a table of named tensors per input (R4).
///
/// The tower's inputs seed the table. Each stage reads its declared inputs from the table, or
/// builds them by a fixed rule (`value(_:of:row:length:)`), and writes its declared outputs back;
/// the last stage's `embedding` is the vector. An input that neither provides fails the first
/// call that needs it.
///
/// | Bundle | Seed | Stages |
/// |---|---|---|
/// | CLIP text, EG2 macOS text | `input_ids`, `attention_mask` | `text_encoder` |
/// | CLIP image | `pixel_values [3,S,S]` | `image_encoder` |
/// | EG2 iOS text | `input_ids`, `attention_mask` | `token_table`, `encoder` (bias, pool weights built) |
/// | EG2 image | patches, positions, image token ids | `image_encoder`, `token_table`, `encoder` (soft tokens merged) |
struct PipelineDriver: Sendable {
    let modality: EmbeddingModality
    let stages: [(name: String, component: Component)]
    let block: EmbeddingBlock

    /// Looks up the components of `tower`.
    ///
    /// - Throws: `EmbeddingError.invalidMetadata` for an unknown component, a pipeline whose last
    ///   stage has no `embedding` output, or an image tower without an `image` block.
    init(tower: Tower, block: EmbeddingBlock) throws {
        stages = try tower.pipeline.map { name in
            guard let component = block.components[name] else {
                throw EmbeddingError.invalidMetadata("unknown component '\(name)'")
            }
            return (name, component)
        }
        guard stages.last?.component.outputs["embedding"] != nil else {
            throw EmbeddingError.invalidMetadata("the \(tower.modality.rawValue) pipeline has no embedding output")
        }
        if tower.modality == .image && block.image == nil {
            throw EmbeddingError.invalidMetadata("the image tower needs an image block")
        }
        self.modality = tower.modality
        self.block = block
    }

    // MARK: - Running

    /// Named tensors of one input, without the batch axis.
    struct Row: Sendable {
        var tensors: [String: HostTensor] = [:]
        /// Real token count, for inputs with tokens.
        var length: Int?
    }

    /// The image token ids `<bos> <boi> <image>*n <eoi> <eos>`, BOS and EOS as the text block adds them.
    func imageTokenIds(softTokens n: Int) -> [Int32] {
        guard let p = block.image?.patches else { return [] }
        let text = block.text
        return (text.addBosToken ? text.bosTokenId.map { [$0] } ?? [] : []) + [p.boiTokenId]
            + [Int32](repeating: p.imageTokenId, count: n) + [p.eoiTokenId]
            + (text.addEosToken ? [text.eosTokenId] : [])
    }

    /// Seed rows from token ids.
    func rows(tokens: [[Int32]]) -> [Row] {
        tokens.map { ids in
            Row(
                tensors: [
                    "input_ids": HostTensor(int32: ids, shape: [ids.count]),
                    "attention_mask": HostTensor(int32: [Int32](repeating: 1, count: ids.count), shape: [ids.count]),
                ], length: ids.count)
        }
    }

    /// Seed rows from images.
    func rows(images: [RGB8Image]) throws -> [Row] {
        let config = block.image!
        return try images.map { image in
            switch config.strategy {
            case .centerCrop:
                let size = config.size!
                return Row(tensors: [
                    "pixel_values": HostTensor(
                        float32: ImagePreparer.centerCrop(image, config), shape: [3, size, size])
                ])
            case .aspectRatioPreserving:
                let p = try ImagePreparer.patchify(image, config)
                let g = config.patches!
                var row = rows(tokens: [imageTokenIds(softTokens: p.softTokens)])[0]
                row.tensors["pixel_values"] = HostTensor(
                    float32: p.pixelValues, shape: [g.maxPatches, g.patchDimension])
                row.tensors["position_ids"] = HostTensor(int32: p.positionIds, shape: [g.maxPatches, 2])
                return row
            }
        }
    }

    /// Runs the stages over `rows` with programs from `programs` and returns each row's `embedding`.
    /// Checks for cancellation before each program call.
    func run(_ rows: [Row], programs: inout ProgramCache) async throws -> [[Float]] {
        var rows = rows
        for stage in stages {
            let program = try await programs.program(for: stage.component)
            for call in try stage.component.plan(lengths: rows.map { $0.length ?? 0 }, name: stage.name) {
                try Task.checkCancellation()
                var inputs: [String: HostTensor] = [:]
                for (name, spec) in stage.component.inputs {
                    let values = try call.rows.map { r in
                        try value(name, of: stage.component, row: rows[r], length: call.length)
                    }
                    inputs[name] = try stack(
                        values, spec: spec, call: call, pad: name == "input_ids" ? block.text.padTokenId : 0)
                }
                let span = InstrumentsProfiler.beginEmbedInference(
                    component: stage.name, function: call.function, batch: call.rows.count)
                let outputs = try await InstrumentsProfiler.profile(span) {
                    try await program.run(function: call.function, inputs: inputs)
                }
                for (name, spec) in stage.component.outputs {
                    guard let output = outputs[name] else {
                        throw EmbeddingError.programFailed("\(stage.name) returned no '\(name)'")
                    }
                    for (slot, r) in call.rows.enumerated() {
                        rows[r].tensors[name] = try unstack(output, slot: slot, spec: spec, length: rows[r].length)
                    }
                }
            }
        }
        return try rows.map { row in
            guard let vector = row.tensors["embedding"]?.floats, vector.count == block.dimension else {
                throw EmbeddingError.programFailed(
                    "\(stages.last?.name ?? "") returned no \(block.dimension)-wide embedding")
            }
            return vector
        }
    }

    /// Input `name` of `component` for `row` at the call length `length`: built by a fixed rule
    /// (`inputs_embeds` with soft tokens merged, `bias`, `pool_weights`), else read from the table.
    func value(_ name: String, of component: Component, row: Row, length: Int?) throws -> HostTensor {
        let mask = row.tensors["attention_mask"]?.ints ?? []
        switch name {
        case "inputs_embeds" where row.tensors["soft_tokens"] != nil:
            // `inputs_embeds` with `soft_tokens` written over the `image_token_id` positions.
            guard let embeds = row.tensors["inputs_embeds"], let soft = row.tensors["soft_tokens"],
                let ids = row.tensors["input_ids"]?.ints, let hidden = embeds.shape.last, soft.shape.last == hidden,
                let imageTokenId = block.image?.patches?.imageTokenId
            else {
                throw EmbeddingError.invalidMetadata(
                    "cannot run the \(modality.rawValue) pipeline: soft-token merge needs inputs_embeds, soft_tokens, input_ids"
                )
            }
            var out = embeds.floats
            let source = soft.floats
            var next = 0
            for (position, id) in ids.enumerated() where id == imageTokenId {
                guard (next + 1) * hidden <= source.count else {
                    throw EmbeddingError.programFailed(
                        "image_encoder returned fewer soft tokens than image tokens")
                }
                out.replaceSubrange(
                    position * hidden..<(position + 1) * hidden, with: source[next * hidden..<(next + 1) * hidden])
                next += 1
            }
            return HostTensor(float32: out, shape: embeds.shape)
        case "bias" where component.bias != nil:
            // `bias[k, 0, q]` from `attention_mask`.
            let bias = component.bias!
            let l = length ?? mask.count
            var values = [Float](repeating: bias.unmasked, count: l * l)
            for key in 0..<l where key >= mask.count || mask[key] == 0 {
                values.replaceSubrange(key * l..<(key + 1) * l, with: repeatElement(bias.masked, count: l))
            }
            return HostTensor(float32: values, shape: [l, 1, l])
        case "pool_weights":
            // `attention_mask / attention_mask.sum(-1)`.
            let l = length ?? mask.count
            let total = Float(max(mask.reduce(0) { $0 + Int($1) }, 1))
            let weights = (0..<l).map { $0 < mask.count ? Float(mask[$0]) / total : 0 }
            return HostTensor(float32: weights, shape: [l])
        default:
            guard let value = row.tensors[name] else {
                throw EmbeddingError.invalidMetadata(
                    "cannot run the \(modality.rawValue) pipeline: nothing provides input '\(name)'")
            }
            return value
        }
    }

    /// The batch tensor of a call: each row padded along its leading `L` axis to the call length,
    /// then padding rows up to the call batch.
    func stack(_ values: [HostTensor], spec: TensorSpec, call: Call, pad: Int32) throws -> HostTensor {
        let rowShape = try spec.shape.dropFirst().map { axis -> Int in
            if case .fixed(let n) = axis { return n }
            guard let l = call.length else {
                throw EmbeddingError.invalidMetadata("\(spec) has L but its functions have no length")
            }
            return l
        }
        let rowCount = rowShape.reduce(1, *)
        let integer = values.first?.ints != nil
        var ints: [Int32] = []
        var floats: [Float] = []
        for value in values {
            let padded = try padRow(value, to: rowShape, pad: pad)
            if integer { ints += padded.ints ?? [] } else { floats += padded.floats }
        }
        let missing = (call.batch - values.count) * rowCount
        let shape = [call.batch] + rowShape
        if integer {
            return HostTensor(int32: ints + [Int32](repeating: pad, count: missing), shape: shape)
        }
        return HostTensor(float32: floats + [Float](repeating: Float(pad), count: missing), shape: shape)
    }

    private func padRow(_ value: HostTensor, to shape: [Int], pad: Int32) throws -> HostTensor {
        if value.shape == shape { return value }
        guard !shape.isEmpty, value.shape.count == shape.count, value.shape[0] <= shape[0],
            value.shape.dropFirst() == shape.dropFirst()
        else { throw EmbeddingError.invalidMetadata("a \(value.shape) row does not fit \(shape)") }
        let extra = (shape[0] - value.shape[0]) * shape.dropFirst().reduce(1, *)
        switch value.storage {
        case .int32(let v): return HostTensor(int32: v + [Int32](repeating: pad, count: extra), shape: shape)
        case .float32(let v): return HostTensor(float32: v + [Float](repeating: Float(pad), count: extra), shape: shape)
        }
    }

    /// Row `slot` of a call output, cut back to the row's `length` along a leading `L` axis.
    func unstack(_ output: HostTensor, slot: Int, spec: TensorSpec, length: Int?) throws -> HostTensor {
        var rowShape = Array(output.shape.dropFirst())
        let rowCount = rowShape.reduce(1, *)
        guard output.shape.count == spec.shape.count, slot < output.shape[0] else {
            throw EmbeddingError.invalidMetadata("output \(output.shape) does not match \(spec)")
        }
        var values = Array(output.floats[slot * rowCount..<(slot + 1) * rowCount])
        if spec.shape.dropFirst().first == .length, let length, length < rowShape[0] {
            values.removeLast((rowShape[0] - length) * rowShape.dropFirst().reduce(1, *))
            rowShape[0] = length
        }
        return HostTensor(float32: values, shape: rowShape)
    }
}
