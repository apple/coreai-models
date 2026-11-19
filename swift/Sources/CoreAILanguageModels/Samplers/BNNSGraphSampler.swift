// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAIShared
import Synchronization

/// A CPU sampler built on BNNSGraph, with one compiled graph per sampling configuration.
///
/// It samples from the same distribution as ``ReferenceSampler``, but it walks the kept tokens
/// in vocabulary order, so for the same random number it can return a different token. Select it
/// with `SamplingConfiguration.Backend.bnns`; seeded requests always use ``ReferenceSampler``.
///
/// Each temperature, topK, topP, minP and vocabulary size gets its own graph, compiled on first use
/// and kept for the life of the process. Repetition penalty, the logit soft cap and grammar masks
/// are applied to the logits before this sampler runs.
///
/// A row the graph can't decide the way ``ReferenceSampler`` does goes to ``ReferenceSampler``:
/// - a NaN or +infinity logit, or no finite logit;
/// - topP without topK, when the nucleus doesn't close inside the first ``candidateWindow``
///   tokens;
/// - a rounding miss at the end of the cumulative distribution.
///
/// TODO: optimize. The graphs are a direct port of the per-configuration sampler benchmark
/// graphs; the threading choice, the window size and the non-finite check are not tuned yet.
enum BNNSGraphSampler {
    /// Candidates kept for topP without topK. A larger nucleus goes to ``ReferenceSampler``.
    static let candidateWindow = 128

    /// Samples the next token with the system random number generator.
    static func sample(from logits: inout [LogitsScalarType], config: SamplingConfiguration) -> Int32 {
        var rng = SystemRandomNumberGenerator()
        return sample(from: &logits, config: config, using: &rng)
    }

    /// Samples the next token, drawing the random number from `rng`.
    ///
    /// The draw order differs from ``ReferenceSampler``'s, so the same `rng` state gives a token
    /// from the same distribution, not necessarily the same token.
    static func sample(
        from logits: inout [LogitsScalarType],
        config: SamplingConfiguration,
        using rng: inout some RandomNumberGenerator
    ) -> Int32 {
        let key = GraphKey(config, vocabSize: logits.count)
        guard !logits.isEmpty, let graph = graph(for: key) else {
            SamplerUsage.recordBNNSFallback()
            return ReferenceSampler.sample(from: &logits, config: config, using: &rng)
        }
        let u = key.isGreedy ? 0 : Float.random(in: 0..<1, using: &rng)
        if let token = graph.run(logits: &logits, u: u) {
            SamplerUsage.recordBNNS()
            return token
        }
        SamplerUsage.recordBNNSFallback()
        return ReferenceSampler.sample(from: &logits, config: config, using: &rng)
    }

    // MARK: - Graph cache

    /// What a graph is specialized for. Repetition penalty and the seed don't change the graph.
    struct GraphKey: Hashable {
        let temperature: Double
        let topK: Int?
        let topP: Double?
        let minP: Double?
        let vocabSize: Int

        init(_ config: SamplingConfiguration, vocabSize: Int) {
            let greedy = config.temperature == 0
            temperature = config.temperature
            topK = greedy ? nil : config.topK.map { min($0, vocabSize) }
            // ReferenceSampler treats topP >= 1 as off.
            topP = greedy ? nil : config.topP.flatMap { $0 < 1 ? $0 : nil }
            minP = greedy ? nil : config.minP
            self.vocabSize = vocabSize
        }

        var isGreedy: Bool { temperature == 0 }
    }

    /// The most graphs kept; the oldest is dropped beyond it.
    static let cacheLimit = 32

    private struct Cache {
        /// Compiled graphs by key; nil records a graph that failed to compile.
        var graphs: [GraphKey: CompiledGraph?] = [:]
        /// Keys in insertion order, for eviction.
        var order: [GraphKey] = []
    }

    private static let cache = Mutex(Cache())

    static func graph(for key: GraphKey) -> CompiledGraph? {
        if let cached = cache.withLock({ $0.graphs[key] }) {
            return cached
        }
        // Compile outside the lock so requests with cached graphs don't wait on it. Two callers
        // may compile the same key; the first to finish is kept.
        let compiled: CompiledGraph?
        do {
            compiled = try CompiledGraph(key: key)
        } catch {
            CLILogger.log(
                "BNNSGraphSampler: graph for \(key) failed to compile (\(error)); using ReferenceSampler",
                component: "Sampling")
            compiled = nil
        }
        return cache.withLock { cache in
            if let raced = cache.graphs[key] {
                return raced
            }
            if cache.order.count >= cacheLimit {
                cache.graphs[cache.order.removeFirst()] = nil
            }
            cache.graphs[key] = compiled
            cache.order.append(key)
            return compiled
        }
    }

    // MARK: - Compiled graph

    enum GraphError: Error {
        /// The compiled graph's arguments aren't [token, status, inputs...].
        case unexpectedArguments([String])
    }

    /// One compiled graph with its output and random-number buffers. Every graph has two
    /// outputs: the token (as a Float) and a status flag (1 = let ``ReferenceSampler`` decide).
    final class CompiledGraph: @unchecked Sendable {
        // `arguments` and the buffers are only touched under `lock`.
        private let context: BNNSGraph.Context
        private let lock = Mutex(())
        private let vocabSize: Int
        private let logitsPosition: Int
        private let token = UnsafeMutablePointer<Float>.allocate(capacity: 1)
        private let status = UnsafeMutablePointer<Float>.allocate(capacity: 1)
        private let u = UnsafeMutablePointer<Float>.allocate(capacity: 1)
        private var arguments: [UnsafeMutableBufferPointer<UInt8>] = []

        init(key: GraphKey) throws {
            // Multithreading pays off on the full-vocabulary draw (temperature only, minP only);
            // the topK graphs are faster on one thread.
            let fullVocabulary = !key.isGreedy && key.topK == nil && key.topP == nil
            let context = try BNNSGraph.makeContext(options: .init(useSingleThread: !fullVocabulary)) { builder in
                BNNSGraphSampler.build(builder, key: key)
            }
            self.context = context
            vocabSize = key.vocabSize

            // A built graph lists its outputs first, in the order returned, then its inputs.
            let inputs = key.isGreedy ? ["x"] : ["x", "u"]
            let names = context.argumentNames()
            let inputPositions = inputs.map { names.contains($0) ? context.argumentPosition(argument: $0) : -1 }
            guard names.count == 2 + inputs.count, inputPositions.allSatisfy({ $0 >= 2 }) else {
                throw GraphError.unexpectedArguments(names)
            }
            logitsPosition = inputPositions[0]
            arguments = [UnsafeMutableBufferPointer<UInt8>](repeating: Self.bytes(token), count: names.count)
            arguments[1] = Self.bytes(status)
            if inputs.count > 1 {
                arguments[inputPositions[1]] = Self.bytes(u)
            }
        }

        deinit {
            token.deallocate()
            status.deallocate()
            u.deallocate()
        }

        /// The sampled token, or nil when ``ReferenceSampler`` has to decide this row.
        func run(logits: inout [LogitsScalarType], u draw: Float) -> Int32? {
            lock.withLock { _ in
                u.pointee = draw
                let succeeded = logits.withUnsafeMutableBytes { logitBytes -> Bool in
                    arguments[logitsPosition] = UnsafeMutableBufferPointer(
                        start: logitBytes.baseAddress!.assumingMemoryBound(to: UInt8.self),
                        count: logitBytes.count)
                    do {
                        try context.executeFunction(nil, arguments: arguments)
                        return true
                    } catch {
                        return false
                    }
                }
                guard succeeded, status.pointee == 0 else { return nil }
                let index = Int(token.pointee)
                return (0..<vocabSize).contains(index) ? Int32(index) : nil
            }
        }

        private static func bytes(_ pointer: UnsafeMutablePointer<Float>) -> UnsafeMutableBufferPointer<UInt8> {
            UnsafeMutableBufferPointer(
                start: UnsafeMutableRawPointer(pointer).assumingMemoryBound(to: UInt8.self),
                count: MemoryLayout<Float>.size)
        }
    }

    // MARK: - Graph construction

    typealias Tensor<T: BNNSScalar> = BNNSGraph.Builder.Tensor<T>

    /// Builds the graph for `key`. Outputs: [token, status], both [1, 1] Float.
    ///
    /// ReferenceSampler's order: z = logits / temperature, then minP (z >= max + log(minP)), topP
    /// (the shortest prefix of the z-descending order whose share reaches topP, among the
    /// minP-kept tokens) and topK, then a draw in proportion to exp(z - max) over the kept tokens.
    fileprivate static func build(_ b: BNNSGraph.Builder, key: GraphKey) -> [any BNNSGraph.TensorDescriptor] {
        let k = Constants(b)
        let vocabSize = key.vocabSize
        let x = b.argument(name: "x", dataType: LogitsScalarType.self, shape: [1, vocabSize]).cast(to: Float.self)

        // ReferenceSampler's handling of NaN, +infinity and all-masked rows is kept by sending
        // those rows to it.
        let nonFinite = (x .!= x) .| (x .== b.constant(value: Float.infinity))
        let noFiniteLogit = x.maximum(axes: [1], keepDimensions: true) .== b.constant(value: -Float.infinity)
        let unusualRow = (k.count(nonFinite) .> k.zero) .| noFiniteLogit

        if key.isGreedy {
            return [x.argMax(axis: 1, keepDimension: true).cast(to: Float.self), k.flag(unusualRow)]
        }

        let u = b.argument(name: "u", dataType: Float.self, shape: [1, 1])
        let temperature = Float(key.temperature)
        let logMinP = key.minP.map { logf(Float($0)) }

        if let topK = key.topK {
            // Select on the raw logits (same order as z), then scale only the K candidates.
            let (values, indices) = x.topK(topK, axis: 1, findLargest: true)
            let z = values / temperature
            let maxZ = z[0..<1, 0..<1]
            let weights = (z - maxZ).exp()
            let rank = iota(b, topK)
            var keep: Tensor<Bool>? = logMinP.map { z .>= maxZ + $0 }
            if let topP = key.topP {
                let total = keep.map { $0.select(weights, k.zero) } ?? weights
                keep =
                    nucleus(
                        k, weights: weights, kept: keep, total: total.sum(axes: [1], keepDimensions: true),
                        topP: b.constant(value: Float(topP)), rank: rank
                    ).keep
            }
            let token = drawCandidate(k, weights: weights, indices: indices, keep: keep, u: u, rank: rank, count: topK)
            return [token.cast(to: Float.self), k.flag(unusualRow)]
        }

        let z = x / temperature

        if let topP = key.topP {
            // The nucleus is a prefix of the z-descending order, so the first `window` tokens
            // decide it whenever it closes inside them. Its share is over the whole minP-kept
            // vocabulary.
            let window = min(candidateWindow, vocabSize)
            let (candidates, indices) = z.topK(window, axis: 1, findLargest: true)
            let maxZ = candidates[0..<1, 0..<1]
            var allWeights = (z - maxZ).exp()
            if let logMinP {
                allWeights = (z .>= maxZ + logMinP).select(allWeights, k.zero)
            }
            let weights = (candidates - maxZ).exp()
            let rank = iota(b, window)
            let minPKept: Tensor<Bool>? = logMinP.map { candidates .>= maxZ + $0 }
            let (keep, cut) = nucleus(
                k, weights: weights, kept: minPKept, total: allWeights.sum(axes: [1], keepDimensions: true),
                topP: b.constant(value: Float(topP)), rank: rank)
            let token = drawCandidate(
                k, weights: weights, indices: indices, keep: keep, u: u, rank: rank, count: window)
            let nucleusClosed = cut .< b.constant(value: Float(window))
            let closed =
                minPKept.map { nucleusClosed .| (k.count($0) .< b.constant(value: Float(window))) } ?? nucleusClosed
            let decided = window >= vocabSize ? .!unusualRow : (closed .& .!unusualRow)
            return [token.cast(to: Float.self), k.flag(.!decided)]
        }

        // Temperature only, or minP only: a draw over the whole vocabulary.
        let maxZ = z.maximum(axes: [1], keepDimensions: true)
        var weights = (z - maxZ).exp()
        if let logMinP {
            weights = (z .>= maxZ + logMinP).select(weights, k.zero)
        }
        let draw = drawFullVocabulary(b, k, weights: weights, u: u, vocabSize: vocabSize)
        let roundingMiss = draw .>= b.constant(value: Float(vocabSize))
        return [draw, k.flag(unusualRow .| roundingMiss)]
    }

    private struct Constants {
        let zero: Tensor<Float>
        let one: Tensor<Float>
        let minusOne: Tensor<Float>

        init(_ b: BNNSGraph.Builder) {
            zero = b.constant(value: Float(0))
            one = b.constant(value: Float(1))
            minusOne = b.constant(value: Float(-1))
        }

        /// The number of true entries along axis 1, as [1, 1].
        func count(_ mask: Tensor<Bool>) -> Tensor<Float> {
            mask.select(one, zero).sum(axes: [1], keepDimensions: true)
        }

        /// 1 where `mask` is true, else 0.
        func flag(_ mask: Tensor<Bool>) -> Tensor<Float> {
            mask.select(one, zero)
        }
    }

    /// [0, 1, …, n - 1] as [1, n].
    private static func iota(_ b: BNNSGraph.Builder, _ n: Int) -> Tensor<Float> {
        b.constant(values: (0..<n).map(Float.init), shape: [1, n])
    }

    /// The nucleus over candidates sorted by z descending: the shortest prefix whose share of
    /// `total` reaches `topP`, among the `kept` candidates. Returns the kept mask and the cut,
    /// the number of positions whose running share is below `topP`.
    private static func nucleus(
        _ k: Constants, weights: Tensor<Float>, kept: Tensor<Bool>?, total: Tensor<Float>,
        topP: Tensor<Float>, rank: Tensor<Float>
    ) -> (keep: Tensor<Bool>, cut: Tensor<Float>) {
        let share = (kept.map { $0.select(weights, k.zero) } ?? weights) * (Float(1) / total)
        let cut = k.count(share.cumulativeSum(axis: 1) .< topP)
        let inNucleus = rank .<= cut
        return (kept.map { $0 .& inNucleus } ?? inNucleus, cut)
    }

    /// Draws among `count` candidates in vocabulary order: the first token whose cumulative
    /// weight exceeds u times the kept total. A rounding miss takes the last kept token.
    private static func drawCandidate(
        _ k: Constants, weights: Tensor<Float>, indices: Tensor<Int32>, keep: Tensor<Bool>?,
        u: Tensor<Float>, rank: Tensor<Float>, count: Int
    ) -> Tensor<Int32> {
        let order = indices.argSort(axis: 1, sortOrder: .ascending)
        let sortedIndices = indices.gatherAlongAxis(indices: order, axis: 1)
        let p = (keep.map { $0.select(weights, k.zero) } ?? weights).gatherAlongAxis(indices: order, axis: 1)
        let cdf = p.cumulativeSum(axis: 1)
        let total = cdf[0..<1, (count - 1)..<count]
        let position = k.count(cdf .<= total * u)
        let lastKept = (p .> k.zero).select(rank, k.minusOne).maximum(axes: [1], keepDimensions: true)
        let chosen = position.min(y: lastKept).max(y: k.zero).cast(to: Int32.self)
        return sortedIndices.gatherAlongAxis(indices: chosen, axis: 1)
    }

    /// Draws over the whole vocabulary in index order, in two levels (block sums, then inside the
    /// chosen block). Returns the token as a Float, or `vocabSize` on a rounding miss.
    private static func drawFullVocabulary(
        _ b: BNNSGraph.Builder, _ k: Constants, weights: Tensor<Float>, u: Tensor<Float>, vocabSize: Int
    ) -> Tensor<Float> {
        let (blocks, blockSize) = blockSplit(vocabSize)
        let rows = weights.reshape(to: [blocks, blockSize])
        let blockCDF = rows.sum(axes: [1], keepDimensions: false).reshape(to: [1, blocks]).cumulativeSum(axis: 1)
        let target = blockCDF[0..<1, (blocks - 1)..<blocks] * u
        let block = k.count(blockCDF .<= target)
        let blockIndex = block.min(y: b.constant(value: Float(blocks - 1))).cast(to: Int32.self)
        // The weight before each block; one block (prime vocabSize) has nothing before it.
        let prefix =
            blocks == 1
            ? b.constant(values: [Float(0)], shape: [1, 1])
            : b.concatenate(
                [b.constant(values: [Float(0)], shape: [1, 1]), blockCDF[0..<1, 0..<(blocks - 1)]], axis: 1
            ).gatherAlongAxis(indices: blockIndex, axis: 1)
        let row = rows.gather(indices: blockIndex.reshape(to: [1]), axis: 0, batchDimensionCount: 0)
            .reshape(to: [1, blockSize])
        let inBlock = k.count(row.cumulativeSum(axis: 1) + prefix .<= target)
        let found = (block .< b.constant(value: Float(blocks))) .& (inBlock .< b.constant(value: Float(blockSize)))
        return found.select(block * Float(blockSize) + inBlock, b.constant(value: Float(vocabSize)))
    }

    /// vocabSize = blocks × blockSize, with the divisor pair closest to √vocabSize (one block when vocabSize is prime).
    static func blockSplit(_ vocabSize: Int) -> (blocks: Int, blockSize: Int) {
        var best = (blocks: 1, blockSize: vocabSize)
        for blocks in 1...max(1, Int(Double(vocabSize).squareRoot())) where vocabSize % blocks == 0 {
            if blocks + vocabSize / blocks < best.blocks + best.blockSize {
                best = (blocks, vocabSize / blocks)
            }
        }
        return best
    }
}
