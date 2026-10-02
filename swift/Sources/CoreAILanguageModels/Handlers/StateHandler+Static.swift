// Persistent state handlers for static-shape (bucketed) engines.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - Shared storage

/// Storage shared by ``FixedStaticState`` and ``BucketedStaticState``: binding
/// into a running function's views, and zeroing.
///
/// A class that solely owns its `NDArray`s, so ``bind(into:for:)`` takes
/// `mutableRawView()` at refcount 1 without copying the cache.
class StaticStateStorage {
    let stateNames: [String]

    /// Backing storage per state. `var` so `bind` can take a mutable raw view.
    var arrays: [String: NDArray]

    init(stateNames: [String], arrays: [String: NDArray]) {
        self.stateNames = stateNames
        self.arrays = arrays
    }

    /// Insert every state this handler owns into `views`, sliced to the shape the
    /// running function declares. States the function does not declare are skipped.
    ///
    /// The slice is what lets one buffer serve several buckets: a fixed state
    /// allocated at the maximum context binds as the leading `ctx` positions of
    /// itself, with the strides the program was compiled against.
    @_lifetime(views: borrow self)
    func bind(
        into views: inout InferenceFunction.MutableViews,
        for descriptor: InferenceFunctionDescriptor
    ) {
        for name in stateNames {
            guard case .ndArray(let stateDescriptor) = descriptor.stateDescriptor(of: name) else {
                continue
            }
            let view = _overrideLifetime(
                arrays[name]!.mutableRawView().slice(at: stateDescriptor.shape.map { 0..<$0 }),
                borrowing: Void())
            views.insert(view, for: name)
        }
    }

    /// Zero all backing storage.
    ///
    /// Chunked-flash attention reads *every* key position each step, including
    /// positions past the write cursor, with an additive mask. Garbage there feeds
    /// `q @ k` and can overflow fp16 to `Inf`, which adding `-40000` cannot
    /// suppress, yielding NaN logits.
    func reset() {
        for name in stateNames { zeroFillNDArray(&arrays[name]!) }
    }
}

// MARK: - Fixed

/// Static state whose backing buffer does not vary with the context bucket.
///
/// Allocated once from a reference (largest-context) descriptor and zero-filled.
/// Covers both the ordinary `key_cache` / `value_cache` of models whose buckets
/// all share one cache buffer, and fixed-size auxiliary caches such as a
/// sliding-window ring.
///
/// A fixed state's declared *shape* may still shrink with the bucket: a model
/// compiled against one max-context cache gives every bucket the max-context
/// strides and declares only the first `ctx` sequence positions. ``bind(into:for:)``
/// slices to whatever the running function declares, so that case needs no
/// re-layout — the storage underneath is already the one the program indexes.
final class FixedStaticState: StaticStateStorage {
    init(states: [(name: String, descriptor: NDArrayDescriptor)]) {
        var arrays: [String: NDArray] = [:]
        for (name, descriptor) in states {
            var array = NDArray(descriptor: descriptor)
            zeroFillNDArray(&array)
            arrays[name] = array
            CLILogger.log(
                "Static state '\(name)' allocated: \(descriptor.minimumByteCount) bytes "
                    + "(fixed, shape \(descriptor.shape))")
        }
        super.init(stateNames: states.map(\.name), arrays: arrays)
    }
}

// MARK: - Bucketed

/// Static state whose backing buffer scales with the context bucket.
///
/// Allocates at the session's current bucket rather than the model maximum, and
/// re-lays-out the written prefix when the running bucket changes. Each bucket's
/// program is compiled with its own sequence stride (`ctx · interleave`), so a
/// buffer laid out for one bucket is not a valid slice of another.
final class BucketedStaticState: StaticStateStorage {
    /// Context bucket → per-state descriptor for that bucket.
    private let descriptorsByContext: [Int: [String: NDArrayDescriptor]]

    /// The bucket the backing storage is currently laid out for.
    private(set) var currentContextBucket: Int

    enum LayoutError: Error, CustomStringConvertible {
        case noBuckets
        case missingDescriptor(state: String, context: Int)
        case sequenceDimensionNotLast(state: String, context: Int, shape: [Int])
        case interleaveOutsideSequence(state: String, dimension: Int, sequenceDimension: Int)
        case paddedBuffer(state: String, context: Int, expected: Int, actual: Int)
        case unsupportedScalarType(state: String, type: String)
        case inconsistentAcrossBuckets(state: String, context: Int)

        var description: String {
            switch self {
            case .noBuckets:
                return "BucketedStaticState needs at least one context bucket"
            case .missingDescriptor(let state, let context):
                return "State '\(state)' is not declared by the ctx \(context) program"
            case .sequenceDimensionNotLast(let state, let context, let shape):
                return
                    "State '\(state)' has shape \(shape) at ctx \(context) — expected the context "
                    + "length as the last dimension, so the written prefix cannot be re-laid-out"
            case .interleaveOutsideSequence(let state, let dimension, let sequenceDimension):
                return
                    "State '\(state)' interleaves dim \(dimension), which is not inside the "
                    + "sequence dim \(sequenceDimension) — prefix re-layout would reorder elements"
            case .paddedBuffer(let state, let context, let expected, let actual):
                return
                    "State '\(state)' at ctx \(context) is padded (\(actual) bytes for \(expected) "
                    + "bytes of elements) — per-group run copy would land at wrong offsets"
            case .unsupportedScalarType(let state, let type):
                return "State '\(state)' has unsupported scalar type \(type) for prefix re-layout"
            case .inconsistentAcrossBuckets(let state, let context):
                return
                    "State '\(state)' at ctx \(context) differs in scalar type or interleave from the "
                    + "smallest bucket — prefix re-layout needs both to match"
            }
        }
    }

    /// - Parameters:
    ///   - stateNames: The states this handler owns.
    ///   - descriptorsByContext: Every context bucket's descriptor for each state.
    /// - Throws: ``LayoutError`` when the physical layout cannot support a
    ///   written-prefix re-layout. Callers must not fall back to allocating at
    ///   the maximum bucket — for a state whose stride scales with ctx, that
    ///   binds a buffer the program will index incorrectly.
    init(stateNames: [String], descriptorsByContext: [Int: [String: NDArrayDescriptor]]) throws {
        guard let smallest = descriptorsByContext.keys.min() else { throw LayoutError.noBuckets }

        for (context, byName) in descriptorsByContext {
            for name in stateNames {
                guard let descriptor = byName[name] else {
                    throw LayoutError.missingDescriptor(state: name, context: context)
                }
                try Self.validateLayout(descriptor, state: name, context: context)
                let reference = descriptorsByContext[smallest]![name]!
                guard descriptor.scalarType == reference.scalarType,
                    descriptor.interleaveLayout?.dimension == reference.interleaveLayout?.dimension,
                    descriptor.interleaveLayout?.factor == reference.interleaveLayout?.factor
                else {
                    throw LayoutError.inconsistentAcrossBuckets(state: name, context: context)
                }
            }
        }

        self.descriptorsByContext = descriptorsByContext
        self.currentContextBucket = smallest

        var arrays: [String: NDArray] = [:]
        for name in stateNames {
            let descriptor = descriptorsByContext[smallest]![name]!
            var array = NDArray(descriptor: descriptor)
            zeroFillNDArray(&array)
            arrays[name] = array
            CLILogger.log(
                "Static state '\(name)' allocated: \(descriptor.minimumByteCount) bytes "
                    + "(bucketed, starting at ctx \(smallest))")
        }
        super.init(stateNames: stateNames, arrays: arrays)
    }

    // MARK: Layout validation

    private static func byteWidth(_ type: NDArray.ScalarType) -> Int? {
        switch type {
        case .float16, .bfloat16: return 2
        case .float32: return 4
        default: return nil
        }
    }

    private static func validateLayout(
        _ descriptor: NDArrayDescriptor, state: String, context: Int
    ) throws {
        let shape = descriptor.shape
        guard shape.last == context else {
            throw LayoutError.sequenceDimensionNotLast(state: state, context: context, shape: shape)
        }
        let sequenceDimension = shape.count - 1
        if let interleave = descriptor.interleaveLayout, interleave.dimension >= sequenceDimension {
            throw LayoutError.interleaveOutsideSequence(
                state: state, dimension: interleave.dimension,
                sequenceDimension: sequenceDimension)
        }
        guard let width = byteWidth(descriptor.scalarType) else {
            throw LayoutError.unsupportedScalarType(
                state: state, type: String(describing: descriptor.scalarType))
        }
        let expected = shape.reduce(1, *) * width
        guard descriptor.minimumByteCount == expected else {
            throw LayoutError.paddedBuffer(
                state: state, context: context, expected: expected,
                actual: descriptor.minimumByteCount)
        }
    }

    // MARK: Re-layout

    /// Lay out backing storage for `contextBucket`, preserving the first
    /// `writtenTokenCount` sequence positions.
    func prepare(contextBucket: Int, writtenTokenCount: Int) throws {
        guard contextBucket != currentContextBucket else { return }
        guard let byName = descriptorsByContext[contextBucket] else {
            throw LayoutError.missingDescriptor(state: stateNames.first ?? "", context: contextBucket)
        }

        // Clamp to both layouts so a shrink never reads or writes past either end.
        let copyLength = min(writtenTokenCount, currentContextBucket, contextBucket)

        var totalBytes = 0
        for name in stateNames {
            let descriptor = byName[name]!
            var replacement = NDArray(descriptor: descriptor)
            zeroFillNDArray(&replacement)
            if copyLength > 0 {
                Self.copyPrefix(from: arrays[name]!, to: &replacement, copyLength: copyLength)
            }
            arrays[name] = replacement
            totalBytes += descriptor.minimumByteCount
        }

        CLILogger.log(
            "Static state re-laid-out: ctx \(currentContextBucket) → \(contextBucket) "
                + "(copied \(copyLength) positions, \(totalBytes) bytes across \(stateNames.count) states)")
        currentContextBucket = contextBucket
    }

    // MARK: Prefix re-layout

    /// Copies the first `copyLength` sequence positions from `source` to
    /// `destination`, re-laying-out for the destination's context length.
    ///
    /// The state is `[…, ctx]` with an optional channel interleave `(dim, factor)`
    /// inside the sequence dim: physically `[…, ctx, factor]` row-major, with the
    /// interleaved elements innermost and the sequence next. So for each
    /// `groupCount = product(shape) / ctx / factor` group, positions
    /// `[0, copyLength)` across the interleaved channels form ONE contiguous run
    /// of `copyLength · factor` elements at group base `g · (ctx · factor)`.
    /// Source and destination share interleave and group order and differ only in
    /// `ctx` — the sequence stride scale — so a per-group run copy is correct
    /// without knowing the interleave details. ``validateLayout`` enforces the
    /// preconditions this relies on.
    static func copyPrefix(from source: NDArray, to destination: inout NDArray, copyLength: Int) {
        let sourceShape = source.shape
        let destinationShape = destination.shape
        let sequenceDimension = sourceShape.count - 1
        let sourceSequence = sourceShape[sequenceDimension]
        let destinationSequence = destinationShape[sequenceDimension]
        let factor = source.interleaveLayout?.factor ?? 1
        precondition(
            copyLength <= sourceSequence && copyLength <= destinationSequence,
            "copyPrefix overflow: \(copyLength) into \(sourceSequence) → \(destinationSequence)")

        let groupCount = sourceShape.reduce(1, *) / sourceSequence / factor
        let sourceGroupStride = sourceSequence * factor
        let destinationGroupStride = destinationSequence * factor
        copyBlockPrefixes(
            from: source, to: &destination, blockCount: groupCount,
            sourceBlockStride: sourceGroupStride, destinationBlockStride: destinationGroupStride,
            runElements: copyLength * factor)
    }
}

// MARK: - Sliding-window ring

/// Geometry of a sliding-window ring cache, which bounds how far the written
/// prefix can be rewound.
struct SlidingRing: Equatable {
    /// Ring depth `S`: the sequence extent of the sliding key cache.
    let depth: Int
    let window: Int

    /// Name of the state whose sequence extent is the ring depth.
    static let keyCacheName = "sliding_key_cache"

    /// Whether the written prefix can be rewound from `processed` to `target` positions.
    ///
    /// The ring holds each position `p` at slot `p % depth`, so positions written
    /// after the rewind point overwrite the slots of earlier ones; the next query at
    /// `target` still needs the `window - 1` keys before it, which survive only while
    /// at most `depth - window + 1` positions have been written past it.
    func allowsRewind(processed: Int, to target: Int) -> Bool {
        target == 0 || processed - target <= depth - window + 1
    }
}

// MARK: - Handler set

/// The static states of one asset, split by lifecycle.
///
/// Two named slots rather than an array: views inserted from a `for` loop would
/// escape the loop variable's scope.
struct StaticStateSet {
    /// States whose backing buffer scales with the context bucket, right-sized per session.
    let bucketed: BucketedStaticState?
    /// States with one buffer across every bucket, allocated at the maximum.
    let fixed: FixedStaticState?
    /// The sliding-window ring among the fixed states, when the asset has one.
    let slidingRing: SlidingRing?

    /// Whether the states can be truncated from `processed` back to `target`
    /// positions. Flat caches always can; a sliding-window ring only while it still
    /// holds the keys the next query needs.
    func canTruncate(processed: Int, to target: Int) -> Bool {
        slidingRing?.allowsRewind(processed: processed, to: target) ?? true
    }

    /// Lay out the bucketed states for the bucket about to run. Fixed states
    /// need nothing: one buffer serves every bucket, and `bind` slices it.
    func prepare(contextBucket: Int, writtenTokenCount: Int) throws {
        try bucketed?.prepare(contextBucket: contextBucket, writtenTokenCount: writtenTokenCount)
    }

    func reset() {
        bucketed?.reset()
        fixed?.reset()
    }
}

// MARK: - Factory

/// Builds the ``StaticStateSet`` for a static-shape asset.
///
/// A state whose backing buffer (byte count and strides) is identical across
/// every context bucket is fixed; one whose buffer varies is bucketed. The
/// declared shape doesn't decide: a model compiled against one max-context cache
/// declares a shrinking shape per bucket over the same allocation.
enum StaticStateFactory {
    /// - Parameters:
    ///   - descriptorsByContext: Context bucket → a representative function
    ///     descriptor for that bucket (any query length; states don't vary with it).
    ///   - referenceDescriptor: The largest-context descriptor, used to enumerate
    ///     state names and to size fixed states.
    ///   - slidingWindow: The model's sliding-attention window, if it has one. With a
    ///     ``SlidingRing/keyCacheName`` state it describes the set's sliding ring.
    static func makeStateSet(
        descriptorsByContext: [Int: InferenceFunctionDescriptor],
        referenceDescriptor: InferenceFunctionDescriptor,
        slidingWindow: Int?
    ) throws -> StaticStateSet {
        let names = referenceDescriptor.stateNames

        var bucketedNames: [String] = []
        var fixedStates: [(name: String, descriptor: NDArrayDescriptor)] = []

        for name in names {
            guard case .ndArray(let reference) = referenceDescriptor.stateDescriptor(of: name) else {
                continue
            }

            let isBucketed = storageVariesByContext(
                name: name, descriptorsByContext: descriptorsByContext)
            if isBucketed {
                bucketedNames.append(name)
            } else {
                fixedStates.append((name, reference))
            }
        }

        var bucketed: BucketedStaticState?
        if !bucketedNames.isEmpty {
            var perContext: [Int: [String: NDArrayDescriptor]] = [:]
            for (context, descriptor) in descriptorsByContext {
                var byName: [String: NDArrayDescriptor] = [:]
                for name in bucketedNames {
                    guard case .ndArray(let stateDescriptor) = descriptor.stateDescriptor(of: name) else {
                        continue
                    }
                    byName[name] = stateDescriptor
                }
                perContext[context] = byName
            }
            CLILogger.log("Bucketed (right-sized) states: \(bucketedNames.sorted())")
            bucketed = try BucketedStaticState(
                stateNames: bucketedNames, descriptorsByContext: perContext)
        }

        var fixed: FixedStaticState?
        if !fixedStates.isEmpty {
            CLILogger.log("Fixed states: \(fixedStates.map(\.name).sorted())")
            fixed = FixedStaticState(states: fixedStates)
        }

        var slidingRing: SlidingRing?
        if let slidingWindow,
            case .ndArray(let ring) = referenceDescriptor.stateDescriptor(of: SlidingRing.keyCacheName),
            let depth = ring.shape.last
        {
            slidingRing = SlidingRing(depth: depth, window: slidingWindow)
        }

        return StaticStateSet(bucketed: bucketed, fixed: fixed, slidingRing: slidingRing)
    }

    /// Whether a state needs its own buffer per context bucket.
    ///
    /// Compares physical layout — byte count and strides — not the declared
    /// shape. A state that is one allocation viewed at a shrinking extent reports
    /// the same layout at every bucket and is therefore fixed.
    private static func storageVariesByContext(
        name: String,
        descriptorsByContext: [Int: InferenceFunctionDescriptor]
    ) -> Bool {
        var layouts: [StorageLayout] = []
        for (_, descriptor) in descriptorsByContext {
            guard case .ndArray(let stateDescriptor) = descriptor.stateDescriptor(of: name) else {
                continue
            }
            layouts.append(
                StorageLayout(
                    byteCount: stateDescriptor.minimumByteCount,
                    strides: stateDescriptor.preferredStrides))
        }
        return footprintVaries(layouts)
    }

    /// One bucket's physical footprint for a state: what it costs and how it is
    /// addressed. Deliberately excludes `shape`, which is the *view* extent the
    /// running program declares and shrinks with the bucket even when the buffer
    /// underneath does not.
    struct StorageLayout: Equatable {
        var byteCount: Int
        var strides: [Int]
    }

    /// The classification rule, over plain footprints so it is testable
    /// without a model (`InferenceFunctionDescriptor` can't be built in a test).
    static func footprintVaries(_ layouts: [StorageLayout]) -> Bool {
        guard let first = layouts.first else { return false }
        return layouts.contains { $0 != first }
    }
}

// MARK: - Run helper

/// Run one static-shape step with the asset's states bound.
///
/// Mirrors ``runWithStates`` for the dynamic engines: binding and the `run` call
/// stay in one straight-line scope, and `_unsafeEscapeMutableViews` detaches the
/// lifetime dependency so the views survive the `await`.
func runStaticStep(
    function: InferenceFunction,
    descriptor: InferenceFunctionDescriptor,
    inputs: [String: NDArray],
    states stateSet: StaticStateSet
) async throws -> InferenceFunction.Outputs {
    var states = InferenceFunction.MutableViews()
    stateSet.bucketed?.bind(into: &states, for: descriptor)
    stateSet.fixed?.bind(into: &states, for: descriptor)
    return try await function.run(
        inputs: inputs,
        states: _unsafeEscapeMutableViews(consume states),
        outputViews: InferenceFunction.MutableViews())
}
