// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIImageSegmenter
import CoreAIShared
import CoreGraphics
import Foundation
import ImageIO

@main
struct ImageSegmenterCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image-segmenter",
        abstract: "Run image segmentation using a text prompt (SAM3) or point/box prompts (EfficientSAM)."
    )

    // MARK: - Options

    @Option(name: .long, help: "Path to the model dir.")
    var model: String

    @Option(name: .long, help: "Path to the input image.")
    var image: String?

    // Query mode — exactly one of --prompt, --point, or --segment-everything must be provided.

    @Option(name: .long, help: "Text prompt describing the object to segment (SAM3).")
    var prompt: String?

    @Option(
        name: .long,
        help: "Point prompt as 'x,y' in input-image pixel coordinates (repeatable). EfficientSAM only."
    )
    var point: [String] = []

    @Option(
        name: .long,
        help: "Label for each --point: foreground (default), background, box-top-left, box-bottom-right (repeatable)."
    )
    var pointLabel: [String] = []

    @Flag(
        name: .long,
        help: "Segment without prompts — the engine uses a center foreground point. EfficientSAM only."
    )
    var segmentEverything: Bool = false

    @Option(
        name: .customLong("queries-json"),
        help: """
            Path to a JSON file with multiple point queries. Format: \
            [[{"x":N,"y":N,"label":"foreground"}, ...], ...] — outer array is queries, \
            inner array is points per query. Label is optional (defaults to foreground); \
            accepted values: foreground, background, box-top-left, box-bottom-right. \
            EfficientSAM only.
            """
    )
    var queriesJson: String?

    @Option(name: .long, help: "Maximum number of segments to process and return.")
    var maxSegments: Int = 5

    @Option(name: .long, help: "Mask sigmoid activation threshold (0–1).")
    var maskThreshold: Float = 0.5

    @Flag(name: .long, help: "Run a warmup pass before timed inference.")
    var warmup: Bool = false

    @Option(name: .long, help: "Write JSON results to this path.")
    var outputJson: String?

    @Option(
        name: .long,
        help: "Output PNG path. Defaults to output_<timestamp>.png in the current directory."
    )
    var outputPath: String?

    @Flag(name: .long, help: "Print verbose progress information.")
    var verbose: Bool = false

    @Option(
        name: .customLong("parity-test"),
        help: """
            Path to a parity-data dir containing source_image.npy + input_ids.npy + \
            ref_<output>.npy files. \
            Reconstructs the image from source_image.npy, drives the full \
            ImagePreprocessor + engine path with the supplied tokens, and compares \
            each raw model output against its reference via PSNR + cosine similarity.
            """
    )
    var parityTest: String?

    @Option(
        name: .customLong("psnr-floor"),
        help: "Minimum PSNR (dB) per output in --parity-test mode. The run fails if any output falls below this."
    )
    var psnrFloor: Float = 30.0

    @Option(
        name: .customLong("cosine-floor"),
        help: "Minimum cosine similarity per output in --parity-test mode."
    )
    var cosineFloor: Float = 0.999

    // MARK: - Validation

    func validate() throws {
        if parityTest != nil {
            // Parity mode loads its image from <dir>/source_image.npy and its tokens
            // from <dir>/input_ids.npy, so the normal --image / prompt-or-point
            // requirements don't apply.
            return
        }

        guard image != nil else {
            throw ValidationError("--image is required (unless --parity-test is set).")
        }

        let modes = [
            prompt != nil,
            !point.isEmpty || segmentEverything,
            queriesJson != nil,
        ].filter { $0 }.count
        if modes == 0 {
            throw ValidationError(
                "Specify one of: --prompt, --point, --segment-everything, or --queries-json."
            )
        }
        if modes > 1 {
            throw ValidationError(
                "--prompt, --point/--segment-everything, and --queries-json are mutually exclusive."
            )
        }
        if segmentEverything && !point.isEmpty {
            throw ValidationError("--segment-everything and --point are mutually exclusive.")
        }
        if !pointLabel.isEmpty && pointLabel.count != point.count {
            throw ValidationError(
                "--point-label count (\(pointLabel.count)) must match --point count (\(point.count))."
            )
        }
    }

    // MARK: - Run

    func run() async throws {
        if let parityDir = parityTest {
            try await runParityTest(dataDir: URL(fileURLWithPath: parityDir))
            return
        }

        guard let imagePath = image else {
            // Already enforced in validate(); guard to satisfy the compiler.
            throw ValidationError("--image is required.")
        }
        if verbose {
            CLILogger.setLevel(to: 1)
        }

        if verbose { print("Creating image segmenter...") }
        let runner = try await ImageSegmenter(resourcesAt: model)

        let cgImage = try loadCGImage(from: imagePath)
        if verbose { print("Loaded image: \(cgImage.width)×\(cgImage.height)") }

        let params = SegmentationParameters(maskThreshold: maskThreshold, maxSegments: maxSegments)

        if warmup {
            if verbose { print("Running warmup...") }
            try await runner.warmup()
        }

        let start = SuspendingClock().now
        let (response, promptBoxes) = try await runInference(runner: runner, image: cgImage, params: params)
        let elapsed = SuspendingClock().now - start
        if verbose { print("Inference time (including pre and post processing): \(elapsed)") }

        let results = makeDetectionResults(from: response)
        try emitResults(results, totalSegments: response.segments.count)

        try renderAndOpenOutput(response: response, baseImage: cgImage, promptBoxes: promptBoxes)
    }

    // MARK: - Parity test

    /// End-to-end ImageSegmenter parity check against PyTorch SAM3 references.
    ///
    /// Routes through the same `CoreAISegmentationEngine.segment(...)` call the
    /// production path uses, so `ImagePreprocessor` is in scope. Tokens come
    /// pre-computed from Python (via `input_ids.npy`) to isolate the test from
    /// any `CLIPTokenizer` drift — tokenizer parity is a separate concern.
    private func runParityTest(dataDir: URL) async throws {
        print("Running SAM3 parity test with data from: \(dataDir.path)")

        // Precheck: list missing reference files up front so the user gets an
        // actionable error instead of a generic Foundation "no such file"
        // partway through.
        let requiredFiles = [
            "source_image.npy",
            "ref_pixel_values.npy",
            "input_ids.npy",
            "prompt.txt",
            "ref_pred_masks.npy",
            "ref_pred_boxes.npy",
            "ref_pred_logits.npy",
            "ref_presence_logits.npy",
            "ref_semantic_seg.npy",
        ]
        let missing = requiredFiles.filter {
            !FileManager.default.fileExists(atPath: dataDir.appendingPathComponent($0).path)
        }
        if !missing.isEmpty {
            throw ValidationError(
                """
                --parity-test dir is missing required reference files:
                  \(missing.joined(separator: ", "))
                Generate reference data using the parity trace script.
                """
            )
        }

        // Load the bundle and resolve the main asset URL. Mirrors the
        // ModelBundle path in ImageSegmenter+CoreAI.swift's convenience init —
        // we construct the engine ourselves so we can capture raw model outputs
        // before SegmentationPostprocessor collapses them into top-N segments.
        let bundle = try ModelBundle(from: model)
        guard bundle.kind == .segmenter else {
            throw ValidationError(
                "Bundle at \(model) has kind \(bundle.kind.rawValue), expected segmenter"
            )
        }
        let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        let params = SegmentationParameters(maskThreshold: maskThreshold, maxSegments: maxSegments)
        let engine = try await CoreAISegmentationEngine(parameters: params, modelURL: modelURL)

        // Load inputs.
        let refPixelValues = try NpyArray.load(dataDir.appendingPathComponent("ref_pixel_values.npy"))
        let inputIdsArray = try NpyArray.load(dataDir.appendingPathComponent("input_ids.npy"))
        let cgImage = try makeCGImage(
            fromHWCUInt8: try NpyArray.load(dataDir.appendingPathComponent("source_image.npy"))
        )
        let prompt = try String(contentsOf: dataDir.appendingPathComponent("prompt.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Run Swift's preprocessing + tokenization. The engine call below also
        // runs ImagePreprocessor internally; this separate invocation is what
        // makes the pixel_values comparison row work as an independent check.
        guard refPixelValues.shape.count == 4 else {
            throw ValidationError(
                "ref_pixel_values.npy must have 4 dimensions (B, C, H, W), got shape \(refPixelValues.shape)"
            )
        }
        let height = refPixelValues.shape[2]
        let width = refPixelValues.shape[3]
        let actualPixelValues = try ImagePreprocessor(
            targetSize: CGSize(width: width, height: height),
            mean: params.normalizationMeans,
            std: params.normalizationStds,
            rescaleFactor: 1.0
        ).preprocessCHW(cgImage: cgImage)
        let swiftTokens = try CLIPTokenizer(folder: bundle.bundlePath.appending(path: "tokenizer"))
            .encode(prompt, contextLength: params.tokenizerContextLength)

        // Feed the model Python's tokens (apples-to-apples model parity — Swift
        // tokenizer drift is reported separately by the tokenizer row).
        let pyTokens = try inputIdsArray.asInt32()
        let batchSize = inputIdsArray.shape.first ?? 1
        let sequenceLength = inputIdsArray.shape.count > 1 ? inputIdsArray.shape[1] : pyTokens.count
        let batched = (0..<batchSize).map { batchIndex in
            Array(pyTokens[(batchIndex * sequenceLength)..<((batchIndex + 1) * sequenceLength)])
        }
        let output = try await engine.segment(
            image: cgImage, textQuery: .tokens(batched), parameters: params
        )

        // Build rows: tokenizer first, then preprocessing, then each model output.
        var rows: [ParityRow] = [
            tokenizerRow(swift: swiftTokens, py: Array(pyTokens.prefix(sequenceLength))),
            metricRow(name: "pixel_values", actual: actualPixelValues, ref: try refPixelValues.asFloat()),
        ]
        for (name, actual) in [
            ("pred_masks", output.predictedMasks),
            ("pred_boxes", output.predictedBoxes),
            ("pred_logits", output.predictedLogits),
            ("presence_logits", output.presenceLogits),
            ("semantic_seg", output.semanticSegment),
        ] {
            let ref = try NpyArray.load(dataDir.appendingPathComponent("ref_\(name).npy")).asFloat()
            rows.append(metricRow(name: name, actual: actual, ref: ref))
        }

        print("\n=== SAM3 ImageSegmenter parity ===")
        for row in rows {
            let label = row.name.padding(toLength: 16, withPad: " ", startingAt: 0)
            print("  \(label) \(row.status)  \(row.ok ? "✓" : "✗")")
        }
        if rows.contains(where: { !$0.ok }) { throw ExitCode.failure }
        print("All outputs within tolerance (PSNR≥\(psnrFloor) dB, cosine≥\(cosineFloor)).")
    }

    // MARK: - Inference

    /// Dispatch to text-based or point-based segmentation based on which CLI flag was supplied.
    /// Returns the model's response and any user-drawn prompt boxes (used to stroke the output PNG).
    private func runInference(
        runner: ImageSegmenter, image cgImage: CGImage, params: SegmentationParameters
    ) async throws -> (SegmentationResponse, [CGRect]) {
        if let promptText = prompt {
            if verbose { print("Running inference with prompt: \"\(promptText)\"") }
            let response = try await runner.segment(image: cgImage, prompt: promptText, parameters: params)
            return (response, [])
        }

        let pq: PointQuery
        if let jsonPath = queriesJson {
            pq = try parseQueriesJson(at: jsonPath)
            if verbose {
                let totalPoints = pq.queries.reduce(0) { $0 + $1.count }
                print(
                    "Running inference with \(pq.queries.count) queries (\(totalPoints) total points) from \(jsonPath)."
                )
            }
        } else if segmentEverything {
            pq = PointQuery()
            if verbose { print("Running inference with segment-everything grid.") }
        } else {
            let resolvedPoints = try parsePoints()
            pq = PointQuery(points: resolvedPoints)
            if verbose {
                print("Running inference with one query of \(resolvedPoints.count) point(s).")
            }
        }
        let response = try await runner.segment(image: cgImage, pointQuery: pq, parameters: params)
        return (response, Self.boxes(fromQueries: pq.queries))
    }

    // MARK: - Result output

    private func makeDetectionResults(from response: SegmentationResponse) -> [DetectionResult] {
        response.segments.map { seg in
            DetectionResult(
                score: seg.score,
                box: DetectionResult.BoxResult(
                    x: seg.box.origin.x, y: seg.box.origin.y,
                    width: seg.box.size.width, height: seg.box.size.height
                ),
                maskForegroundPixels: seg.mask.filter { $0 }.count
            )
        }
    }

    /// Print summary to stdout, or write JSON to `--output-json` when set.
    private func emitResults(_ results: [DetectionResult], totalSegments: Int) throws {
        if let jsonPath = outputJson {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(results)
            try data.write(to: URL(fileURLWithPath: NSString(string: jsonPath).expandingTildeInPath))
            print("Results written to \(jsonPath)")
            return
        }
        print("\nSegments (\(totalSegments)):")
        for (i, r) in results.enumerated() {
            print(
                "  [\(i)] score=\(String(format: "%.4f", r.score))"
                    + "  box=(\(Int(r.box.x)),\(Int(r.box.y)),\(Int(r.box.width))×\(Int(r.box.height)))"
                    + "  foreground_px=\(r.maskForegroundPixels)"
            )
        }
    }

    /// Render the result overlay, write the PNG, and open it in the default viewer.
    private func renderAndOpenOutput(
        response: SegmentationResponse, baseImage: CGImage, promptBoxes: [CGRect]
    ) throws {
        let resolvedOutputPath = outputPath ?? "output_\(Int(Date().timeIntervalSince1970)).png"
        guard let rendered = renderOutput(response: response, baseImage: baseImage, promptBoxes: promptBoxes)
        else {
            print("Warning: could not render output image (no segments or maps produced).")
            return
        }
        try writeImage(rendered, to: resolvedOutputPath)
        print("Output image written to \(resolvedOutputPath)")
        #if os(macOS)
        openFile(at: resolvedOutputPath)
        #endif
    }

    // MARK: - Rendering

    /// Renders a semantic overlay when the model produced a probability map, otherwise
    /// falls back to an instance mask overlay. Strokes any prompt boxes on top.
    private func renderOutput(
        response: SegmentationResponse, baseImage: CGImage, promptBoxes: [CGRect]
    ) -> CGImage? {
        let base: CGImage?
        if let map = response.probabilityMap {
            base = SegmentationVisualization.renderSemanticOverlay(onto: baseImage, map: map)
        } else if !response.segments.isEmpty {
            base = SegmentationVisualization.renderInstanceMasks(onto: baseImage, segments: response.segments)
        } else if !promptBoxes.isEmpty {
            base = baseImage  // render boxes alone if there's nothing else
        } else {
            base = nil
        }
        guard let composed = base else { return nil }
        return SegmentationVisualization.renderPromptBoxes(onto: composed, boxes: promptBoxes)
    }

    /// Build pixel-space `CGRect`s (top-left origin) from any per-query `[.boxTopLeft, .boxBottomRight]` pairs.
    private static func boxes(fromQueries queries: [[PointQuery.Point]]) -> [CGRect] {
        queries.compactMap { q -> CGRect? in
            guard let tl = q.first(where: { $0.label == .boxTopLeft }),
                let br = q.first(where: { $0.label == .boxBottomRight })
            else { return nil }
            return CGRect(
                x: CGFloat(tl.x), y: CGFloat(tl.y),
                width: CGFloat(br.x - tl.x), height: CGFloat(br.y - tl.y)
            )
        }
    }

    // MARK: - Helpers

    private func parsePoints() throws -> [PointQuery.Point] {
        try point.enumerated().map { idx, raw in
            let parts = raw.split(separator: ",")
            guard parts.count == 2,
                let x = Float(parts[0].trimmingCharacters(in: .whitespaces)),
                let y = Float(parts[1].trimmingCharacters(in: .whitespaces))
            else {
                throw ValidationError("Invalid --point '\(raw)': expected 'x,y' in pixel coordinates.")
            }
            let label: PointQuery.Label =
                idx < pointLabel.count
                ? try parseLabel(pointLabel[idx])
                : .foreground
            return PointQuery.Point(x: x, y: y, label: label)
        }
    }

    private struct QueryPointJSON: Decodable {
        let x: Float
        let y: Float
        let label: String?
    }

    private func parseQueriesJson(at path: String) throws -> PointQuery {
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ValidationError(
                "Cannot read --queries-json file at \(path): \(error.localizedDescription)"
            )
        }
        let raw: [[QueryPointJSON]]
        do {
            raw = try JSONDecoder().decode([[QueryPointJSON]].self, from: data)
        } catch {
            throw ValidationError(
                "Invalid --queries-json: expected [[{\"x\":N,\"y\":N,\"label\":\"...\"}, ...], ...]. "
                    + "\(error.localizedDescription)"
            )
        }
        if raw.isEmpty {
            throw ValidationError(
                "--queries-json file at \(path) contains no queries. "
                    + "To run segment-everything, use --segment-everything instead."
            )
        }
        let queries: [[PointQuery.Point]] = try raw.map { query in
            try query.map { pt in
                let label: PointQuery.Label =
                    try pt.label.map { try parseLabel($0) } ?? .foreground
                return PointQuery.Point(x: pt.x, y: pt.y, label: label)
            }
        }
        return PointQuery(queries: queries)
    }

    private func parseLabel(_ raw: String) throws -> PointQuery.Label {
        switch raw.lowercased() {
        case "foreground", "fg", "1": return .foreground
        case "background", "bg", "0": return .background
        case "box-top-left", "boxtopleft", "tl", "2": return .boxTopLeft
        case "box-bottom-right", "boxbottomright", "br", "3": return .boxBottomRight
        default:
            throw ValidationError(
                "Unknown --point-label '\(raw)'. Use: foreground, background, box-top-left, box-bottom-right."
            )
        }
    }

    private func writeImage(_ cgImage: CGImage, to path: String) throws {
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            throw ValidationError("Cannot create image destination at \(path)")
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ValidationError("Failed to write image to \(path)")
        }
    }

    #if os(macOS)
    private func openFile(at path: String) {
        let expanded = NSString(string: path).expandingTildeInPath
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = [expanded]
        try? process.run()
    }
    #endif

    // MARK: - Parity helpers (.npy reader + CGImage builder + metrics)

    /// Minimal .npy reader covering the dtypes this CLI consumes:
    /// `float16` / `float32` (model outputs + pixel_values), `int32` (input_ids),
    /// `uint8` (source_image).
    private struct NpyArray {
        enum DType { case float16, float32, int32, uint8 }
        let shape: [Int]
        let dtype: DType
        let data: Data

        static func load(_ url: URL) throws -> NpyArray {
            let raw = try Data(contentsOf: url)
            guard raw.count > 10, raw[0] == 0x93, raw[1] == 0x4E else {
                throw ValidationError("Not a .npy file: \(url.path)")
            }
            let version = raw[6]
            let headerLen: Int
            let headerStart: Int
            if version == 1 {
                headerLen = Int(raw[8]) | (Int(raw[9]) << 8)
                headerStart = 10
            } else {
                headerLen =
                    Int(raw[8]) | (Int(raw[9]) << 8) | (Int(raw[10]) << 16) | (Int(raw[11]) << 24)
                headerStart = 12
            }
            let dataStart = headerStart + headerLen
            let header = String(data: raw[headerStart..<dataStart], encoding: .ascii) ?? ""

            let dtype: DType
            if header.contains("f2") {
                dtype = .float16
            } else if header.contains("f4") {
                dtype = .float32
            } else if header.contains("i4") {
                dtype = .int32
            } else if header.contains("u1") {
                dtype = .uint8
            } else {
                throw ValidationError("Unsupported .npy dtype in \(url.lastPathComponent)")
            }

            return NpyArray(
                shape: Self.parseShape(from: header),
                dtype: dtype,
                data: raw.subdata(in: dataStart..<raw.count)
            )
        }

        private static func parseShape(from header: String) -> [Int] {
            guard let start = header.range(of: "("),
                let end = header.range(of: ")", range: start.upperBound..<header.endIndex)
            else { return [] }
            return header[start.upperBound..<end.lowerBound]
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        }

        func asFloat() throws -> [Float] {
            let n = shape.reduce(1, *)
            var out = [Float](repeating: 0, count: n)
            data.withUnsafeBytes { ptr in
                switch dtype {
                case .float16:
                    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
                    let src = ptr.bindMemory(to: Float16.self)
                    for i in 0..<n { out[i] = Float(src[i]) }
                    #else
                    fatalError("Float16 is not supported on this platform")
                    #endif
                case .float32:
                    let src = ptr.bindMemory(to: Float.self)
                    for i in 0..<n { out[i] = src[i] }
                case .int32:
                    let src = ptr.bindMemory(to: Int32.self)
                    for i in 0..<n { out[i] = Float(src[i]) }
                case .uint8:
                    let src = ptr.bindMemory(to: UInt8.self)
                    for i in 0..<n { out[i] = Float(src[i]) }
                }
            }
            return out
        }

        func asInt32() throws -> [Int32] {
            guard case .int32 = dtype else {
                throw ValidationError("Cannot read \(dtype) as Int32")
            }
            let n = shape.reduce(1, *)
            var out = [Int32](repeating: 0, count: n)
            data.withUnsafeBytes { ptr in
                let src = ptr.bindMemory(to: Int32.self)
                for i in 0..<n { out[i] = src[i] }
            }
            return out
        }

        func asUInt8() throws -> [UInt8] {
            guard case .uint8 = dtype else {
                throw ValidationError("Cannot read \(dtype) as UInt8")
            }
            let n = shape.reduce(1, *)
            var out = [UInt8](repeating: 0, count: n)
            data.withUnsafeBytes { ptr in
                let src = ptr.bindMemory(to: UInt8.self)
                for i in 0..<n { out[i] = src[i] }
            }
            return out
        }
    }

    /// Build a `CGImage` from an HWC RGB uint8 npy array. Used so the parity test
    /// exercises the same CGImage decode + ImagePreprocessor path as the normal
    /// `--image foo.png` flow.
    private func makeCGImage(fromHWCUInt8 arr: NpyArray) throws -> CGImage {
        guard arr.shape.count == 3, arr.shape[2] == 3 else {
            throw ValidationError(
                "source_image.npy must be HWC RGB uint8 with shape [H, W, 3]; got \(arr.shape)"
            )
        }
        let h = arr.shape[0]
        let w = arr.shape[1]
        let rgb = try arr.asUInt8()
        // RGB → RGBX with alpha=255 (the ImagePreprocessor expects RGBX).
        var pixels = [UInt8](repeating: 0, count: h * w * 4)
        for i in 0..<(h * w) {
            pixels[i * 4 + 0] = rgb[i * 3 + 0]
            pixels[i * 4 + 1] = rgb[i * 3 + 1]
            pixels[i * 4 + 2] = rgb[i * 3 + 2]
            pixels[i * 4 + 3] = 255
        }
        let bytesPerRow = w * 4
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            throw ValidationError("Could not build CGDataProvider for source image")
        }
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard
            let cg = CGImage(
                width: w, height: h,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                provider: provider,
                decode: nil, shouldInterpolate: false, intent: .defaultIntent
            )
        else {
            throw ValidationError("Could not build CGImage from source_image.npy")
        }
        return cg
    }

    /// Peak-signal-to-noise ratio in dB. `peak` is the reference tensor's max
    /// absolute value, so the metric scales with the signal — meaningful for both
    /// (0,1) sigmoid masks and unbounded logits.
    private func psnr(_ actual: [Float], _ ref: [Float]) -> Float {
        guard actual.count == ref.count, !ref.isEmpty else { return -.infinity }
        var mse: Double = 0
        var peak: Double = 0
        for i in 0..<ref.count {
            let r = Double(ref[i])
            let d = Double(actual[i]) - r
            mse += d * d
            let ar = abs(r)
            if ar > peak { peak = ar }
        }
        mse /= Double(ref.count)
        if mse == 0 { return .infinity }
        if peak == 0 { return -.infinity }
        return Float(10.0 * log10((peak * peak) / mse))
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Double = 0
        var normA: Double = 0
        var normB: Double = 0
        for i in 0..<a.count {
            dot += Double(a[i]) * Double(b[i])
            normA += Double(a[i]) * Double(a[i])
            normB += Double(b[i]) * Double(b[i])
        }
        let denom = (normA * normB).squareRoot()
        return denom > 0 ? Float(dot / denom) : 0
    }

    /// Return the leading slice of `ids` up to and including the first EOT
    /// (token id 49407) after position 0. Trailing pad tokens are excluded.
    /// If no EOT is found, returns the full sequence.
    private func contentPrefix(_ ids: [Int32]) -> [Int32] {
        for i in 1..<ids.count where ids[i] == CLIPTokenizer.eotTokenId {
            return Array(ids[0...i])
        }
        return ids
    }

    // MARK: - Parity report rows

    /// One row in the `=== SAM3 ImageSegmenter parity ===` output.
    private struct ParityRow {
        let name: String
        let status: String  // pre-formatted (PSNR/cosine or "N/N tokens match")
        let ok: Bool
    }

    /// PSNR + cosine row used for the pixel_values + 5 model outputs.
    private func metricRow(name: String, actual: [Float], ref: [Float]) -> ParityRow {
        if actual.count != ref.count {
            return ParityRow(
                name: name,
                status: "✗ element-count mismatch — actual \(actual.count) vs ref \(ref.count)",
                ok: false
            )
        }
        let p = psnr(actual, ref)
        let c = cosineSimilarity(actual, ref)
        let ok = (p.isInfinite || p >= psnrFloor) && c >= cosineFloor
        let psnrStr = p.isInfinite ? "INF" : String(format: "%.2f", p)
        return ParityRow(
            name: name,
            status: "PSNR=\(psnrStr) dB  cosine=\(String(format: "%.5f", c))",
            ok: ok
        )
    }

    /// Exact-match row for the tokenizer; reports first-divergence on failure.
    private func tokenizerRow(swift: [Int32], py: [Int32]) -> ParityRow {
        let s = contentPrefix(swift)
        let r = contentPrefix(py)
        if s == r {
            return ParityRow(name: "tokenizer", status: "\(s.count)/\(s.count) tokens match", ok: true)
        }
        let n = min(s.count, r.count)
        let firstDiff = (0..<n).first(where: { s[$0] != r[$0] })
        let detail =
            firstDiff.map { "first diff at index \($0): swift=\(s[$0]) vs ref=\(r[$0])" }
            ?? "length mismatch: swift=\(s.count) vs ref=\(r.count)"
        return ParityRow(
            name: "tokenizer",
            status: "\(min(s.count, r.count))/\(max(s.count, r.count)) — \(detail)",
            ok: false
        )
    }
}

// MARK: - Helpers

private func loadCGImage(from path: String) throws -> CGImage {
    let expanded = NSString(string: path).expandingTildeInPath
    let url = URL(fileURLWithPath: expanded)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        throw ValidationError("Cannot open image at \(path)")
    }
    guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw ValidationError("Cannot decode image at \(path)")
    }
    return cgImage
}

// MARK: - JSON output types

private struct DetectionResult: Codable {
    let score: Float
    let box: BoxResult
    let maskForegroundPixels: Int

    struct BoxResult: Codable {
        let x, y, width, height: Double
    }
}
