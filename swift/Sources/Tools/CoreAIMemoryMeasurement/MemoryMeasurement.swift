// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIShared
import Foundation

/// The running tool's executable name, such as "llm-benchmark" or "llm-runner", for labels and prompts.
package let toolName = ProcessInfo.processInfo.processName

/// Command-line options that turn on memory measurement.
package struct MemoryOptions: ParsableArguments {
    @Flag(
        name: .customLong("measure-peak-memory"),
        help:
            "Report peak physical memory for prepare (incl. compile), warmup, and the run that follows. Combine with --clear-coreai-cache to measure a fresh compile. On macOS, also includes the out-of-process Neural Engine compiler, which runs as root: \(toolName) asks for your sudo credentials to start a helper that only reads memory."
    )
    package var measurePeakMemory: Bool = false

    @Option(
        name: .customLong("save-memory-csv"),
        help:
            "Write memory over time to this CSV path: for each 100 ms, the highest peak memory and its parts, the data behind --print-memory-chart (implies --measure-peak-memory)"
    )
    package var saveMemoryCSV: String?

    @Flag(
        name: .customLong("print-memory-chart"),
        help:
            "Print a chart of the in-process memory and peak memory over time: the highest of the 5 ms polls in each 100 ms (implies --measure-peak-memory)"
    )
    package var printMemoryChart: Bool = false

    @Flag(
        name: .customLong("no-sudo"),
        help:
            "With memory measurement, don't ask for sudo. The Neural Engine services aren't measured unless already root"
    )
    package var noSudo: Bool = false

    package init() {}

    package var measuresMemory: Bool { measurePeakMemory || saveMemoryCSV != nil || printMemoryChart }
}

/// Runs the memory probes for one run of a tool and turns their readings into a `MemoryReport`.
///
/// The tool calls `begin(_:)` with "prepare", then optionally "warmup", then one or more phases of the run
/// itself, such as llm-benchmark's "trial 1", "trial 2", … or llm-runner's "inference". Every phase other than
/// prepare and warmup counts toward the run, which the output calls `runPhase`.
package final class MemoryMeasurement {
    private let options: MemoryOptions
    private let runPhase: String
    private let tracker: PeakFootprintTracker
    private let modelDirectories: [String]
    /// Mapped model files that stayed resident through the end-of-run check in `finish(generate:)`; nil
    /// until then, or where the check can't run, and then all model files count.
    private var neededModelFiles: UInt64?
    #if os(macOS)
    private let neuralEngineProbe: ProcessMemoryProbe?
    #endif

    /// Starts the background probes; nil when `options` don't ask for memory measurement.
    /// `modelPath` is the model bundle, whose mapped files count toward the system memory. `runPhase` names
    /// the phases after warmup in the output, such as "trials" or "inference".
    package init?(options: MemoryOptions, modelPath: String, runPhase: String) {
        guard options.measuresMemory else { return nil }
        self.options = options
        self.runPhase = runPhase
        let modelDirectories = modelFileDirectories(modelPath: modelPath)
        self.modelDirectories = modelDirectories
        #if os(macOS)
        if options.noSudo {
            CLILogger.log(
                "WARNING: Without sudo, this tool cannot measure the Neural Engine services' memory use and it will not be included in the report.",
                level: 0)
        }
        let neuralEngineProbe = Self.rootReader(noSudo: options.noSudo).map {
            ProcessMemoryProbe(names: ProcessMemoryProbe.neuralEngineServiceNames, reader: $0)
        }
        self.neuralEngineProbe = neuralEngineProbe
        tracker = PeakFootprintTracker(modelDirectories: modelDirectories, neuralEngineProbe: neuralEngineProbe)
        #else
        tracker = PeakFootprintTracker(modelDirectories: modelDirectories)
        #endif
    }

    #if os(macOS)
    /// A reader for processes running as root, such as the Neural Engine services. If already running as root,
    /// they're read in-process. Otherwise, a helper gets started under sudo. nil when sudo is declined or
    /// `noSudo`.
    private static func rootReader(noSudo: Bool) -> (any ProcessMemoryReader)? {
        if geteuid() == 0 { return LocalProcessMemoryReader() }
        guard !noSudo else { return nil }
        CLILogger.log(
            "\nThe Neural Engine services run as root; sudo starts a helper that only reads their memory (--no-sudo skips).",
            level: 0
        )
        fflush(stdout)
        let reader = HelperProcessMemoryReader.launch()
        if reader == nil {
            CLILogger.log(
                "'sudo' command was cancelled or failed, as a result the Neural Engine services' memory measurement will be skipped and will not be included in the memory report.",
                level: 0)
        }
        return reader
    }
    #endif

    /// Ends the current phase (if any) and starts measuring `phase`.
    package func begin(_ phase: String) {
        tracker.begin(phase)
    }

    /// Stops measuring, finds which mapped model files the model needs, and builds the report.
    ///
    /// Mapped model files stay resident on an idle Mac whether or not anything reads them. So, as memory
    /// pressure would on iOS, this drops them, then calls `generate`, which should run the model briefly.
    /// What's resident afterwards is what the model needs: pages a driver wired, such as weights the GPU
    /// runs from, and pages it read again. Peak memory counts model files only up to that. It runs after the
    /// tracker stops, so it doesn't show in the chart, the CSV, or the phases.
    package func finish(generate: () async throws -> Void) async rethrows -> MemoryReport {
        let phases = tracker.finish()
        #if os(macOS)
        evictMappedFiles(in: modelDirectories)
        try await generate()
        neededModelFiles = mappedFiles()?.model(modelDirectories: modelDirectories)
        let measuredNeuralEngineServices = neuralEngineProbe != nil
        #else
        let measuredNeuralEngineServices = false
        #endif
        let byName = Dictionary(phases.map { ($0.name, $0) }, uniquingKeysWith: { $1 })
        return MemoryReport(
            runPhase: runPhase,
            hasWarmup: byName["warmup"] != nil,
            preparePeakMB: megabytes(byName["prepare"]?.peak ?? 0),
            warmupPeakMB: megabytes(byName["warmup"]?.peak ?? 0),
            runPeakMB: megabytes(phases.filter { isRunPhase($0.name) }.map(\.peak).max() ?? 0),
            lifetimePeakMB: megabytes(phases.last?.lifetimePeakAtEnd ?? 0),
            compressedPeakMB: megabytes(tracker.compressedPeak),
            peakPressureLevel: tracker.peakPressureLevel,
            measuredNeuralEngineServices: measuredNeuralEngineServices,
            system: systemReport(phases: phases)
        )
    }

    /// With `--print-memory-chart`, prints the memory chart; otherwise does nothing.
    package func chart(memory: MemoryReport, cacheHit: Bool) {
        guard options.printMemoryChart else { return }
        // Each phase's peaks and the parts of the peak memory, above the chart.
        print("\nMemory Chart:")
        print(String(repeating: "=", count: 50))
        for line in memory.detailedLines(cacheHit: cacheHit) { print(line) }
        defer { print(String(repeating: "=", count: 50)) }
        // The highest 5 ms poll in each 100 ms window: this process's footprint with the rest of the peak memory
        // stacked on top.
        let windows = tracker.timeline(neededModelFiles: neededModelFiles)
        let timeline = MemoryChart.Timeline(
            title: "Memory over time (MB)",
            samples: windows.map {
                let reading = $0.highestSystem
                return MemoryChart.Sample(
                    elapsedSeconds: reading.elapsedSeconds, phase: $0.phase, footprintMB: megabytes(reading.footprint),
                    uncountedMB: megabytes(reading.system(neededModelFiles: neededModelFiles) - reading.footprint))
            },
            // The summary's peak can be this process's kernel high-water mark, which no poll caught.
            labeledPeak: (memory.system.peakMB, memory.system.peakSeconds),
            // The kernel's high-water mark, at the time of the highest polled footprint.
            labeledFootprintPeak: windows.map(\.highestSystem).max { $0.footprint < $1.footprint }.map {
                (max(memory.lifetimePeakMB, megabytes($0.footprint)), $0.elapsedSeconds)
            })
        guard let text = MemoryChart.render(timeline) else { return }
        print("\n" + text)
    }

    /// With `--save-memory-csv`, writes the timeline the chart draws as CSV; otherwise does nothing.
    package func writeCSVIfRequested() throws {
        guard let path = options.saveMemoryCSV else { return }
        // One row per 100 ms window: its poll with the highest peak memory, and that poll's parts, named as in the
        // summary's breakdown and with the model files capped as there. The parts add up to `total_mb`. Blank
        // cells are parts that weren't measured.
        var csv =
            "time_s,phase,total_mb,in_process_mb,neural_engine_mb,model_files_in_use_mb,neural_engine_services_mb\n"
        func cell(_ bytes: UInt64?) -> String { bytes.map { String(format: "%.1f", megabytes($0)) } ?? "" }
        for window in tracker.timeline(neededModelFiles: neededModelFiles) {
            let reading = window.highestSystem
            let cells = [
                String(format: "%.3f", reading.elapsedSeconds), window.phase,
                cell(reading.system(neededModelFiles: neededModelFiles)), cell(reading.footprint),
                cell(reading.neuralNotCounted), cell(reading.countedModelFiles(neededModelFiles: neededModelFiles)),
                cell(reading.servicesGrowth),
            ]
            csv += cells.joined(separator: ",") + "\n"
        }
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try csv.write(to: url, atomically: true, encoding: .utf8)
        CLILogger.log("\nWrote: \(path)", level: 0)
    }

    // MARK: - Report

    private func systemReport(phases: [PeakFootprintTracker.Phase]) -> MemoryReport.System {
        func peakMB(_ matches: (String) -> Bool) -> Double {
            megabytes(
                phases.filter { matches($0.name) }.map { $0.systemPeak(neededModelFiles: neededModelFiles) }.max()
                    ?? 0)
        }
        let reading = phases.map { $0.systemReading(neededModelFiles: neededModelFiles) }
            .max { $0.system(neededModelFiles: neededModelFiles) < $1.system(neededModelFiles: neededModelFiles) }
        return MemoryReport.System(
            preparePeakMB: peakMB { $0 == "prepare" },
            warmupPeakMB: peakMB { $0 == "warmup" },
            runPeakMB: peakMB(isRunPhase),
            peakMB: peakMB { _ in true },
            peakSeconds: reading?.elapsedSeconds ?? 0,
            footprintMB: megabytes(reading?.footprint ?? 0),
            neuralEngineServicesGrowthMB: reading?.servicesGrowth.map(megabytes),
            neuralNotCountedMB: reading?.neuralNotCounted.map(megabytes),
            modelMappedFilesMB: megabytes(reading?.countedModelFiles(neededModelFiles: neededModelFiles) ?? 0))
    }
}

// MARK: - Terminal summary

extension MemoryReport {
    /// The summary level of the terminal output: the peak memory of the whole run, then any caveat that changes
    /// what that number means.
    package func summaryLines() -> [String] {
        var lines = [
            "Peak memory: " + fmtMB(system.peakMB) + " during " + peakMemoryPhase + " @ "
                + fmtSeconds(system.peakSeconds)
        ]
        if !measuredNeuralEngineServices {
            lines.append("   Neural Engine services not measured (they need sudo)")
        }
        if isUnderPressure {
            lines.append(
                "   ⚠️  Memory pressure: the system reached \(pressureName) level; \(fmtMB(compressedPeakMB)) was"
                    + " compressed. Model file pages may have been evicted and timings slowed; rerun with more free"
                    + " memory.")
        }
        return lines
    }

    /// The `--print-memory-chart` details, which `MemoryMeasurement.chart` prints above the chart: each phase's
    /// peaks, then the parts of the peak memory at its peak.
    package func detailedLines(cacheHit: Bool) -> [String] {
        let prepare = cacheHit ? "Prepare" : "Prepare (incl. specialization)"
        var lines: [String] = []
        let rows =
            [(prepare, system.preparePeakMB, preparePeakMB)]
            + (hasWarmup ? [("Warmup", system.warmupPeakMB, warmupPeakMB)] : [])
            + [(runPhaseTitle, system.runPeakMB, runPeakMB)]
        let title = "Peak by phase"
        let nameWidth = max(title.count, rows.map { $0.0.count + 2 }.max()!)
        func name(_ text: String) -> String { text + String(repeating: " ", count: nameWidth - text.count) }
        func cell(_ text: String) -> String { pad(text, 18) }
        lines.append(name(title) + cell("Peak memory") + cell("In-process peak"))
        for (phase, systemPeak, peak) in rows {
            lines.append(name("  " + phase) + cell(fmtMB(systemPeak)) + cell(fmtMB(peak)))
        }
        // The in-process part is the footprint at the peak, not the phase's in-process peak in the table.
        lines.append("At peak (" + fmtSeconds(system.peakSeconds) + "): " + system.breakdown(fmtMB))
        return lines
    }

    /// The peak memory and its parts for `--output-json`: the numbers behind `summaryLines()`'s first line and
    /// `detailedLines(cacheHit:)`'s `At peak` line.
    package var json: PeakMemoryJSON {
        PeakMemoryJSON(
            peakMemoryMB: system.peakMB, peakPhase: peakMemoryPhase, peakSeconds: system.peakSeconds,
            atPeak: PeakMemoryJSON.AtPeak(
                inProcessMB: system.footprintMB, neuralEngineMB: system.neuralNotCountedMB,
                modelFilesInUseMB: system.modelMappedFilesMB,
                neuralEngineServicesMB: system.neuralEngineServicesGrowthMB))
    }

    /// The phase of the peak memory: "prepare", "warmup", or `runPhase`.
    private var peakMemoryPhase: String {
        [("prepare", system.preparePeakMB), ("warmup", system.warmupPeakMB), (runPhase, system.runPeakMB)]
            .max { $0.1 < $1.1 }!.0
    }

    /// `runPhase` capitalized, for table rows: "Trials" or "Inference".
    private var runPhaseTitle: String { runPhase.prefix(1).uppercased() + runPhase.dropFirst() }

    /// Whether the system reached the warning memory pressure level during the run.
    private var isUnderPressure: Bool { (peakPressureLevel ?? 1) >= 2 }

    /// The highest pressure level, "warning" or "critical", for the memory pressure caveat.
    private var pressureName: String { (peakPressureLevel ?? 1) >= 4 ? "critical" : "warning" }
}

/// `MemoryReport.json`, in MiB. Encoded with `.convertToSnakeCase`, `peakMemoryMB` becomes `peak_memory_mb`.
package struct PeakMemoryJSON: Codable {
    package let peakMemoryMB: Double
    /// "prepare", "warmup", or the run phase, such as "trials".
    package let peakPhase: String
    package let peakSeconds: Double
    package let atPeak: AtPeak

    /// The highest poll's parts, at `peakSeconds`. They add up to `peakMemoryMB` unless an in-process spike
    /// between polls set the peak.
    package struct AtPeak: Codable {
        package let inProcessMB: Double
        package let neuralEngineMB: Double?
        package let modelFilesInUseMB: Double
        package let neuralEngineServicesMB: Double?

        /// Writes unmeasured parts as null rather than leaving their keys out, so every run has the same keys.
        package func encode(to encoder: any Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(inProcessMB, forKey: .inProcessMB)
            try container.encode(neuralEngineMB, forKey: .neuralEngineMB)
            try container.encode(modelFilesInUseMB, forKey: .modelFilesInUseMB)
            try container.encode(neuralEngineServicesMB, forKey: .neuralEngineServicesMB)
        }
    }
}

private func pad(_ text: String, _ width: Int) -> String {
    String(repeating: " ", count: max(0, width - text.count)) + text
}

private func fmtSeconds(_ v: Double) -> String {
    String(format: "%.1fs", v)
}

private func fmtMB(_ v: Double) -> String {
    String(format: "%.1f MB", v)
}

/// Whether `phase` is part of the run, after prepare and warmup.
private func isRunPhase(_ phase: String) -> Bool {
    phase != "prepare" && phase != "warmup"
}

private func megabytes(_ bytes: UInt64) -> Double {
    Double(bytes) / 1_048_576
}

// MARK: - Report

/// What one measured run needed, in MiB. The "peak" figures without `System` are this process's in-process
/// memory (its physical footprint), which iOS's per-app memory limit applies to.
package struct MemoryReport {
    /// What the output calls the phases after warmup, such as "trials" or "inference".
    package let runPhase: String
    /// False when the tool skipped warmup, as with llm-runner's `--warmup off`; the warmup figures are then 0.
    package let hasWarmup: Bool
    package let preparePeakMB: Double
    package let warmupPeakMB: Double
    /// The highest peak of the run's phases.
    package let runPeakMB: Double
    /// The kernel's high-water mark of the in-process memory over the whole run.
    package let lifetimePeakMB: Double
    /// The most of this process's memory that was compressed at once, polled; already in the in-process
    /// figures.
    package let compressedPeakMB: Double
    /// The highest system memory pressure level during the run: 1 normal, 2 warning, 4 critical. nil where
    /// the kernel doesn't report it.
    package let peakPressureLevel: Int?
    /// False when the Neural Engine services couldn't be read (they need root on macOS, through sudo).
    package let measuredNeuralEngineServices: Bool
    /// What the run needed from the system, which doesn't all count toward the limit.
    package let system: System

    /// The highest sum, polled every 5 ms, of this process's footprint, the Neural Engine services' growth,
    /// the uncounted Neural Engine memory, and the mapped model files the model needs (what stayed resident
    /// through `MemoryMeasurement.finish(generate:)`'s check): what the system has to provide for the run.
    /// Leaves out the model files the model doesn't need, such as a staged copy a Neural Engine model no
    /// longer reads, and the other mapped files, code and system data that would be resident anyway; the
    /// system can reclaim them. Only `footprintMB` counts toward the limit.
    package struct System {
        package let preparePeakMB: Double
        package let warmupPeakMB: Double
        /// The run's phases, as in `MemoryReport.runPeakMB`.
        package let runPeakMB: Double
        /// Never below this process's own peak, which may be a kernel high-water mark that no poll caught;
        /// then the parts below, from the highest poll, add up to less.
        package let peakMB: Double
        /// Seconds since measurement started, of the highest poll.
        package let peakSeconds: Double
        /// The parts of the highest poll.
        package let footprintMB: Double
        /// nil when the Neural Engine services weren't measured.
        package let neuralEngineServicesGrowthMB: Double?
        /// nil when the kernel doesn't report it.
        package let neuralNotCountedMB: Double?
        /// Mapped files in the model bundle, the temporary directory, and Core AI's cache (see
        /// `modelFileDirectories(modelPath:)`), up to what the model needs: the part that counts, such as
        /// weights the GPU runs from.
        package let modelMappedFilesMB: Double

        /// "in-process X + Neural Engine memory Y + model files in use Z + Neural Engine services W", leaving out
        /// parts under 1 MB.
        package func breakdown(_ format: (Double) -> String) -> String {
            let parts = [
                ("Neural Engine memory", neuralNotCountedMB ?? 0), ("model files in use", modelMappedFilesMB),
                ("Neural Engine services", neuralEngineServicesGrowthMB ?? 0),
            ]
            .filter { $0.1 >= 1 }
            .map { "\($0.0) \(format($0.1))" }
            return (["in-process \(format(footprintMB))"] + parts).joined(separator: " + ")
        }
    }
}
