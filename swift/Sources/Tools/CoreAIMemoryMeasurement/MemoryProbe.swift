// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Darwin
import Dispatch
import Foundation

/// Memory of this process, in bytes.
package struct ProcessMemory {
    /// Physical footprint: what iOS enforces the per-app memory limit against (its in-process memory).
    package var footprint: UInt64 = 0
    /// The kernel's high-water mark of `footprint` over the process's lifetime.
    package var lifetimePeak: UInt64 = 0
    /// File-backed pages currently in RAM: memory-mapped model files (such as mapped weights) and
    /// framework code. They don't count toward the memory limit. Some must stay resident, such as weights
    /// the GPU runs from; see `PeakFootprintTracker.TotalReading.system(neededModelFiles:)`.
    package var fileBacked: UInt64 = 0
    /// The part of `footprint` that is compressed. Diagnostic only, since `footprint` already includes it.
    package var compressed: UInt64 = 0
    /// Neural Engine memory charged to this process but excluded from `footprint`, so it doesn't count
    /// toward the memory limit; nil when the kernel doesn't report it.
    package var neuralNotCounted: UInt64?
}

/// How many `natural_t`s of `task_vm_info` the kernel must fill in for `ledger_tag_neural_nofootprint`
/// to be valid; older kernels return a shorter struct.
private let neuralFieldCount = mach_msg_type_number_t(
    (MemoryLayout<task_vm_info_data_t>.offset(of: \.ledger_tag_neural_nofootprint)!
        + MemoryLayout<Int64>.size) / MemoryLayout<natural_t>.size)

/// The system's memory pressure level: 1 normal, 2 warning, 4 critical. nil where the kernel doesn't report it.
package func memoryPressureLevel() -> Int? {
    var level: Int32 = 0
    var size = MemoryLayout<Int32>.size
    guard sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, nil, 0) == 0 else { return nil }
    return Int(level)
}

/// This process's memory; all zeros if `task_info` fails.
package func processMemory() -> ProcessMemory {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return ProcessMemory() }
    return ProcessMemory(
        footprint: UInt64(info.phys_footprint), lifetimePeak: UInt64(max(0, info.ledger_phys_footprint_peak)),
        fileBacked: UInt64(info.external), compressed: UInt64(info.compressed),
        neuralNotCounted: count >= neuralFieldCount ? UInt64(max(0, info.ledger_tag_neural_nofootprint)) : nil)
}

/// Tracks peak physical footprint per named phase by polling on a background queue.
///
/// Since spikes can happen between polls, each phase also records the kernel's lifetime high-water mark
/// at its start and end. If it rose during the phase, the process peaked during that phase and the kernel
/// value is exact.
///
/// With a Neural Engine service probe, each poll also reads the Neural Engine services and adds up how much
/// they've grown since the run started.
package final class PeakFootprintTracker: @unchecked Sendable {
    package struct Phase {
        package let name: String
        package let polledPeak: UInt64
        package let lifetimePeakAtStart: UInt64
        package let lifetimePeakAtEnd: UInt64
        /// From each `slowInterval` window, the poll with the highest `system(neededModelFiles: 0)` and the one
        /// with the highest `system(neededModelFiles: nil)`, kept because which model files count is only known
        /// at the end of the run. That finds the highest `system(neededModelFiles:)` for any cap, unless the
        /// model files cross the cap within a window, and keeps the tracker's own memory small.
        package let polls: [TotalReading]
        /// The reading at the end of the phase, with the Neural Engine services counted as not having grown;
        /// stands in for the polls in a phase too short to be polled.
        package let readingAtEnd: TotalReading

        package var isExact: Bool { lifetimePeakAtEnd > lifetimePeakAtStart }
        /// The kernel's high-water mark if it rose during the phase, otherwise the highest poll.
        package var peak: UInt64 { isExact ? max(lifetimePeakAtEnd, polledPeak) : polledPeak }

        /// The poll where `TotalReading.system(neededModelFiles:)` was highest.
        package func systemReading(neededModelFiles: UInt64?) -> TotalReading {
            polls.max { $0.system(neededModelFiles: neededModelFiles) < $1.system(neededModelFiles: neededModelFiles) }
                ?? readingAtEnd
        }

        package func systemPeak(neededModelFiles: UInt64?) -> UInt64 {
            max(systemReading(neededModelFiles: neededModelFiles).system(neededModelFiles: neededModelFiles), peak)
        }
    }

    /// One poll: this process's memory and the Neural Engine services' growth, all read at the same moment.
    package struct TotalReading {
        package let elapsedSeconds: Double
        package let footprint: UInt64
        /// nil without a Neural Engine service probe.
        package let servicesGrowth: UInt64?
        package let neuralNotCounted: UInt64?
        /// The part of this process's mapped files that holds the model.
        package let modelFileBacked: UInt64

        /// This process's footprint, the Neural Engine services' growth, uncounted Neural Engine memory, and the
        /// model files it needs: memory the system can't reclaim during the run. The model files count up to
        /// `neededModelFiles`, what stayed resident through `MemoryMeasurement`'s end-of-run check, such as
        /// weights the GPU runs from; nil counts them all. The rest, such as a staged copy that a Neural Engine
        /// model no longer reads, can be dropped under pressure.
        package func system(neededModelFiles: UInt64?) -> UInt64 {
            footprint + (servicesGrowth ?? 0) + (neuralNotCounted ?? 0)
                + countedModelFiles(neededModelFiles: neededModelFiles)
        }

        /// The part of `modelFileBacked` that counts toward peak memory: up to `neededModelFiles`, or all of it
        /// when nil.
        package func countedModelFiles(neededModelFiles: UInt64?) -> UInt64 {
            min(modelFileBacked, neededModelFiles ?? modelFileBacked)
        }
    }

    /// One window of `timeline(neededModelFiles:)`: its poll with the highest `system(neededModelFiles:)`.
    /// It's a single poll, so its parts were read at the same moment.
    package struct TimelineWindow {
        package let phase: String
        package let highestSystem: TotalReading
    }

    private let lock = NSLock()
    private let queue = DispatchQueue(label: "peak-footprint-poll", qos: .userInitiated)
    private let pollInterval: DispatchTimeInterval = .milliseconds(5)
    /// How often the timeline's windows close and the memory pressure level is read.
    private let slowInterval: Duration = .milliseconds(100)
    private let modelDirectories: [String]
    #if os(macOS)
    private let neuralEngineProbe: ProcessMemoryProbe?
    #endif
    private let origin = ContinuousClock.now
    private var timer: DispatchSourceTimer?
    private var lastSlowTick: ContinuousClock.Instant?

    private var phases: [Phase] = []
    private var currentPhase: String?
    private var currentPeak: UInt64 = 0
    private var currentLifetimePeakAtStart: UInt64 = 0
    private var currentPolls: [TotalReading] = []
    /// The current `slowInterval` window's candidates for `Phase.polls`.
    private var windowHighestBase: TotalReading?
    private var windowHighestWithFiles: TotalReading?
    private var highestCompressed: UInt64 = 0
    private var fileBackedAtStart: UInt64?
    /// Mapped files outside `modelDirectories` at the latest region walk.
    private var sharedFileBacked: UInt64?
    private var highestPressureLevel: Int?
    /// Region walks run here, off the poll queue; see `walkRegionsIfChanged`. Only the poll queue touches
    /// `walkedFootprint`, `walkedFileBacked`, `lastWalkFileBacked`, and `isWalking`.
    private let walkQueue = DispatchQueue(label: "peak-footprint-region-walk", qos: .userInitiated)
    private let regionWalkStep: UInt64 = 16 * 1_048_576
    private var walkedFootprint: UInt64 = 0
    private var walkedFileBacked: UInt64 = 0
    /// Mapped files at the latest walk, to walk again when they fall.
    private var lastWalkFileBacked: UInt64 = 0
    private var isWalking = false

    #if os(macOS)
    package init(modelDirectories: [String] = [], neuralEngineProbe: ProcessMemoryProbe? = nil) {
        self.modelDirectories = modelDirectories
        self.neuralEngineProbe = neuralEngineProbe
    }
    #else
    package init(modelDirectories: [String] = []) {
        self.modelDirectories = modelDirectories
    }
    #endif

    /// Ends the current phase (if any) and starts measuring `name`.
    package func begin(_ name: String) {
        let files = mappedFiles()
        let reading = processMemory()
        lock.withLock {
            recordShared(files)
            closeCurrentPhase(reading)
            if fileBackedAtStart == nil { fileBackedAtStart = reading.fileBacked }
            currentPhase = name
            currentPeak = reading.footprint
            currentLifetimePeakAtStart = reading.lifetimePeak
        }
        if timer == nil {
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now(), repeating: pollInterval, leeway: .milliseconds(1))
            t.setEventHandler { [weak self] in self?.tick() }
            t.resume()
            timer = t
        }
    }

    /// Ends the current phase, stops polling, and returns all phases in order.
    package func finish() -> [Phase] {
        timer?.cancel()
        timer = nil
        // Cancelling doesn't stop a poll already running, so wait for it; then wait for a walk in progress so
        // its shared mapped files are recorded.
        queue.sync {}
        walkQueue.sync {}
        let files = mappedFiles()
        let reading = processMemory()
        return lock.withLock {
            recordShared(files)
            closeCurrentPhase(reading)
            return phases
        }
    }

    /// Each phase's polls, reduced to the highest `system(neededModelFiles:)` in each `slowInterval` window,
    /// so spikes between windows still show up; what the chart and the CSV show. Complete after `finish()`.
    package func timeline(neededModelFiles: UInt64?) -> [TimelineWindow] {
        let windowSeconds = slowInterval.inSeconds
        return lock.withLock {
            var windows: [TimelineWindow] = []
            for phase in phases {
                var highest: TotalReading?
                var windowStart = 0.0
                for poll in phase.polls {
                    if let current = highest, poll.elapsedSeconds - windowStart >= windowSeconds {
                        windows.append(TimelineWindow(phase: phase.name, highestSystem: current))
                        highest = nil
                    }
                    if highest == nil { windowStart = poll.elapsedSeconds }
                    let isHigher = highest.map {
                        poll.system(neededModelFiles: neededModelFiles) > $0.system(neededModelFiles: neededModelFiles)
                    }
                    if isHigher ?? true { highest = poll }
                }
                if let highest { windows.append(TimelineWindow(phase: phase.name, highestSystem: highest)) }
            }
            return windows
        }
    }

    /// The highest system memory pressure level seen, read every `slowInterval`; see `memoryPressureLevel()`.
    package var peakPressureLevel: Int? { lock.withLock { highestPressureLevel } }

    /// The highest compressed memory of this process, polled.
    package var compressedPeak: UInt64 { lock.withLock { highestCompressed } }

    private func closeCurrentPhase(_ reading: ProcessMemory) {
        highestCompressed = max(highestCompressed, reading.compressed)
        guard let name = currentPhase else { return }
        let elapsed = (ContinuousClock.now - origin).inSeconds
        let readingAtEnd = TotalReading(
            elapsedSeconds: elapsed, footprint: reading.footprint, servicesGrowth: hasServiceProbe ? 0 : nil,
            neuralNotCounted: reading.neuralNotCounted, modelFileBacked: modelFileBacked(reading.fileBacked))
        keepWindowPolls()
        phases.append(
            Phase(
                name: name, polledPeak: max(currentPeak, reading.footprint),
                lifetimePeakAtStart: currentLifetimePeakAtStart, lifetimePeakAtEnd: reading.lifetimePeak,
                polls: currentPolls, readingAtEnd: readingAtEnd))
        currentPhase = nil
        currentPolls = []
    }

    private var hasServiceProbe: Bool {
        #if os(macOS)
        neuralEngineProbe != nil
        #else
        false
        #endif
    }

    private func recordShared(_ files: MappedFiles?) {
        if let files { sharedFileBacked = files.shared(modelDirectories: modelDirectories) }
    }

    /// The part of `fileBacked` that holds the model. A poll can't tell which file a page belongs to, so
    /// this is the mapped files minus the shared ones (frameworks, system data) from the latest region walk.
    /// Before the first walk, or where regions can't be walked, it's minus the mapped files at the start.
    private func modelFileBacked(_ fileBacked: UInt64) -> UInt64 {
        let shared = sharedFileBacked ?? fileBackedAtStart ?? fileBacked
        return fileBacked > shared ? fileBacked - shared : 0
    }

    /// Moves the current window's candidates into `currentPolls`, in time order.
    private func keepWindowPolls() {
        let candidates = [windowHighestBase, windowHighestWithFiles].compactMap { $0 }
        for poll in candidates.sorted(by: { $0.elapsedSeconds < $1.elapsedSeconds })
        where poll.elapsedSeconds != currentPolls.last?.elapsedSeconds {
            currentPolls.append(poll)
        }
        windowHighestBase = nil
        windowHighestWithFiles = nil
    }

    private func tick() {
        let memory = processMemory()
        let now = ContinuousClock.now
        let elapsed = (now - origin).inSeconds
        let isSlowTick = lastSlowTick.map { now - $0 >= slowInterval } ?? true
        let pressureLevel = isSlowTick ? memoryPressureLevel() : nil
        #if os(macOS)
        let servicesGrowth = neuralEngineProbe.map { probe in probe.poll().reduce(0) { $0 + probe.growth(of: $1) } }
        #else
        let servicesGrowth: UInt64? = nil
        #endif
        lock.withLock {
            if let pressureLevel { highestPressureLevel = max(highestPressureLevel ?? 0, pressureLevel) }
            currentPeak = max(currentPeak, memory.footprint)
            let total = TotalReading(
                elapsedSeconds: elapsed, footprint: memory.footprint, servicesGrowth: servicesGrowth,
                neuralNotCounted: memory.neuralNotCounted, modelFileBacked: modelFileBacked(memory.fileBacked))
            if currentPhase != nil {
                highestCompressed = max(highestCompressed, memory.compressed)
                let base = total.system(neededModelFiles: 0)
                if base > windowHighestBase?.system(neededModelFiles: 0) ?? 0 { windowHighestBase = total }
                let withFiles = total.system(neededModelFiles: nil)
                if withFiles > windowHighestWithFiles?.system(neededModelFiles: nil) ?? 0 {
                    windowHighestWithFiles = total
                }
            }
            guard isSlowTick else { return }
            lastSlowTick = now
            keepWindowPolls()
        }
        walkRegionsIfChanged(memory)
    }

    /// Walks the regions to update the shared mapped files when the footprint or the mapped files have risen
    /// by `regionWalkStep` since the last walk they triggered, or when the mapped files have fallen by as much
    /// since any walk. A walk is slow enough to delay polls, so it runs on `walkQueue`. While one is running,
    /// changes wait for the next poll.
    private func walkRegionsIfChanged(_ memory: ProcessMemory) {
        guard !isWalking else { return }
        let footprintRose = memory.footprint >= walkedFootprint + regionWalkStep
        let fileBackedRose = memory.fileBacked >= walkedFileBacked + regionWalkStep
        let fileBackedFell = memory.fileBacked + regionWalkStep <= lastWalkFileBacked
        guard footprintRose || fileBackedRose || fileBackedFell else { return }
        if footprintRose { walkedFootprint = memory.footprint }
        if fileBackedRose { walkedFileBacked = memory.fileBacked }
        lastWalkFileBacked = memory.fileBacked
        isWalking = true
        walkQueue.async { [self] in
            if let files = mappedFiles() {
                lock.withLock { recordShared(files) }
            }
            queue.async { [self] in isWalking = false }
        }
    }
}

#if os(macOS)
/// Reads the memory of other processes. Without root, only processes running as the same user can be
/// read. The Neural Engine services run as root, so reading them takes a `LocalProcessMemoryReader` in a root
/// process, or a `HelperProcessMemoryReader`.
package protocol ProcessMemoryReader: Sendable {
    /// Every process whose executable name is in `names`.
    func readAll(names: Set<String>) -> [ProcessMemoryProbe.Reading]
    /// The given processes, keyed by pid; ones that have exited or aren't readable are left out.
    func read(_ pids: [pid_t: String]) -> [ProcessMemoryProbe.Reading]
}

/// Reads other processes from this process, with this process's privileges.
package struct LocalProcessMemoryReader: ProcessMemoryReader {
    /// Processes never reported: this process itself, and in the root helper the helper too.
    package var excluded: Set<pid_t> = [getpid()]

    package func readAll(names: Set<String>) -> [ProcessMemoryProbe.Reading] {
        // Room for processes that start between the two calls.
        let capacity = Int(proc_listallpids(nil, 0)) + 64
        var pids = [pid_t](repeating: 0, count: capacity)
        let count = Int(proc_listallpids(&pids, Int32(capacity * MemoryLayout<pid_t>.size)))
        var readings: [ProcessMemoryProbe.Reading] = []
        var pathBuffer = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN))
        for pid in pids.prefix(max(0, count)) where pid > 0 && !excluded.contains(pid) {
            // Match on the executable's file name; proc_name can be truncated.
            guard proc_pidpath(pid, &pathBuffer, UInt32(pathBuffer.count)) > 0 else { continue }
            let name = (nullTerminatedString(pathBuffer) as NSString).lastPathComponent
            guard names.contains(name) else { continue }
            if let reading = read(pid: pid, name: name) { readings.append(reading) }
        }
        return readings
    }

    package func read(_ pids: [pid_t: String]) -> [ProcessMemoryProbe.Reading] {
        pids.compactMap { read(pid: $0.key, name: $0.value) }
    }

    /// Reads one process's memory; nil if it has exited or can't be read.
    private func read(pid: pid_t, name: String) -> ProcessMemoryProbe.Reading? {
        var usage = rusage_info_v6()
        let rc = withUnsafeMutablePointer(to: &usage) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(pid, RUSAGE_INFO_V6, $0)
            }
        }
        guard rc == 0 else { return nil }
        return ProcessMemoryProbe.Reading(pid: pid, name: name, footprint: usage.ri_phys_footprint)
    }
}

/// Reads the memory of other processes, such as the Neural Engine services, through a `ProcessMemoryReader`,
/// and tells how much each has grown since the probe was made.
///
/// System services are long-lived and shared by every client, so what counts toward a run is how far their
/// footprint rises above where it was when the run started.
package final class ProcessMemoryProbe: @unchecked Sendable {
    package struct Reading {
        package let pid: pid_t
        package let name: String
        package let footprint: UInt64
    }

    /// Out-of-process Neural Engine services: the compilers, and the daemon that loads programs onto the
    /// Neural Engine.
    package static let neuralEngineServiceNames: Set<String> = [
        "ANECompilerService", "ANELargeModelCompilerService", "aned",
    ]

    private let names: Set<String>
    private let reader: any ProcessMemoryReader
    /// Footprint of each process running when the probe was made.
    private let baselines: [pid_t: UInt64]

    // Finding processes by name means scanning every pid, which is much slower than reading the pids
    // already found. So a full scan runs at most once per `rescanInterval`, and polls in between read
    // only the known pids. The interval is short so a compiler that starts and exits quickly is still seen.
    private let scanLock = NSLock()
    private let rescanInterval: Duration = .milliseconds(250)
    private var lastScan: ContinuousClock.Instant = .now
    private var watchedNames: [pid_t: String] = [:]

    /// Watches processes whose executable name is in `names`. Processes that `reader` can't read are skipped.
    package init(names: Set<String>, reader: any ProcessMemoryReader) {
        self.names = names
        self.reader = reader
        let running = reader.readAll(names: names)
        baselines = Dictionary(running.map { ($0.pid, $0.footprint) }, uniquingKeysWith: { $1 })
        lastScan = .now
        watchedNames = Dictionary(running.map { ($0.pid, $0.name) }, uniquingKeysWith: { $1 })
    }

    /// How far `reading` is above the process's footprint when the run started; a process that started
    /// during the run counts in full.
    package func growth(of reading: Reading) -> UInt64 {
        let baseline = baselines[reading.pid] ?? 0
        return reading.footprint > baseline ? reading.footprint - baseline : 0
    }

    /// Samples the watched processes. Processes that started since the last scan are picked up within
    /// `rescanInterval`.
    package func poll() -> [Reading] {
        scanLock.withLock {
            let now = ContinuousClock.now
            if now - lastScan < rescanInterval {
                return reader.read(watchedNames)
            }
            let readings = reader.readAll(names: names)
            lastScan = now
            watchedNames = Dictionary(readings.map { ($0.pid, $0.name) }, uniquingKeysWith: { $1 })
            return readings
        }
    }
}
#endif
