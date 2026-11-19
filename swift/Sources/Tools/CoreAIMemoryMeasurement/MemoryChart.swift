// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Draws the memory timeline as an area chart in the terminal, with the peaks labeled and the phases marked
/// along the time axis. Consecutive `trial N` phases share one "trials" label, with a tick where each trial
/// starts. Each sample's `uncountedMB` is stacked on `footprintMB` in a lighter shade.
package enum MemoryChart {
    /// What the chart draws.
    package struct Timeline {
        package let title: String
        package let samples: [Sample]
        /// The peak memory to label. It can be higher than any sample, since it includes this process's
        /// kernel high-water mark, which no poll may have caught.
        package let labeledPeak: (megabytes: Double, seconds: Double)
        /// The in-process memory peak, marked on the in-process line.
        package var labeledFootprintPeak: (megabytes: Double, seconds: Double)? = nil
    }

    package struct Sample {
        package let elapsedSeconds: Double
        package let phase: String
        package let footprintMB: Double
        /// Memory the run needs outside in-process memory: the uncounted Neural Engine memory, the mapped
        /// model files in use, and the Neural Engine services' growth since the run started.
        package var uncountedMB: Double = 0

        /// The peak memory.
        package var totalMB: Double { footprintMB + uncountedMB }
    }

    private static let footprintLegend = "in-process"
    private static let outsideLegend = "Neural Engine memory + model files in use + Neural Engine services"

    private static func peakLabel(_ peak: (megabytes: Double, seconds: Double)) -> String {
        String(format: "peak memory %.1f MB @ %.1fs", peak.megabytes, peak.seconds)
    }

    private static func inProcessPeakLabel(_ peak: (megabytes: Double, seconds: Double)) -> String {
        String(format: "in-process peak %.1f MB @ %.1fs", peak.megabytes, peak.seconds)
    }

    private static let eighths: [Character] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    private static let uncountedCell: Character = "░"

    /// The chart as text; nil when there's nothing to draw.
    package static func render(_ timeline: Timeline, width: Int = 64, height: Int = 12) -> String? {
        let samples = timeline.samples
        guard let last = samples.last, let peak = samples.max(by: { $0.totalMB < $1.totalMB }),
            peak.totalMB > 0
        else { return nil }
        let duration = max(last.elapsedSeconds, 1e-9)
        let width = max(8, min(width, samples.count))
        func column(_ seconds: Double) -> Int { min(width - 1, Int(seconds / duration * Double(width))) }

        // Each column shows the largest sample in its time bucket so short spikes stay visible; columns
        // without samples repeat the previous value.
        var values = [Sample?](repeating: nil, count: width)
        for sample in samples {
            let c = column(sample.elapsedSeconds)
            if sample.totalMB > values[c]?.totalMB ?? -1 { values[c] = sample }
        }
        var filled: [(total: Double, process: Double)] = []
        for value in values {
            filled.append(value.map { ($0.totalMB, $0.footprintMB) } ?? filled.last ?? (0, 0))
        }

        let (phaseStarts, groups) = phases(of: samples)
        let ticks = phaseStarts.map(column)

        let labeledPeak = timeline.labeledPeak
        let top = max(peak.totalMB, labeledPeak.megabytes)
        let labelWidth = String(format: "%.0f", top).count
        let gutter = String(repeating: " ", count: labelWidth + 1)
        var lines = [timeline.title]
        let legend = ["█ " + footprintLegend, "\(uncountedCell) " + outsideLegend]
        lines.append(gutter + " " + legend.joined(separator: "  "))

        // Peak markers, flipped to the left of the arrow when they would run past the right edge.
        func marker(_ arrow: String, _ text: String, at seconds: Double) -> String {
            let peakColumn = column(seconds)
            return peakColumn + 2 + text.count <= width
                ? String(repeating: " ", count: peakColumn) + arrow + " " + text
                : String(repeating: " ", count: max(0, peakColumn - text.count - 1)) + text + " " + arrow
        }
        lines.append(gutter + " " + marker("▼", peakLabel(labeledPeak), at: labeledPeak.seconds))
        if let footprintPeak = timeline.labeledFootprintPeak {
            lines.append(gutter + " " + marker("▽", inProcessPeakLabel(footprintPeak), at: footprintPeak.seconds))
        }

        func level(_ megabytes: Double) -> Int { Int((megabytes / top * Double(height * 8)).rounded()) }
        let levels = filled.map { (total: level($0.total), process: level($0.process)) }
        for row in stride(from: height - 1, through: 0, by: -1) {
            let isLabeled = row == height - 1 || row % 3 == 2
            let label = isLabeled ? String(format: "%.0f", top * Double(row + 1) / Double(height)) : ""
            // Where there's uncounted memory, both parts round to whole cells so its shading always sits
            // above the footprint's solid cells. Elsewhere the top cell keeps its eighth.
            let cells = levels.map { bar -> Character in
                let processRows = (bar.process + 4) / 8
                let stackRows = (bar.total + 4) / 8
                return stackRows > processRows
                    ? (row < processRows ? eighths[8] : row < stackRows ? uncountedCell : " ")
                    : eighths[min(8, max(0, bar.total - row * 8))]
            }
            lines.append(pad(label, labelWidth) + " ┤" + String(cells))
        }

        var axis = [Character](repeating: "─", count: width)
        for tick in ticks { axis[tick] = "┬" }
        lines.append(pad("0", labelWidth) + " ┼" + String(axis) + "▶")

        // Group names under the axis, then the time each group starts and the end time.
        var names = [Character](repeating: " ", count: width + 8)
        var times = [Character](repeating: " ", count: width + 8)
        for (i, group) in groups.enumerated() {
            let start = column(group.start)
            let end = i + 1 < groups.count ? column(groups[i + 1].start) : width
            // Phases narrower than a column (such as a cache-hit prepare) get no name.
            guard end > start else { continue }
            let span = end - start
            // Center the name in its span. A name wider than a short span spills right if there's room;
            // otherwise it's cut to fit the span.
            if !place(group.label, at: start + max(0, (span - group.label.count) / 2), in: &names) {
                place(String(group.label.prefix(max(1, span - 1))), at: start, in: &names)
            }
            place(seconds(group.start), at: start, in: &times)
        }
        place(seconds(last.elapsedSeconds), at: width, in: &times)
        lines.append(gutter + " " + String(names).trimmingTrailingSpaces())
        lines.append(gutter + " " + String(times).trimmingTrailingSpaces())
        return lines.joined(separator: "\n")
    }

    /// When each phase starts, plus runs of consecutive phases that share a label (every `trial N` phase is
    /// labeled "trials") with when each run starts.
    private static func phases(of samples: [Sample]) -> (starts: [Double], groups: [(label: String, start: Double)]) {
        var starts: [Double] = []
        var groups: [(label: String, start: Double)] = []
        var previousPhase: String?
        for sample in samples where sample.phase != previousPhase {
            previousPhase = sample.phase
            starts.append(sample.elapsedSeconds)
            let label = sample.phase.hasPrefix("trial ") ? "trials" : sample.phase
            if groups.last?.label != label { groups.append((label, sample.elapsedSeconds)) }
        }
        return (starts, groups)
    }

    private static func pad(_ text: String, _ width: Int) -> String {
        String(repeating: " ", count: max(0, width - text.count)) + text
    }

    /// A time-axis label: one decimal under 10 s, whole seconds above.
    private static func seconds(_ value: Double) -> String {
        String(format: value < 10 ? "%.1fs" : "%.0fs", value)
    }

    /// Writes `text` into `line` at `offset`, but only onto blank space with a blank on each side, so labels
    /// never touch. Returns whether it was written.
    @discardableResult
    private static func place(_ text: String, at offset: Int, in line: inout [Character]) -> Bool {
        let range = offset..<(offset + text.count)
        guard range.upperBound <= line.count,
            line[max(0, offset - 1)..<min(line.count, range.upperBound + 1)].allSatisfy({ $0 == " " })
        else { return false }
        line.replaceSubrange(range, with: Array(text))
        return true
    }
}

extension String {
    fileprivate func trimmingTrailingSpaces() -> String {
        String(self.reversed().drop(while: { $0 == " " }).reversed())
    }
}
