// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CryptoKit
import Foundation
import Synchronization

/// The prefix cache's second tier: ``PrefixTree`` nodes as files, kept across restarts.
///
/// Every chunk the tree takes is written here as well, off the engine's path, and a prompt that runs
/// past what the tree holds in memory reads the next chunks back from here. So the disk budget can be
/// larger than the memory one, and a restarted server reuses what the last run computed.
///
/// A node's files are named by its key, the chained SHA-256 of its prefix, so no token ids are written:
/// an arriving prompt finds its nodes by hashing its own ids. The KV in the files is still derived from
/// the prompts, so the directory is as private as they are.
///
/// ```
/// geometry      the layout the files were written for; a directory written for another is emptied
/// index.jsonl   one record per line, appended to, and compacted when the store opens and as it grows
/// <key>.kv      a chunk's KV, one blob per KV state
/// <key>.state   the recurrent state at the chunk's end, one blob per state
/// lock          held while a store has the directory open
/// ```
///
/// Like the tree, the store keeps every chain whole: it evicts the least recently used leaf first. One
/// process at a time may use a directory: a second finds it locked and runs without a disk tier.
package final class PrefixCacheStore: Sendable {
    package typealias Key = PrefixTree.Key

    /// Where the files live: one directory per model build, since the KV only means anything for the
    /// weights that computed it.
    package let directory: URL

    /// Bytes the files may take, chunks and snapshots together.
    package let byteBudget: Int

    /// One node, as the index records it.
    package struct Record: Codable, Equatable, Sendable {
        package var key: Key
        /// The node above, so a chain is rebuilt from the index without searching.
        package var parent: Key?
        package var depth: Int
        package var kvBytes: Int
        /// Zero when the node has no snapshot.
        package var snapshotBytes: Int
        /// Set, with the rest left empty, when the node is gone: the log is only appended to.
        package var removed: Bool?

        package var bytes: Int { kvBytes + snapshotBytes }
    }

    private struct Entry {
        var record: Record
        /// Nodes whose parent this is. Only a childless node is evictable.
        var children = 0
        var lastUsed = 0
    }

    private struct Index {
        var entries: [Key: Entry] = [:]
        var bytesHeld = 0
        var clock = 0
        var writtenNodes = 0
        var readNodes = 0
        var evictedNodes = 0
        /// Lines in the index file, live or not, so it is compacted before it outgrows the live ones.
        var indexLines = 0
    }

    private let index = Mutex(Index())

    /// Every write, eviction and index append, in order, so the engine never waits on the disk.
    private let queue = DispatchQueue(label: "com.apple.coreai.prefix-cache-store", qos: .utility)

    private static let geometryName = "geometry"
    private static let indexName = "index.jsonl"
    private static let kvExtension = "kv"
    private static let stateExtension = "state"
    private static let lockName = "lock"

    /// The lock file, open for as long as the store: closing it lets another process have the directory.
    private let lock: Int32

    /// Opens, or creates, the store in `directory`.
    ///
    /// - Parameters:
    ///   - geometry: Describes how the files' bytes are laid out. A directory stamped with another is
    ///     emptied first, since its keys would still match but its bytes would not fit.
    /// - Returns: Nil when the directory can't be created, another process has it open, or the budget
    ///   is not positive.
    package init?(directory: URL, byteBudget: Int, geometry: String) {
        guard byteBudget > 0 else { return nil }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            CLILogger.log("prefix cache: no disk tier, cannot create \(directory.path): \(error)")
            return nil
        }
        // Another process's store would delete the files this one writes, and this one its files.
        let lock = open(directory.appendingPathComponent(Self.lockName).path, O_CREAT | O_RDWR, 0o644)
        guard lock >= 0, flock(lock, LOCK_EX | LOCK_NB) == 0 else {
            if lock >= 0 { close(lock) }
            CLILogger.log(
                "prefix cache: no disk tier, another process is using \(directory.path)", level: 0)
            return nil
        }
        self.lock = lock
        self.directory = directory
        self.byteBudget = byteBudget
        Self.matchOrDiscard(geometry: geometry, in: directory)
        load()
    }

    deinit {
        close(lock)
    }

    // MARK: - Opening

    /// Empties `directory` unless its stamp is `geometry`, then stamps it.
    private static func matchOrDiscard(geometry: String, in directory: URL) {
        let stamp = directory.appendingPathComponent(geometryName)
        let found = try? String(contentsOf: stamp, encoding: .utf8)
        guard found != geometry else { return }
        if found != nil {
            CLILogger.log("prefix cache: emptying \(directory.lastPathComponent), written for another layout")
        }
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        for file in files where file.lastPathComponent != lockName { try? FileManager.default.removeItem(at: file) }
        try? geometry.write(to: stamp, atomically: true, encoding: .utf8)
    }

    /// Replays the index, keeps the records whose chain and files are whole, evicts the least recently
    /// used leaves while they are over the budget, and rewrites it with what is left, least recently
    /// used first. Then deletes every file it does not name.
    private func load() {
        var replayed: [Key: Record] = [:]
        // A node's last line is its last use, so sorting by it puts the least recently used first.
        var lastLine: [Key: Int] = [:]
        let text = (try? String(contentsOf: indexURL, encoding: .utf8)) ?? ""
        for (number, line) in text.split(separator: "\n").enumerated() {
            guard let record = try? JSONDecoder().decode(Record.self, from: Data(line.utf8)) else {
                // A line cut off by a crash.
                continue
            }
            replayed[record.key] = record.removed == true ? nil : record
            lastLine[record.key] = number
        }
        let order = replayed.keys.sorted { lastLine[$0, default: 0] < lastLine[$1, default: 0] }

        let kept: [Record] = index.withLock { index in
            // By depth, so a parent is always considered before its children.
            for record in order.compactMap({ replayed[$0] }).sorted(by: { $0.depth < $1.depth }) {
                let parentKept = record.parent.map { index.entries[$0] != nil } ?? (record.depth == 1)
                guard parentKept, record.depth > 0, fileSize(kvURL(record.key)) == record.kvBytes,
                    record.snapshotBytes == 0 || fileSize(stateURL(record.key)) == record.snapshotBytes
                else { continue }
                index.entries[record.key] = Entry(record: record)
                index.bytesHeld += record.bytes
                if let parent = record.parent { index.entries[parent]?.children += 1 }
            }
            for key in order where index.entries[key] != nil {
                index.clock += 1
                index.entries[key]?.lastUsed = index.clock
            }
            // A budget smaller than the one the files were written under: the coldest go first.
            _ = makeRoom(for: 0, keepingChainOf: nil, in: &index)
            return liveRecords(in: index)
        }

        writeIndex(kept)

        let names = Set(kept.map(\.key))
        let snapshotted = Set(kept.filter { $0.snapshotBytes > 0 }.map(\.key))
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        for file in files {
            switch file.pathExtension {
            case Self.kvExtension:
                if !names.contains(file.deletingPathExtension().lastPathComponent) {
                    try? FileManager.default.removeItem(at: file)
                }
            case Self.stateExtension:
                let key = file.deletingPathExtension().lastPathComponent
                if !snapshotted.contains(key) {
                    try? FileManager.default.removeItem(at: file)
                }
            default:
                // The stamp, the index and the lock; anything else is a write a crash left behind.
                if ![Self.geometryName, Self.indexName, Self.lockName].contains(file.lastPathComponent) {
                    try? FileManager.default.removeItem(at: file)
                }
            }
        }
        if !kept.isEmpty { CLILogger.log("prefix cache: disk tier holds \(kept.count) node(s); \(summary)") }
    }

    // MARK: - Asking

    /// `key`'s record, if the store holds it.
    package func record(of key: Key) -> Record? {
        index.withLock { $0.entries[key]?.record }
    }

    /// Nodes held.
    package var nodeCount: Int { index.withLock { $0.entries.count } }

    /// Bytes held.
    package var bytesHeld: Int { index.withLock { $0.bytesHeld } }

    // MARK: - Writing

    /// Writes one chunk as a child of `parent`, in the background.
    ///
    /// Dropped when `parent` is not held, since nothing could reach it, when the node is already held,
    /// or when the budget can't make room beside the chain above it.
    package func write(chunk: [Data], for key: Key, parent: Key?, depth: Int) {
        // Everything that changes the index runs on the queue, one at a time, so nothing moves between
        // making room and taking it.
        queue.async { [self] in
            let size = Self.encodedSize(of: chunk)
            let evicted = index.withLock { index -> [Key]? in
                guard index.entries[key] == nil, depth > 0 else { return nil }
                if let parent {
                    guard index.entries[parent]?.record.depth == depth - 1 else { return nil }
                } else if depth != 1 {
                    return nil
                }
                return makeRoom(for: size, keepingChainOf: parent, in: &index)
            }
            guard let evicted else { return }
            for victim in evicted { deleteFiles(of: victim) }
            guard (try? Self.encode(chunk).write(to: kvURL(key), options: .atomic)) != nil else {
                CLILogger.log("prefix cache: could not write a chunk to \(directory.path)")
                return
            }
            let written = Record(key: key, parent: parent, depth: depth, kvBytes: size, snapshotBytes: 0)
            index.withLock { index in
                index.clock += 1
                index.entries[key] = Entry(record: written, lastUsed: index.clock)
                index.bytesHeld += written.bytes
                if let parent { index.entries[parent]?.children += 1 }
                index.writtenNodes += 1
            }
            append(written)
        }
    }

    /// Writes the recurrent state at the end of `key`'s chunk, in the background, unless the node is
    /// gone or already has one, or the budget can't make room beside its chain.
    package func write(snapshot: [Data], for key: Key) {
        queue.async { [self] in
            let size = Self.encodedSize(of: snapshot)
            let evicted = index.withLock { index -> [Key]? in
                guard let entry = index.entries[key], entry.record.snapshotBytes == 0 else { return nil }
                return makeRoom(for: size, keepingChainOf: key, in: &index)
            }
            guard let evicted else { return }
            for victim in evicted { deleteFiles(of: victim) }
            guard (try? Self.encode(snapshot).write(to: stateURL(key), options: .atomic)) != nil else {
                CLILogger.log("prefix cache: could not write a snapshot to \(directory.path)")
                return
            }
            let updated = index.withLock { index -> Record? in
                index.entries[key]?.record.snapshotBytes = size
                index.bytesHeld += size
                return index.entries[key]?.record
            }
            if let updated { append(updated) }
        }
    }

    /// Marks `keys` used, so they are the last to go, here and after a restart.
    package func touch(_ keys: [Key]) {
        guard !keys.isEmpty else { return }
        queue.async { [self] in
            let records = index.withLock { index -> [Record] in
                index.clock += 1
                return keys.compactMap { key in
                    index.entries[key]?.lastUsed = index.clock
                    return index.entries[key]?.record
                }
            }
            for record in records { append(record) }
        }
    }

    /// Waits for every write queued so far.
    package func flush() {
        queue.sync {}
    }

    // MARK: - Reading

    /// `key`'s chunk, and its snapshot when `withSnapshot` and it has one, read from disk.
    ///
    /// Read rather than mapped: the tree keeps what this returns, and a mapping would hold an evicted
    /// file's space on disk past the budget, and count the system's file cache as the tree's memory.
    ///
    /// A file that is gone or does not parse takes the node, and everything below it, out of the store.
    package func read(_ key: Key, withSnapshot: Bool) -> (chunk: [Data], snapshot: [Data]?)? {
        guard let record = record(of: key) else { return nil }
        guard let chunk = Self.decode(contentsOf: kvURL(key)) else {
            remove(key)
            return nil
        }
        var snapshot: [Data]?
        if withSnapshot, record.snapshotBytes > 0 {
            guard let state = Self.decode(contentsOf: stateURL(key)) else {
                remove(key)
                return nil
            }
            snapshot = state
        }
        index.withLock { $0.readNodes += 1 }
        return (chunk, snapshot)
    }

    /// Takes `key` and everything below it out of the store, in the background.
    package func remove(_ key: Key) {
        queue.async { [self] in
            let removed = index.withLock { index -> [Key] in
                var doomed: [Key] = []
                var frontier = [key]
                while let next = frontier.popLast() {
                    guard index.entries[next] != nil else { continue }
                    doomed.append(next)
                    frontier += index.entries.filter { $0.value.record.parent == next }.map(\.key)
                }
                for key in doomed.reversed() { drop(key, in: &index) }
                return doomed
            }
            for key in removed { deleteFiles(of: key) }
        }
    }

    // MARK: - Eviction

    /// Evicts least-recently-used leaves from the index until `bytes` more fit, never `key` or a node
    /// above it. Returns them, for the caller to delete once the lock is let go, or nil, having evicted
    /// nothing, when `bytes` can't fit.
    private func makeRoom(for bytes: Int, keepingChainOf key: Key?, in index: inout Index) -> [Key]? {
        guard index.bytesHeld + bytes > byteBudget else { return [] }
        // Every other node is evictable, leaves first, so this is exactly what can be freed.
        let protected = ancestors(of: key, in: index)
        let protectedBytes = protected.reduce(0) { $0 + (index.entries[$1]?.record.bytes ?? 0) }
        guard protectedBytes + bytes <= byteBudget else { return nil }
        var evicted: [Key] = []
        while index.bytesHeld + bytes > byteBudget {
            let candidate = index.entries.lazy
                .filter { $0.value.children == 0 && !protected.contains($0.key) }
                .min { $0.value.lastUsed < $1.value.lastUsed }
            guard let victim = candidate?.key else { break }
            drop(victim, in: &index)
            index.evictedNodes += 1
            evicted.append(victim)
        }
        return evicted
    }

    /// Takes a leaf out of the index, and records that it is gone.
    private func drop(_ key: Key, in index: inout Index) {
        guard let entry = index.entries.removeValue(forKey: key) else { return }
        index.bytesHeld -= entry.record.bytes
        if let parent = entry.record.parent { index.entries[parent]?.children -= 1 }
    }

    private func deleteFiles(of key: Key) {
        try? FileManager.default.removeItem(at: kvURL(key))
        try? FileManager.default.removeItem(at: stateURL(key))
        append(Record(key: key, parent: nil, depth: 0, kvBytes: 0, snapshotBytes: 0, removed: true))
    }

    /// `key` and every node above it.
    private func ancestors(of key: Key?, in index: Index) -> Set<Key> {
        var out = Set<Key>()
        var next = key
        while let current = next, let entry = index.entries[current] {
            out.insert(current)
            next = entry.record.parent
        }
        return out
    }

    // MARK: - Files

    private var indexURL: URL { directory.appendingPathComponent(Self.indexName) }
    private func kvURL(_ key: Key) -> URL { directory.appendingPathComponent("\(key).\(Self.kvExtension)") }
    private func stateURL(_ key: Key) -> URL { directory.appendingPathComponent("\(key).\(Self.stateExtension)") }

    private func fileSize(_ url: URL) -> Int {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.intValue ?? -1
    }

    /// Appends one record to the index. A crash costs at most the line being written. Only ever called
    /// on the queue.
    ///
    /// Every use and removal adds a line, so once the file holds four times the live records, and at
    /// least 256 lines, it is rewritten with just those.
    private func append(_ record: Record) {
        guard var line = try? JSONEncoder().encode(record) else { return }
        line.append(0x0A)
        let live = index.withLock { index -> [Record]? in
            index.indexLines += 1
            guard index.indexLines >= Swift.max(256, 4 * index.entries.count) else { return nil }
            return liveRecords(in: index)
        }
        if let live {
            writeIndex(live)
            return
        }
        if let handle = try? FileHandle(forWritingTo: indexURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: line)
        } else {
            try? line.write(to: indexURL)
        }
    }

    /// The records held, least recently used first, as the index lists them.
    private func liveRecords(in index: Index) -> [Record] {
        index.entries.values.sorted { $0.lastUsed < $1.lastUsed }.map(\.record)
    }

    /// Replaces the index with `records`, one line each.
    private func writeIndex(_ records: [Record]) {
        var out = Data()
        for record in records {
            guard let line = try? JSONEncoder().encode(record) else { continue }
            out.append(line)
            out.append(0x0A)
        }
        try? out.write(to: indexURL, options: .atomic)
        index.withLock { $0.indexLines = records.count }
    }

    /// A file's bytes: the blob count, each blob's length, then the blobs, all little-endian.
    static func encode(_ blobs: [Data]) -> Data {
        var out = Data()
        out.reserveCapacity(encodedSize(of: blobs))
        var count = UInt32(blobs.count).littleEndian
        withUnsafeBytes(of: &count) { out.append(contentsOf: $0) }
        for blob in blobs {
            var length = UInt64(blob.count).littleEndian
            withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        }
        for blob in blobs { out.append(blob) }
        return out
    }

    static func encodedSize(of blobs: [Data]) -> Int {
        4 + 8 * blobs.count + blobs.reduce(0) { $0 + $1.count }
    }

    /// The blobs in `url`, or nil when it is gone or does not parse.
    static func decode(contentsOf url: URL) -> [Data]? {
        guard let file = try? Data(contentsOf: url) else { return nil }
        return decode(file)
    }

    static func decode(_ file: Data) -> [Data]? {
        guard file.count >= 4 else { return nil }
        let count = Int(file.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian })
        let header = 4 + 8 * count
        guard count > 0, count < 1024, file.count >= header else { return nil }
        // Each length is checked against what is left, so a damaged header can't overflow the sum.
        var remaining = file.count - header
        var lengths: [Int] = []
        for blob in 0..<count {
            let raw = file.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4 + 8 * blob, as: UInt64.self) }
            guard let length = Int(exactly: UInt64(littleEndian: raw)), length <= remaining else { return nil }
            remaining -= length
            lengths.append(length)
        }
        guard remaining == 0 else { return nil }
        // Slices of the one read, so the file's bytes are not copied again.
        var cursor = file.startIndex + header
        return lengths.map { length in
            defer { cursor += length }
            return file[cursor..<(cursor + length)]
        }
    }

    // MARK: - Reporting

    /// One line for a log.
    package var summary: String {
        index.withLock { index in
            let held = ByteCountFormatter.string(fromByteCount: Int64(index.bytesHeld), countStyle: .memory)
            let budget = ByteCountFormatter.string(fromByteCount: Int64(byteBudget), countStyle: .memory)
            return "disk \(index.entries.count) node(s), \(held) of \(budget), \(index.writtenNodes) written, "
                + "\(index.readNodes) read, \(index.evictedNodes) evicted"
        }
    }

    // MARK: - Where

    /// A directory for the model at `modelURL`, under the user's caches directory, which the system
    /// may clear.
    ///
    /// Named by the model's folder and a digest of its path and of every file's size and modification
    /// date, so a re-export, even to the same place, starts a fresh directory.
    package static func defaultDirectory(forModelAt modelURL: URL) -> URL? {
        guard let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        else { return nil }
        let root = modelURL.standardizedFileURL.resolvingSymlinksInPath()
        var hasher = SHA256()
        hasher.update(data: Data(root.path.utf8))
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey, .contentModificationDateKey]
        var files: [(String, Int, Double)] = []
        if let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys) {
            for case let file as URL in walker {
                guard let values = try? file.resourceValues(forKeys: Set(keys)), values.isRegularFile == true
                else { continue }
                files.append(
                    (
                        String(file.path.dropFirst(root.path.count)), values.fileSize ?? 0,
                        values.contentModificationDate?.timeIntervalSince1970 ?? 0
                    ))
            }
        }
        for (path, size, modified) in files.sorted(by: { $0.0 < $1.0 }) {
            hasher.update(data: Data("\(path)\t\(size)\t\(modified)\n".utf8))
        }
        let digest = hasher.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
        let name = root.lastPathComponent.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" ? $0 : "_" }
        return
            caches
            .appendingPathComponent("coreai-prefix-cache", isDirectory: true)
            .appendingPathComponent("\(String(name))-\(digest)", isDirectory: true)
    }
}
