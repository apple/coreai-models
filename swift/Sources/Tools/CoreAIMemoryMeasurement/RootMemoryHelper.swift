// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#if os(macOS)
import Darwin
import Foundation

/// The Neural Engine services run as root, and macOS only lets root read their memory. Running all of the tool
/// under sudo would leave root-owned files in .build and run Core AI against root's cache. Instead,
/// the tool starts its own executable under sudo as a helper that only reads process footprints and
/// answers over a pipe. All the bookkeeping stays in the unprivileged process.
///
/// sudo runs on a private pseudo-terminal rather than the user's, because with a terminal it offers Touch ID
/// (macOS turns Touch ID off for sudo's askpass mode). If the user cancels Touch ID, the helper is skipped.
/// On Macs where sudo doesn't use Touch ID, its password prompt shows up on the private terminal, and
/// the tool answers it with a password from a dialog. The user's terminal is never put into password
/// mode, which some terminal apps didn't recover from when the prompt was cancelled.
///
/// Protocol: one tab-separated request per line on the helper's stdin; replies on its stdout.
/// - `ping` → `ok` when the helper runs as root, `not-root` otherwise.
/// - `scan<TAB>name<TAB>name…` → readings of every process with one of those names.
/// - `read<TAB>pid<TAB>name<TAB>pid<TAB>name…` → readings.
/// - Anything else → an empty line.
/// Readings are one `pid<TAB>footprint<TAB>name` line each, then an empty line.
package enum RootMemoryHelper {
    /// The first argument that makes the tool run as the helper, followed by the client's pid.
    package static let argument = "--memory-helper"
    /// The first argument that makes the tool attach to a terminal and then run a command:
    /// `--sudo-terminal <terminal path> <command> <arguments>…`.
    package static let terminalArgument = "--sudo-terminal"

    /// Makes `terminalPath` this process's controlling terminal and stderr, then replaces this process with
    /// `command`. posix_spawn has made this process a session leader, which it must be to take a controlling
    /// terminal. sudo reads the password from the controlling terminal and writes its prompt and errors
    /// there. Exits with 127 if any step fails.
    package static func runOnTerminal(_ terminalPath: String, command: [String]) -> Never {
        let terminal = open(terminalPath, O_RDWR)
        // TIOCSCTTY, which doesn't import into Swift: _IO('t', 97).
        guard terminal >= 0, ioctl(terminal, UInt(0x2000_7461), 0) == 0 else { exit(127) }
        dup2(terminal, STDERR_FILENO)
        if terminal != STDERR_FILENO { Darwin.close(terminal) }
        let argv = command.map { strdup($0) } + [nil]
        execv(command[0], argv)
        exit(127)
    }

    /// Whether sudo offers Touch ID, meaning pam_tid.so is enabled in its PAM configuration.
    fileprivate static var sudoUsesTouchID: Bool {
        ["/etc/pam.d/sudo_local", "/etc/pam.d/sudo"].contains { path in
            guard let config = try? String(contentsOfFile: path, encoding: .utf8) else { return false }
            return config.split(separator: "\n").contains { line in
                let line = line.trimmingCharacters(in: .whitespaces)
                return !line.hasPrefix("#") && line.contains("pam_tid.so")
            }
        }
    }

    /// Asks for the password in a dialog; nil if the user skips or the dialog can't be shown.
    fileprivate static func askPassword(retry: Bool) -> String? {
        let message =
            (retry ? "Sorry, that password didn't work. " : "")
            + "\(toolName) needs your password to measure the memory of the Neural Engine services, which run as "
            + "root. A helper started with sudo only reads memory. Skip to run without them."
        let script = """
            display dialog "\(message)" default answer "" with hidden answer with title "\(toolName)" \
            buttons {"Skip", "OK"} default button "OK" cancel button "Skip" with icon caution
            text returned of result
            """
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script]
        let output = Pipe()
        process.standardOutput = output
        // Skip makes osascript report "User canceled" on stderr, which isn't worth showing.
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return nil
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        // osascript ends the result with a newline; the password itself may end in spaces.
        var password = String(decoding: data, as: UTF8.self)
        if password.hasSuffix("\n") { password.removeLast() }
        return password
    }

    /// Answers requests until stdin closes, which happens when the tool exits for any reason.
    package static func run(clientPid: pid_t) {
        let reader = LocalProcessMemoryReader(excluded: [getpid(), clientPid])
        let channel = LineChannel(input: STDIN_FILENO, output: STDOUT_FILENO)
        while let request = channel.readLine() {
            let fields = request.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            switch fields.first {
            case "ping":
                channel.write(geteuid() == 0 ? "ok\n" : "not-root\n")
            case "scan":
                channel.write(encode(reader.readAll(names: Set(fields.dropFirst()))))
            case "read":
                var pids: [pid_t: String] = [:]
                var rest = fields.dropFirst()
                while rest.count >= 2, let pid = pid_t(rest.removeFirst()) { pids[pid] = rest.removeFirst() }
                channel.write(encode(reader.read(pids)))
            default:
                channel.write("\n")
            }
        }
    }

    private static func encode(_ readings: [ProcessMemoryProbe.Reading]) -> String {
        readings.map {
            "\($0.pid)\t\($0.footprint)\t\($0.name)\n"
        }.joined() + "\n"
    }
}

/// Reads other processes through `RootMemoryHelper` running under sudo.
package final class HelperProcessMemoryReader: ProcessMemoryReader, @unchecked Sendable {
    /// sudo's password prompt, which we watch for on the private terminal. It has no `%`, since sudo would
    /// expand one.
    private static let passwordPrompt = "\(toolName) sudo password: "

    private let pid: pid_t
    /// Our ends of the helper's stdin and stdout.
    private let toHelper: Int32
    private let fromHelper: Int32
    /// The private terminal's controlling side, kept open for the helper's lifetime.
    private let terminal: Int32
    private let channel: LineChannel
    private let lock = NSLock()
    /// False once the helper has exited or stopped answering.
    private var isAlive = true
    private var isClosed = false

    /// Starts the helper under sudo, which asks for Touch ID; cancelling it skips the helper. On Macs where
    /// sudo doesn't use Touch ID, `askPassword` asks for the password instead. sudo remembers authentication
    /// per terminal, and the private terminal is new each run, so it asks every run. When stdin isn't a
    /// terminal, sudo runs with `-n`, so it only succeeds if it needs no password. Returns nil if sudo fails
    /// or the user skips.
    package static func launch() -> HelperProcessMemoryReader? {
        guard let executable = Bundle.main.executablePath else { return nil }
        let terminal = posix_openpt(O_RDWR | O_NOCTTY)
        guard terminal >= 0 else { return nil }
        guard grantpt(terminal) == 0, unlockpt(terminal) == 0, let terminalName = ptsname(terminal) else {
            Darwin.close(terminal)
            return nil
        }
        let terminalPath = String(cString: terminalName)
        var stdinPipe: [Int32] = [0, 0]
        var stdoutPipe: [Int32] = [0, 0]
        guard pipe(&stdinPipe) == 0 else {
            Darwin.close(terminal)
            return nil
        }
        guard pipe(&stdoutPipe) == 0 else {
            for descriptor in [terminal, stdinPipe[0], stdinPipe[1]] { Darwin.close(descriptor) }
            return nil
        }
        let interactive = isatty(STDIN_FILENO) != 0
        let arguments =
            [executable, RootMemoryHelper.terminalArgument, terminalPath, "/usr/bin/sudo"]
            + (interactive ? ["-p", passwordPrompt] : ["-n"])
            + [executable, RootMemoryHelper.argument, String(getpid())]

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_adddup2(&actions, stdinPipe[0], STDIN_FILENO)
        posix_spawn_file_actions_adddup2(&actions, stdoutPipe[1], STDOUT_FILENO)
        var attributes: posix_spawnattr_t?
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        // A new session, so the private terminal can become its controlling terminal, and with every
        // other descriptor closed.
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT))
        var pid: pid_t = 0
        let argv = arguments.map { strdup($0) } + [nil]
        defer { for argument in argv { free(argument) } }
        let rc = posix_spawn(&pid, executable, &actions, &attributes, argv, environ)
        Darwin.close(stdinPipe[0])
        Darwin.close(stdoutPipe[1])
        guard rc == 0 else {
            for descriptor in [terminal, stdinPipe[1], stdoutPipe[0]] { Darwin.close(descriptor) }
            return nil
        }
        // Writing to a helper that has exited must fail rather than raise SIGPIPE.
        _ = fcntl(stdinPipe[1], F_SETNOSIGPIPE, 1)
        let reader = HelperProcessMemoryReader(
            pid: pid, toHelper: stdinPipe[1], fromHelper: stdoutPipe[0], terminal: terminal)
        guard reader.authenticate() else {
            reader.close()
            return nil
        }
        return reader
    }

    private init(pid: pid_t, toHelper: Int32, fromHelper: Int32, terminal: Int32) {
        self.pid = pid
        self.toHelper = toHelper
        self.fromHelper = fromHelper
        self.terminal = terminal
        channel = LineChannel(input: fromHelper, output: toHelper)
    }

    deinit {
        close()
    }

    /// Pings the helper, which answers once sudo has started it. If a password prompt shows up first, then
    /// either Touch ID was cancelled, which skips the helper, or sudo doesn't use Touch ID, and the prompts
    /// are answered from a dialog. False if sudo gives up or the user skips.
    private func authenticate() -> Bool {
        guard channel.write("ping\n") else { return false }
        let usesTouchID = RootMemoryHelper.sudoUsesTouchID
        var terminalOutput = ""
        var fds = [
            pollfd(fd: fromHelper, events: Int16(POLLIN), revents: 0),
            pollfd(fd: terminal, events: Int16(POLLIN), revents: 0),
        ]
        var chunk = [UInt8](repeating: 0, count: 4096)
        while true {
            if poll(&fds, nfds_t(fds.count), -1) < 0 {
                if errno == EINTR { continue }
                return false
            }
            if fds[0].revents != 0 {
                // sudo has started the helper, or exited.
                return channel.readLine() == "ok"
            }
            guard fds[1].revents != 0 else { continue }
            let count = Darwin.read(terminal, &chunk, chunk.count)
            guard count > 0 else {
                // The terminal is closed; keep waiting for the helper's reply or end of input.
                fds[1].fd = -1
                continue
            }
            terminalOutput += String(decoding: chunk[..<count], as: UTF8.self)
            guard terminalOutput.contains(Self.passwordPrompt) else { continue }
            guard !usesTouchID else { return false }
            // sudo says "Sorry, try again." before asking again.
            let retry = terminalOutput.contains("Sorry")
            terminalOutput = ""
            guard let password = RootMemoryHelper.askPassword(retry: retry) else { return false }
            let line = Array((password + "\n").utf8)
            guard line.withUnsafeBytes({ Darwin.write(terminal, $0.baseAddress, $0.count) }) == line.count else {
                return false
            }
        }
    }

    package func readAll(names: Set<String>) -> [ProcessMemoryProbe.Reading] {
        let request = (["scan"] + names).joined(separator: "\t") + "\n"
        return readings(request)
    }

    package func read(_ pids: [pid_t: String]) -> [ProcessMemoryProbe.Reading] {
        guard !pids.isEmpty else { return [] }
        let request = (["read"] + pids.flatMap { [String($0.key), $0.value] }).joined(separator: "\t") + "\n"
        return readings(request)
    }

    /// Stops the helper by closing its stdin, and waits for it to exit.
    package func close() {
        let alreadyClosed = lock.withLock {
            defer { (isAlive, isClosed) = (false, true) }
            return isClosed
        }
        guard !alreadyClosed else { return }
        Darwin.close(toHelper)
        // Hanging up the terminal ends sudo if it's still asking for a password, and spares it waiting at
        // exit for terminal output nobody reads.
        Darwin.close(terminal)
        var status: Int32 = 0
        while waitpid(pid, &status, 0) < 0, errno == EINTR {}
        Darwin.close(fromHelper)
    }

    private func readings(_ message: String) -> [ProcessMemoryProbe.Reading] {
        (request(message) ?? []).compactMap { line in
            let fields = line.split(separator: "\t", maxSplits: 2, omittingEmptySubsequences: false)
            guard fields.count == 3, let pid = pid_t(fields[0]), let footprint = UInt64(fields[1]) else { return nil }
            return ProcessMemoryProbe.Reading(pid: pid, name: String(fields[2]), footprint: footprint)
        }
    }

    /// Sends `message` and returns the reply's lines, up to the empty one. nil once the helper has exited.
    private func request(_ message: String) -> [String]? {
        lock.withLock {
            guard isAlive, channel.write(message) else {
                isAlive = false
                return nil
            }
            var lines: [String] = []
            while let line = channel.readLine() {
                if line.isEmpty { return lines }
                lines.append(line)
            }
            isAlive = false
            return nil
        }
    }
}

/// Newline-delimited text over a pair of file descriptors.
private final class LineChannel {
    private let input: Int32
    private let output: Int32
    private var buffer: [UInt8] = []

    package init(input: Int32, output: Int32) {
        self.input = input
        self.output = output
    }

    /// The next line without its newline; nil at end of input.
    package func readLine() -> String? {
        var chunk = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            if let newline = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = String(decoding: buffer[..<newline], as: UTF8.self)
                buffer.removeSubrange(...newline)
                return line
            }
            let count = Darwin.read(input, &chunk, chunk.count)
            if count < 0, errno == EINTR { continue }
            guard count > 0 else { return nil }
            buffer.append(contentsOf: chunk[..<count])
        }
    }

    /// Writes all of `text`; false if the other end has closed.
    @discardableResult
    package func write(_ text: String) -> Bool {
        var bytes = Array(text.utf8)[...]
        while !bytes.isEmpty {
            let count = bytes.withUnsafeBytes { Darwin.write(output, $0.baseAddress, $0.count) }
            if count < 0, errno == EINTR { continue }
            guard count > 0 else { return false }
            bytes = bytes.dropFirst(count)
        }
        return true
    }
}
#endif
