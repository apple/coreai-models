// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAILanguageModels

extension KVCacheStrategy: ExpressibleByArgument {
    public init?(argument: String) {
        if let value = KVCacheStrategy(rawValue: argument) {
            self = value
        } else {
            switch argument.lowercased() {
            case "auto":
                self = .auto
            case "fixed-size", "fixed", "fixed_size":
                self = .fixedSize
            case "growing":
                self = .growing
            case "chunked":
                self = .chunked
            default:
                return nil
            }
        }
    }

    public static var allValueStrings: [String] {
        ["auto", "fixed-size", "growing", "chunked"]
    }

    public static var defaultCompletionKind: CompletionKind {
        .list(allValueStrings)
    }
}
