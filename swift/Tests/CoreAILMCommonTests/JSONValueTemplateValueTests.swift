// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILMCommon

@Suite("A request's JSON as a chat template takes it")
struct JSONValueTemplateValueTests {
    @Test("A whole number is an integer, so a tool schema's `\"minimum\": 1` is not written as 1.0")
    func wholeNumbersAreIntegers() throws {
        let json = #"{"min":1,"neg":-3,"zero":0,"max":2.5,"big":1e300,"flag":false,"none":null,"list":[1,2.5]}"#
        let value = try JSONDecoder().decode(JSONValue.self, from: Data(json.utf8)).templateValue
        let object = try #require(value as? [String: any Sendable])
        #expect(object["min"] as? Int == 1)
        #expect(object["neg"] as? Int == -3)
        #expect(object["zero"] as? Int == 0)
        #expect(object["max"] as? Double == 2.5)
        // Too large for `Int`: stays the double it decoded as.
        #expect(object["big"] as? Double == 1e300)
        #expect(object["flag"] as? Bool == false)
        // `null` is an empty optional, which a template takes as `none`.
        let none = try #require(object["none"])
        #expect(Mirror(reflecting: none).displayStyle == .optional && Mirror(reflecting: none).children.isEmpty)
        let list = try #require(object["list"] as? [any Sendable])
        #expect(list[0] as? Int == 1)
        #expect(list[1] as? Double == 2.5)
    }
}
