// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - Completions Request (Legacy /v1/completions)

public struct CompletionRequest: Decodable, Sendable {
    public let model: String?
    public let prompt: [String]
    public let maxTokens: Int?
    public let temperature: Double?
    public let echo: Bool?
    public let logprobs: Int?

    enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, echo, logprobs
        case maxTokens = "max_tokens"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        echo = try container.decodeIfPresent(Bool.self, forKey: .echo)
        logprobs = try container.decodeIfPresent(Int.self, forKey: .logprobs)

        // prompt can be: string, [string], [int] (token IDs), [[int]] (batched token IDs)
        if let s = try? container.decode(String.self, forKey: .prompt) {
            prompt = [s]
        } else if let arr = try? container.decode([String].self, forKey: .prompt) {
            prompt = arr
        } else if let tokenIds = try? container.decode([Int].self, forKey: .prompt) {
            prompt = ["__TOKEN_IDS__:\(tokenIds.map(String.init).joined(separator: ","))"]
        } else if let batchedIds = try? container.decode([[Int]].self, forKey: .prompt) {
            prompt = batchedIds.map { "__TOKEN_IDS__:\($0.map(String.init).joined(separator: ","))" }
        } else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: [CodingKeys.prompt],
                    debugDescription: "Expected string, [string], [int], or [[int]] for 'prompt'"
                ))
        }
    }
}

// MARK: - Completions Response

public struct CompletionResponse: Encodable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [CompletionChoice]

    public init(id: String, object: String, created: Int, model: String, choices: [CompletionChoice]) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
    }

    public struct CompletionChoice: Encodable, Sendable {
        public let index: Int
        public let text: String
        public let logprobs: LogprobsResult?
        public let finishReason: String?

        public init(index: Int, text: String, logprobs: LogprobsResult?, finishReason: String?) {
            self.index = index
            self.text = text
            self.logprobs = logprobs
            self.finishReason = finishReason
        }

        enum CodingKeys: String, CodingKey {
            case index, text, logprobs
            case finishReason = "finish_reason"
        }
    }

    public struct LogprobsResult: Encodable, Sendable {
        public let tokens: [String]
        public let tokenLogprobs: [Double?]
        public let topLogprobs: [[String: Double]?]
        public let textOffset: [Int]

        public init(tokens: [String], tokenLogprobs: [Double?], topLogprobs: [[String: Double]?], textOffset: [Int]) {
            self.tokens = tokens
            self.tokenLogprobs = tokenLogprobs
            self.topLogprobs = topLogprobs
            self.textOffset = textOffset
        }

        enum CodingKeys: String, CodingKey {
            case tokens
            case tokenLogprobs = "token_logprobs"
            case topLogprobs = "top_logprobs"
            case textOffset = "text_offset"
        }
    }
}
