// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - Chat Completion Request

public struct ChatCompletionRequest: Decodable, Sendable {
    public let model: String?
    public let messages: [ChatMessage]
    public let temperature: Double?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let topP: Double?
    public let topK: Int?
    public let stream: Bool?
    public let stop: [String]?
    public let responseFormat: ResponseFormat?
    public let tools: [ToolDefinition]?
    public let toolChoice: ToolChoice?
    public let parallelToolCalls: Bool?

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stream, stop, tools
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case responseFormat = "response_format"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        messages = try container.decode([ChatMessage].self, forKey: .messages)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        maxCompletionTokens = try container.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        responseFormat = try container.decodeIfPresent(ResponseFormat.self, forKey: .responseFormat)
        tools = try container.decodeIfPresent([ToolDefinition].self, forKey: .tools)
        toolChoice = try container.decodeIfPresent(ToolChoice.self, forKey: .toolChoice)
        parallelToolCalls = try container.decodeIfPresent(Bool.self, forKey: .parallelToolCalls)

        if let arr = try? container.decode([String].self, forKey: .stop) {
            stop = arr
        } else if let s = try? container.decode(String.self, forKey: .stop) {
            stop = [s]
        } else {
            stop = nil
        }
    }
}

// MARK: - Response Format (Guided Generation)

public struct ResponseFormat: Decodable, Sendable {
    public let type: String
    public let jsonSchema: JSONSchemaSpec?

    enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }

    public struct JSONSchemaSpec: Decodable, Sendable {
        public let name: String?
        public let schema: JSONValue
    }

    public var extractedSchema: String? {
        switch type {
        case "json_schema":
            guard let spec = jsonSchema else { return nil }
            if let data = try? JSONEncoder().encode(spec.schema),
                let str = String(data: data, encoding: .utf8)
            {
                return str
            }
            return nil
        case "json_object":
            return "{}"
        default:
            return nil
        }
    }
}

/// Generic JSON value for preserving arbitrary schema objects
public enum JSONValue: Codable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let n = try? container.decode(Double.self) {
            self = .number(n)
        } else if let obj = try? container.decode([String: JSONValue].self) {
            self = .object(obj)
        } else if let arr = try? container.decode([JSONValue].self) {
            self = .array(arr)
        } else if container.decodeNil() {
            self = .null
        } else {
            self = .null
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .number(let n): try container.encode(n)
        case .bool(let b): try container.encode(b)
        case .object(let obj): try container.encode(obj)
        case .array(let arr): try container.encode(arr)
        case .null: try container.encodeNil()
        }
    }

    public func asJSONObject() -> Any {
        switch self {
        case .string(let s): return s
        case .number(let n): return n
        case .bool(let b): return b
        case .null: return NSNull()
        case .object(let obj): return obj.mapValues { $0.asJSONObject() }
        case .array(let arr): return arr.map { $0.asJSONObject() }
        }
    }
}

// MARK: - Chat Message

public struct ChatMessage: Decodable, Sendable {
    public let role: String
    public let content: MessageContent
    public let toolCalls: [ToolCall]?
    public let toolCallId: String?
    public let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        toolCalls = try container.decodeIfPresent([ToolCall].self, forKey: .toolCalls)
        toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
        name = try container.decodeIfPresent(String.self, forKey: .name)

        if let text = try? container.decode(String.self, forKey: .content) {
            content = .text(text)
        } else if let parts = try? container.decode([ContentPart].self, forKey: .content) {
            content = .parts(parts)
        } else {
            content = .text("")
        }
    }
}

public enum MessageContent: Sendable {
    case text(String)
    case parts([ContentPart])

    public var textContent: String {
        switch self {
        case .text(let s): return s
        case .parts(let parts):
            return parts.compactMap {
                if case .text(let t) = $0 { return t }
                return nil
            }.joined(separator: " ")
        }
    }

    public var imageDataURLs: [String] {
        switch self {
        case .text: return []
        case .parts(let parts):
            return parts.compactMap {
                if case .imageURL(let url) = $0 { return url }
                return nil
            }
        }
    }
}

public enum ContentPart: Decodable, Sendable {
    case text(String)
    case imageURL(String)

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }

    enum ImageURLKeys: String, CodingKey {
        case url
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)
        case "image_url":
            let imageContainer = try container.nestedContainer(keyedBy: ImageURLKeys.self, forKey: .imageURL)
            let url = try imageContainer.decode(String.self, forKey: .url)
            self = .imageURL(url)
        default:
            self = .text("")
        }
    }
}

// MARK: - Chat Completion Response

public struct ChatCompletionResponse: Encodable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: Usage?

    public init(
        id: String, object: String = "chat.completion",
        created: Int = Int(Date().timeIntervalSince1970),
        model: String, choices: [Choice], usage: Usage? = nil
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }

    public struct Choice: Encodable, Sendable {
        public let index: Int
        public let message: ResponseMessage
        public let finishReason: String?
        public init(index: Int, message: ResponseMessage, finishReason: String?) {
            self.index = index
            self.message = message
            self.finishReason = finishReason
        }
        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    public struct ResponseMessage: Encodable, Sendable {
        public let role: String
        public let content: String?
        public let toolCalls: [ToolCall]?
        public init(role: String, content: String?, toolCalls: [ToolCall]? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
        }
        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(role, forKey: .role)
            try container.encode(content, forKey: .content)
            try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
        }
    }

    public struct Usage: Encodable, Sendable {
        public let promptTokens: Int
        public let completionTokens: Int
        public let totalTokens: Int
        public init(promptTokens: Int, completionTokens: Int, totalTokens: Int) {
            self.promptTokens = promptTokens
            self.completionTokens = completionTokens
            self.totalTokens = totalTokens
        }
        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

// MARK: - Streaming Chunk

public struct ChatCompletionChunk: Encodable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [ChunkChoice]

    public init(
        id: String, object: String = "chat.completion.chunk",
        created: Int = Int(Date().timeIntervalSince1970),
        model: String, choices: [ChunkChoice]
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
    }

    public struct ChunkChoice: Encodable, Sendable {
        public let index: Int
        public let delta: Delta
        public let finishReason: String?
        public init(index: Int, delta: Delta, finishReason: String?) {
            self.index = index
            self.delta = delta
            self.finishReason = finishReason
        }
        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    public struct Delta: Encodable, Sendable {
        public let role: String?
        public let content: String?
        public let toolCalls: [ToolCallDelta]?
        public init(role: String? = nil, content: String? = nil, toolCalls: [ToolCallDelta]? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
        }
        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encodeIfPresent(role, forKey: .role)
            try container.encode(content, forKey: .content)
            try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
        }
    }
}

// MARK: - Models List

public struct ModelsResponse: Encodable, Sendable {
    public let object: String
    public let data: [ModelInfo]

    public init(data: [ModelInfo]) {
        self.object = "list"
        self.data = data
    }

    public struct ModelInfo: Encodable, Sendable {
        public let id: String
        public let object: String
        public let created: Int
        public let ownedBy: String

        public init(id: String, created: Int, ownedBy: String) {
            self.id = id
            self.object = "model"
            self.created = created
            self.ownedBy = ownedBy
        }

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }
}

// MARK: - Health

public struct HealthResponse: Encodable, Sendable {
    public let status: String
    public init(status: String) { self.status = status }
}

// MARK: - Error Response

public struct ErrorResponse: Encodable, Sendable {
    public let error: ErrorDetail

    public init(error: ErrorDetail) { self.error = error }

    public struct ErrorDetail: Encodable, Sendable {
        public let message: String
        public let type: String
        public let code: String?

        public init(message: String, type: String, code: String? = nil) {
            self.message = message
            self.type = type
            self.code = code
        }
    }
}

// MARK: - Tool Calling Types

public struct ToolDefinition: Codable, Sendable {
    public let type: String
    public let function: FunctionDefinition

    public init(type: String = "function", function: FunctionDefinition) {
        self.type = type
        self.function = function
    }
}

public struct FunctionDefinition: Codable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue?
    public let strict: Bool?

    public init(name: String, description: String? = nil, parameters: JSONValue? = nil, strict: Bool? = nil) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.strict = strict
    }
}

public enum ToolChoice: Sendable, Equatable {
    case auto
    case none
    case required
    case function(name: String)
}

extension ToolChoice: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            switch str {
            case "auto": self = .auto
            case "none": self = .none
            case "required": self = .required
            default: self = .auto
            }
        } else {
            let obj = try container.decode(ToolChoiceObject.self)
            self = .function(name: obj.function.name)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .auto: try container.encode("auto")
        case .none: try container.encode("none")
        case .required: try container.encode("required")
        case .function(let name):
            try container.encode(ToolChoiceObject(function: ToolChoiceFunctionName(name: name)))
        }
    }

    private struct ToolChoiceObject: Codable {
        let type: String
        let function: ToolChoiceFunctionName
        init(function: ToolChoiceFunctionName) {
            self.type = "function"
            self.function = function
        }
    }

    private struct ToolChoiceFunctionName: Codable {
        let name: String
    }
}

public struct ToolCall: Codable, Sendable {
    public let id: String
    public let type: String
    public let function: ToolCallFunction

    public init(id: String, type: String = "function", function: ToolCallFunction) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct ToolCallFunction: Codable, Sendable {
    public let name: String
    public let arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

public struct ToolCallDelta: Encodable, Sendable {
    public let index: Int
    public let id: String?
    public let type: String?
    public let function: ToolCallFunctionDelta?

    public init(index: Int, id: String? = nil, type: String? = nil, function: ToolCallFunctionDelta? = nil) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct ToolCallFunctionDelta: Encodable, Sendable {
    public let name: String?
    public let arguments: String?

    public init(name: String? = nil, arguments: String? = nil) {
        self.name = name
        self.arguments = arguments
    }
}

// MARK: - Stats Response

public struct ServerStatsResponse: Codable, Sendable {
    public let totalRequests: Int
    public let totalPromptTokens: Int
    public let totalGenTokens: Int
    public let avgPrefillTokPerSec: Double
    public let avgDecodeTokPerSec: Double
    public let prefixHitRate: Double
    public let prefixHits: Int
    public let prefixMisses: Int
    public let totalToolCalls: Int
    public let topTools: [ToolCallStat]?

    public init(
        totalRequests: Int, totalPromptTokens: Int, totalGenTokens: Int,
        avgPrefillTokPerSec: Double, avgDecodeTokPerSec: Double,
        prefixHitRate: Double, prefixHits: Int, prefixMisses: Int,
        totalToolCalls: Int = 0, topTools: [ToolCallStat]? = nil
    ) {
        self.totalRequests = totalRequests
        self.totalPromptTokens = totalPromptTokens
        self.totalGenTokens = totalGenTokens
        self.avgPrefillTokPerSec = avgPrefillTokPerSec
        self.avgDecodeTokPerSec = avgDecodeTokPerSec
        self.prefixHitRate = prefixHitRate
        self.prefixHits = prefixHits
        self.prefixMisses = prefixMisses
        self.totalToolCalls = totalToolCalls
        self.topTools = topTools
    }

    enum CodingKeys: String, CodingKey {
        case totalRequests = "total_requests"
        case totalPromptTokens = "total_prompt_tokens"
        case totalGenTokens = "total_gen_tokens"
        case avgPrefillTokPerSec = "avg_prefill_tok_per_sec"
        case avgDecodeTokPerSec = "avg_decode_tok_per_sec"
        case prefixHitRate = "prefix_hit_rate"
        case prefixHits = "prefix_hits"
        case prefixMisses = "prefix_misses"
        case totalToolCalls = "total_tool_calls"
        case topTools = "top_tools"
    }
}

public struct ToolCallStat: Codable, Sendable {
    public let name: String
    public let count: Int

    public init(name: String, count: Int) {
        self.name = name
        self.count = count
    }
}

// MARK: - Ready Response

public struct ReadyResponse: Codable, Sendable {
    public let status: String
    public let model: String
    public let maxContextLength: Int
    public let busy: Bool
    public let cache: CacheStatus
    public let toolCalling: Bool

    public init(
        status: String, model: String, maxContextLength: Int,
        busy: Bool, cache: CacheStatus, toolCalling: Bool
    ) {
        self.status = status
        self.model = model
        self.maxContextLength = maxContextLength
        self.busy = busy
        self.cache = cache
        self.toolCalling = toolCalling
    }

    public struct CacheStatus: Codable, Sendable {
        public let usedTokens: Int
        public let maxTokens: Int
        public let utilization: Double

        public init(usedTokens: Int, maxTokens: Int, utilization: Double) {
            self.usedTokens = usedTokens
            self.maxTokens = maxTokens
            self.utilization = utilization
        }

        enum CodingKeys: String, CodingKey {
            case usedTokens = "used_tokens"
            case maxTokens = "max_tokens"
            case utilization
        }
    }

    enum CodingKeys: String, CodingKey {
        case status, model
        case maxContextLength = "max_context_length"
        case busy, cache
        case toolCalling = "tool_calling"
    }
}
