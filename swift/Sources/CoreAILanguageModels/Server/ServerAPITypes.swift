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

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stream, stop
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case responseFormat = "response_format"
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
}

// MARK: - Chat Message

public struct ChatMessage: Decodable, Sendable {
    public let role: String
    public let content: MessageContent

    enum CodingKeys: String, CodingKey {
        case role, content
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)

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
        public let content: String
        public init(role: String, content: String) {
            self.role = role
            self.content = content
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
        public init(role: String? = nil, content: String? = nil) {
            self.role = role
            self.content = content
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
