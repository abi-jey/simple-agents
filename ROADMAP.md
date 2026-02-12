# Roadmap

This document outlines the planned features and improvements for nagents.

## Current Status

nagents currently supports:
- OpenAI-compatible APIs (OpenAI, Azure OpenAI, local models)
- Streaming and non-streaming responses
- Tool/function calling
- Session management with SQLite
- HTTP/SSE traffic logging for debugging
- Tool hallucination handling

## Planned Features

### Provider Support

#### Google Gemini API
- [ ] Native Gemini API implementation
- [ ] Gemini-specific features (grounding, code execution)
- [ ] Examples for Gemini usage

#### Anthropic Claude API
- [ ] Native Anthropic API implementation
- [ ] Claude-specific features (extended thinking, computer use)
- [ ] Examples for Claude usage

#### OpenAI Responses API
- [ ] Support for the new Responses API format
- [ ] Built-in tools (web search, code interpreter, file search)
- [ ] Reasoning models support (o1, o3)

#### Vertex AI
- [ ] Google Cloud Vertex AI support
- [ ] Authentication via service accounts
- [ ] Gemini models on Vertex

#### LiteLLM Integration
- [ ] LiteLLM as a provider option
- [ ] Access to 100+ LLM providers through single interface
- [ ] Fallback and load balancing support

### Observability

#### Langfuse Integration
- [ ] Automatic tracing of agent runs
- [ ] Token usage tracking
- [ ] Cost monitoring
- [ ] Prompt management integration

### Advanced Features

#### Skills Support
- [ ] Pluggable skill system
- [ ] Pre-built skills library
- [ ] Custom skill development API
- [ ] Skill composition and chaining

### Documentation & Examples

- [ ] Comprehensive provider-specific examples
- [ ] Migration guides from other frameworks
- [ ] Best practices documentation
- [ ] Performance tuning guide

## Contributing

We welcome contributions! If you'd like to help implement any of these features, please:

1. Check the [issues](https://github.com/abi-jey/nagents/issues) for existing discussions
2. Open a new issue to discuss your approach
3. Submit a pull request

## Versioning

- **0.1.x** - Current: OpenAI-compatible API support, core features
- **0.2.x** - Multi-provider support (Gemini, Anthropic, Vertex)
- **0.3.x** - Advanced features (Skills, LiteLLM)
- **0.4.x** - Observability (Langfuse, OpenTelemetry)
- **1.0.0** - Stable release with full feature set
