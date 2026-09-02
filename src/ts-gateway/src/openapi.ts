const contextHeaders = [
  {
    name: "X-OpenWebUI-User-Id",
    description: "Stable user identifier used to scope persistent memory.",
  },
  {
    name: "X-User-Id",
    description: "Alternative stable user identifier.",
  },
  {
    name: "X-OpenWebUI-Instance-Id",
    description: "Optional tenant or OpenWebUI instance namespace.",
  },
  {
    name: "X-OpenWebUI-Workspace-Id",
    description: "Optional workspace namespace.",
  },
  {
    name: "X-Workspace-Id",
    description: "Alternative workspace namespace.",
  },
  {
    name: "X-Project-Id",
    description: "Optional project namespace for long-term memory.",
  },
  {
    name: "X-OpenWebUI-Chat-Id",
    description: "Optional conversation namespace for temporary memory.",
  },
  {
    name: "X-Conversation-Id",
    description: "Alternative conversation namespace.",
  },
].map(({ name, description }) => ({
  name,
  in: "header",
  required: false,
  description,
  schema: { type: "string" },
}));

export const openapiDocument = {
  openapi: "3.1.0",
  info: {
    title: "Agentic Circuit API",
    version: "0.2.0",
    description:
      "OpenAI-compatible gateway for Agentic Circuit, including multi-perspective reasoning, isolated long-term memory, provider administration, and streaming completions.",
    license: {
      name: "MIT",
      identifier: "MIT",
    },
  },
  servers: [{ url: "/", description: "Current gateway" }],
  tags: [
    { name: "Chat", description: "OpenAI-compatible inference endpoints." },
    { name: "Providers", description: "Runtime provider configuration." },
    { name: "System", description: "Gateway health and service metadata." },
  ],
  paths: {
    "/healthz": {
      get: {
        tags: ["System"],
        summary: "Gateway health check",
        operationId: "gatewayHealth",
        responses: {
          "200": {
            description: "Gateway is running.",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  required: ["status"],
                  properties: { status: { type: "string", examples: ["ok"] } },
                },
              },
            },
          },
        },
      },
    },
    "/v1/models": {
      get: {
        tags: ["Chat"],
        summary: "List logical models",
        description: "OpenAI-compatible model list proxied from the Python engine.",
        operationId: "listModels",
        responses: {
          "200": {
            description: "Logical model list.",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/ModelList" },
              },
            },
          },
          "502": { $ref: "#/components/responses/UpstreamError" },
        },
      },
    },
    "/v1/chat/completions": {
      post: {
        tags: ["Chat"],
        summary: "Create a chat completion",
        description:
          "OpenAI-compatible chat completions with Agentic Circuit extensions for prism selection and isolated persistent memory. Set stream=true for Server-Sent Events.",
        operationId: "createChatCompletion",
        parameters: contextHeaders,
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/ChatCompletionRequest" },
              examples: {
                standard: {
                  summary: "Non-streaming completion with project memory",
                  value: {
                    model: "agentic-circuit",
                    user: "stable-user-id",
                    project_id: "my-project",
                    conversation_id: "chat-42",
                    prism: "neutral",
                    messages: [
                      {
                        role: "user",
                        content: "Summarize the decisions we made for this project.",
                      },
                    ],
                  },
                },
                streaming: {
                  summary: "Streaming completion",
                  value: {
                    model: "agentic-circuit",
                    stream: true,
                    messages: [{ role: "user", content: "Give me three options." }],
                  },
                },
              },
            },
          },
        },
        responses: {
          "200": {
            description:
              "JSON completion when stream=false, or text/event-stream chunks when stream=true.",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/ChatCompletionResponse" },
              },
              "text/event-stream": {
                schema: { type: "string" },
              },
            },
          },
          "400": { $ref: "#/components/responses/BadRequest" },
          "502": { $ref: "#/components/responses/UpstreamError" },
        },
      },
    },
    "/v1/providers": {
      get: {
        tags: ["Providers"],
        summary: "List configured providers",
        operationId: "listProviders",
        security: [{ AdminToken: [] }, { AdminBearer: [] }],
        responses: {
          "200": {
            description: "Provider configuration keyed by provider name.",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  additionalProperties: { $ref: "#/components/schemas/ProviderConfig" },
                },
              },
            },
          },
          "401": { $ref: "#/components/responses/Unauthorized" },
          "503": { $ref: "#/components/responses/AdminDisabled" },
        },
      },
      post: {
        tags: ["Providers"],
        summary: "Upsert a provider",
        description:
          "Writes providers.yaml atomically and reloads the Python engine after validation.",
        operationId: "upsertProvider",
        security: [{ AdminToken: [] }, { AdminBearer: [] }],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/ProviderUpsert" },
              example: {
                name: "example-provider",
                type: "openai-compatible",
                base_url: "https://api.example.com/v1/chat/completions",
                api_key_env: "EXAMPLE_API_KEY",
                models: ["example-model"],
              },
            },
          },
        },
        responses: {
          "200": { $ref: "#/components/responses/ProviderMutation" },
          "400": { $ref: "#/components/responses/BadRequest" },
          "401": { $ref: "#/components/responses/Unauthorized" },
          "503": { $ref: "#/components/responses/AdminDisabled" },
        },
      },
      delete: {
        tags: ["Providers"],
        summary: "Delete a provider",
        operationId: "deleteProvider",
        security: [{ AdminToken: [] }, { AdminBearer: [] }],
        parameters: [
          {
            name: "name",
            in: "query",
            required: true,
            description: "Provider name to delete.",
            schema: { type: "string" },
          },
        ],
        responses: {
          "200": { $ref: "#/components/responses/ProviderMutation" },
          "400": { $ref: "#/components/responses/BadRequest" },
          "401": { $ref: "#/components/responses/Unauthorized" },
          "404": { $ref: "#/components/responses/NotFound" },
          "503": { $ref: "#/components/responses/AdminDisabled" },
        },
      },
    },
  },
  components: {
    securitySchemes: {
      AdminToken: {
        type: "apiKey",
        in: "header",
        name: "X-Admin-Token",
        description: "Value of PROVIDERS_ADMIN_TOKEN.",
      },
      AdminBearer: {
        type: "http",
        scheme: "bearer",
        description: "Alternative bearer-token form of PROVIDERS_ADMIN_TOKEN.",
      },
    },
    schemas: {
      TextContentPart: {
        type: "object",
        required: ["type", "text"],
        properties: {
          type: { type: "string", const: "text" },
          text: { type: "string" },
        },
      },
      ChatMessage: {
        type: "object",
        required: ["role", "content"],
        properties: {
          role: { type: "string", enum: ["user", "assistant"] },
          content: {
            oneOf: [
              { type: "string" },
              {
                type: "array",
                items: { $ref: "#/components/schemas/TextContentPart" },
              },
            ],
          },
        },
      },
      ChatCompletionRequest: {
        type: "object",
        required: ["messages"],
        properties: {
          model: { type: "string", default: "agentic-circuit" },
          messages: {
            type: "array",
            minItems: 1,
            items: { $ref: "#/components/schemas/ChatMessage" },
          },
          stream: { type: "boolean", default: false },
          user: {
            type: "string",
            description: "Stable user identifier used when no user header is supplied.",
          },
          prism: {
            type: "string",
            default: "neutral",
            enum: [
              "joy",
              "flirt",
              "resentment",
              "arousal",
              "anger",
              "apathy",
              "neutral",
              "sadness",
            ],
          },
          memory: {
            type: "boolean",
            default: true,
            description: "Set to false to disable persistent memory for this request.",
          },
          project_id: { type: "string" },
          conversation_id: { type: "string" },
          metadata: {
            type: "object",
            additionalProperties: true,
            description:
              "Optional user_id, tenant_id, workspace_id, project_id, conversation_id, or chat_id context.",
          },
        },
      },
      ChatCompletionResponse: {
        type: "object",
        required: ["id", "object", "created", "model", "choices"],
        properties: {
          id: { type: "string" },
          object: { type: "string", const: "chat.completion" },
          created: { type: "integer" },
          model: { type: "string" },
          choices: {
            type: "array",
            items: {
              type: "object",
              properties: {
                index: { type: "integer" },
                message: {
                  type: "object",
                  properties: {
                    role: { type: "string", const: "assistant" },
                    content: { type: "string" },
                  },
                },
                finish_reason: { type: "string" },
              },
            },
          },
          usage: {
            type: "object",
            properties: {
              prompt_tokens: { type: "integer" },
              completion_tokens: { type: "integer" },
              total_tokens: { type: "integer" },
            },
          },
        },
      },
      ModelList: {
        type: "object",
        required: ["object", "data"],
        properties: {
          object: { type: "string", const: "list" },
          data: {
            type: "array",
            items: {
              type: "object",
              required: ["id", "object"],
              properties: {
                id: { type: "string", examples: ["agentic-circuit"] },
                object: { type: "string", const: "model" },
                created: { type: "integer" },
                owned_by: { type: "string" },
              },
            },
          },
        },
      },
      ProviderConfig: {
        type: "object",
        required: ["type", "base_url", "api_key_env", "models"],
        properties: {
          type: { type: "string", default: "openai-compatible" },
          base_url: { type: "string", format: "uri" },
          api_key_env: { type: "string" },
          models: { type: "array", items: { type: "string" } },
        },
      },
      ProviderUpsert: {
        allOf: [
          { $ref: "#/components/schemas/ProviderConfig" },
          {
            type: "object",
            required: ["name"],
            properties: {
              name: {
                type: "string",
                pattern: "^[a-zA-Z0-9][a-zA-Z0-9._-]*$",
              },
            },
          },
        ],
      },
      Error: {
        type: "object",
        properties: {
          error: {},
          detail: {},
        },
        additionalProperties: true,
      },
    },
    responses: {
      BadRequest: {
        description: "Invalid request.",
        content: { "application/json": { schema: { $ref: "#/components/schemas/Error" } } },
      },
      Unauthorized: {
        description: "Invalid provider administration token.",
        content: { "application/json": { schema: { $ref: "#/components/schemas/Error" } } },
      },
      NotFound: {
        description: "Requested provider was not found.",
        content: { "application/json": { schema: { $ref: "#/components/schemas/Error" } } },
      },
      AdminDisabled: {
        description: "Provider administration is disabled until PROVIDERS_ADMIN_TOKEN is configured.",
        content: { "application/json": { schema: { $ref: "#/components/schemas/Error" } } },
      },
      UpstreamError: {
        description: "The Python engine or upstream model provider could not complete the request.",
        content: { "application/json": { schema: { $ref: "#/components/schemas/Error" } } },
      },
      ProviderMutation: {
        description: "Provider configuration was updated and the engine was reloaded.",
        content: {
          "application/json": {
            schema: {
              type: "object",
              properties: {
                ok: { type: "boolean" },
                reloaded: { type: "boolean" },
                providers: { type: "object", additionalProperties: true },
              },
            },
          },
        },
      },
    },
  },
} as const;

export function scalarReferenceHtml(): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Agentic Circuit API Reference</title>
  </head>
  <body>
    <div id="app"></div>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@1.66.1"></script>
    <script>
      Scalar.createApiReference('#app', {
        url: '/openapi.json',
        theme: 'default'
      })
    </script>
  </body>
</html>`;
}
