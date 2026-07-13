import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

/**
 * Vercel AI SDK openai-compatible provider pointed at the Python engine.
 *
 * The TS gateway does NOT call LLMs itself — it forwards to the Python engine.
 * This provider is used to validate/normalize the OpenAI-compatible contract and
 * is available for future edge-side model routing.
 */
export function engineProvider(baseURL: string, apiKey = "ts-gateway") {
  return createOpenAICompatible({
    name: "agentic-circuit-engine",
    baseURL,
    apiKey,
  });
}
