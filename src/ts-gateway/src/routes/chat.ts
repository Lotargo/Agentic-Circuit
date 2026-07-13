import type { Request, Response } from "express";
import { engineProvider } from "../client/pyEngine.js";

const ENGINE_URL = process.env.PY_ENGINE_URL || "http://localhost:8823";

/**
 * POST /v1/chat/completions
 * OpenAI-compatible endpoint. Forwards the request to the Python LangGraph engine
 * and streams the response straight back to the OpenWebUI client.
 */
export async function chatCompletions(req: Request, res: Response) {
  // Touch the provider so the AI SDK contract is wired (engine does the real work).
  engineProvider(`${ENGINE_URL}/v1`);

  const stream = Boolean(req.body?.stream);

  try {
    const upstream = await fetch(`${ENGINE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req.body),
    });

    if (!upstream.ok) {
      const text = await upstream.text();
      res.status(upstream.status).json({ error: text });
      return;
    }

    if (stream && upstream.body) {
      res.setHeader("content-type", "text/event-stream");
      res.setHeader("cache-control", "no-cache");
      res.setHeader("connection", "keep-alive");
      // Pipe the engine's SSE frames verbatim (already OpenAI-compatible).
      for await (const chunk of upstream.body) {
        res.write(chunk);
      }
      res.end();
      return;
    }

    const data = await upstream.json();
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: `engine unreachable: ${(err as Error).message}` });
  }
}
