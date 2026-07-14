import { Buffer } from "node:buffer";
import type { Request, Response } from "express";

const ENGINE_URL = (process.env.PY_ENGINE_URL || "http://localhost:8823").replace(
  /\/$/,
  "",
);

const FORWARDED_CONTEXT_HEADERS = [
  "x-openwebui-user-id",
  "x-user-id",
  "x-openwebui-instance-id",
  "x-openwebui-workspace-id",
  "x-workspace-id",
  "x-project-id",
  "x-openwebui-chat-id",
  "x-conversation-id",
] as const;

function engineHeaders(req: Request): Record<string, string> {
  const headers: Record<string, string> = {
    "content-type": "application/json",
  };
  for (const name of FORWARDED_CONTEXT_HEADERS) {
    const value = req.header(name);
    if (value) headers[name] = value;
  }
  return headers;
}

async function forwardError(upstream: globalThis.Response, res: Response) {
  const contentType = upstream.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    res.status(upstream.status).json(await upstream.json());
    return;
  }
  res.status(upstream.status).json({ error: await upstream.text() });
}

/** POST /v1/chat/completions — transparent OpenAI-compatible proxy. */
export async function chatCompletions(req: Request, res: Response) {
  const stream = Boolean(req.body?.stream);
  const controller = new AbortController();
  res.on("close", () => controller.abort());

  try {
    const upstream = await fetch(`${ENGINE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: engineHeaders(req),
      body: JSON.stringify(req.body),
      signal: controller.signal,
    });

    if (!upstream.ok) {
      await forwardError(upstream, res);
      return;
    }

    if (stream && upstream.body) {
      res.status(upstream.status);
      res.setHeader("content-type", "text/event-stream; charset=utf-8");
      res.setHeader("cache-control", "no-cache");
      res.setHeader("connection", "keep-alive");
      res.setHeader("x-accel-buffering", "no");
      res.flushHeaders();

      for await (const chunk of upstream.body) {
        const nodeChunk = Buffer.from(chunk);
        if (!res.write(nodeChunk)) {
          await new Promise<void>((resolve) => res.once("drain", resolve));
        }
      }
      res.end();
      return;
    }

    res.status(upstream.status).json(await upstream.json());
  } catch (error) {
    if (controller.signal.aborted) return;
    const message = error instanceof Error ? error.message : String(error);
    res.status(502).json({ error: `engine unreachable: ${message}` });
  }
}

/** GET /v1/models — expose engine models to OpenWebUI. */
export async function listModels(_req: Request, res: Response) {
  try {
    const upstream = await fetch(`${ENGINE_URL}/v1/models`);
    if (!upstream.ok) {
      await forwardError(upstream, res);
      return;
    }
    res.status(upstream.status).json(await upstream.json());
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    res.status(502).json({ error: `engine unreachable: ${message}` });
  }
}
