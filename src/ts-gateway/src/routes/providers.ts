import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import type { Request, Response } from "express";
import YAML from "yaml";
import { z } from "zod";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function resolveProvidersYaml(): string {
  const configured = process.env.PROVIDERS_YAML;
  if (configured) {
    return path.resolve(configured);
  }

  const candidates = [
    path.resolve(process.cwd(), "config/providers.yaml"),
    path.resolve(process.cwd(), "../../config/providers.yaml"),
    path.resolve(__dirname, "../../../../config/providers.yaml"),
  ];
  return candidates.find(existsSync) ?? candidates[0];
}

const PROVIDERS_YAML = resolveProvidersYaml();

const providerSchema = z.object({
  name: z
    .string()
    .min(1)
    .regex(/^[a-z0-9][a-z0-9._-]*$/i, "invalid provider name"),
  type: z.string().default("openai-compatible"),
  base_url: z.string().url(),
  api_key_env: z.string().min(1),
  models: z.array(z.string().min(1)).default([]),
});

type ProvidersDocument = {
  providers: Record<string, unknown>;
};

function readProviders(): ProvidersDocument {
  const raw = readFileSync(PROVIDERS_YAML, "utf-8");
  const parsed = YAML.parse(raw);
  if (!parsed || typeof parsed !== "object") {
    return { providers: {} };
  }
  const providers = (parsed as { providers?: unknown }).providers;
  return {
    providers:
      providers && typeof providers === "object"
        ? (providers as Record<string, unknown>)
        : {},
  };
}

function writeProviders(document: ProvidersDocument): void {
  const temporary = `${PROVIDERS_YAML}.tmp`;
  writeFileSync(temporary, YAML.stringify(document), "utf-8");
  renameSync(temporary, PROVIDERS_YAML);
}

function internalError(res: Response, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  res.status(500).json({ error: `provider configuration error: ${message}` });
}

/** GET /v1/providers — list all providers. */
export function listProviders(_req: Request, res: Response) {
  try {
    res.json(readProviders().providers);
  } catch (error) {
    internalError(res, error);
  }
}

/** POST /v1/providers — add or update a provider. */
export function upsertProvider(req: Request, res: Response) {
  const parsed = providerSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.flatten() });
    return;
  }

  try {
    const { name, ...provider } = parsed.data;
    const document = readProviders();
    document.providers[name] = provider;
    writeProviders(document);
    res.json({ ok: true, providers: document.providers });
  } catch (error) {
    internalError(res, error);
  }
}

/** DELETE /v1/providers?name=... — remove a provider. */
export function deleteProvider(req: Request, res: Response) {
  const name = typeof req.query.name === "string" ? req.query.name : undefined;
  if (!name) {
    res.status(400).json({ error: "query param 'name' required" });
    return;
  }

  try {
    const document = readProviders();
    if (!(name in document.providers)) {
      res.status(404).json({ error: `provider '${name}' not found` });
      return;
    }
    delete document.providers[name];
    writeProviders(document);
    res.json({ ok: true, providers: document.providers });
  } catch (error) {
    internalError(res, error);
  }
}
