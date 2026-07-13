import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import type { Request, Response } from "express";
import YAML from "yaml";
import { z } from "zod";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ENGINE_URL = (process.env.PY_ENGINE_URL || "http://localhost:8823").replace(/\/$/, "");
const ADMIN_TOKEN = process.env.PROVIDERS_ADMIN_TOKEN || "";

function resolveProvidersYaml(): string {
  const configured = process.env.PROVIDERS_YAML;
  if (configured) return path.resolve(configured);
  const candidates = [
    path.resolve(process.cwd(), "config/providers.yaml"),
    path.resolve(process.cwd(), "../../config/providers.yaml"),
    path.resolve(__dirname, "../../../../config/providers.yaml"),
  ];
  return candidates.find(existsSync) ?? candidates[0];
}

const PROVIDERS_YAML = resolveProvidersYaml();

const providerSchema = z.object({
  name: z.string().min(1).regex(/^[a-z0-9][a-z0-9._-]*$/i, "invalid provider name"),
  type: z.string().default("openai-compatible"),
  base_url: z.string().url(),
  api_key_env: z.string().min(1),
  models: z.array(z.string().min(1)).default([]),
});

type ProvidersDocument = { providers: Record<string, unknown> };

function authorized(req: Request, res: Response): boolean {
  if (!ADMIN_TOKEN) {
    res.status(503).json({ error: "provider administration is disabled until PROVIDERS_ADMIN_TOKEN is set" });
    return false;
  }
  const bearer = req.header("authorization")?.replace(/^Bearer\s+/i, "");
  const token = req.header("x-admin-token") || bearer;
  if (token !== ADMIN_TOKEN) {
    res.status(401).json({ error: "invalid provider administration token" });
    return false;
  }
  return true;
}

function readProviders(): ProvidersDocument {
  const parsed = YAML.parse(readFileSync(PROVIDERS_YAML, "utf-8"));
  const providers = parsed && typeof parsed === "object" ? (parsed as { providers?: unknown }).providers : undefined;
  return { providers: providers && typeof providers === "object" ? providers as Record<string, unknown> : {} };
}

function writeProviders(document: ProvidersDocument): void {
  const temporary = `${PROVIDERS_YAML}.tmp`;
  writeFileSync(temporary, YAML.stringify(document), "utf-8");
  renameSync(temporary, PROVIDERS_YAML);
}

async function reloadEngine(): Promise<void> {
  const response = await fetch(`${ENGINE_URL}/v1/reload`, { method: "POST" });
  if (!response.ok) throw new Error(`engine reload failed with ${response.status}`);
}

function internalError(res: Response, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  res.status(500).json({ error: `provider configuration error: ${message}` });
}

export function listProviders(req: Request, res: Response) {
  if (!authorized(req, res)) return;
  try {
    res.json(readProviders().providers);
  } catch (error) {
    internalError(res, error);
  }
}

export async function upsertProvider(req: Request, res: Response) {
  if (!authorized(req, res)) return;
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
    await reloadEngine();
    res.json({ ok: true, reloaded: true, providers: document.providers });
  } catch (error) {
    internalError(res, error);
  }
}

export async function deleteProvider(req: Request, res: Response) {
  if (!authorized(req, res)) return;
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
    await reloadEngine();
    res.json({ ok: true, reloaded: true, providers: document.providers });
  } catch (error) {
    internalError(res, error);
  }
}
