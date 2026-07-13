import { fileURLToPath } from "node:url";
import path from "node:path";
import { readFileSync, writeFileSync } from "node:fs";
import type { Request, Response } from "express";
import YAML from "yaml";
import { z } from "zod";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROVIDERS_YAML =
  process.env.PROVIDERS_YAML ||
  path.resolve(__dirname, "../../../config/providers.yaml");

const providerSchema = z.object({
  name: z.string().min(1),
  type: z.string().default("openai-compatible"),
  base_url: z.string().url(),
  api_key_env: z.string().min(1),
  models: z.array(z.string()).default([]),
});

function readProviders() {
  const raw = readFileSync(PROVIDERS_YAML, "utf-8");
  const doc = YAML.parse(raw) || { providers: {} };
  return doc;
}

function writeProviders(doc: unknown) {
  writeFileSync(PROVIDERS_YAML, YAML.stringify(doc), "utf-8");
}

/** GET /v1/providers — list all providers. */
export function listProviders(_req: Request, res: Response) {
  const doc = readProviders();
  res.json(doc.providers || {});
}

/** POST /v1/providers — add or update a provider. */
export function upsertProvider(req: Request, res: Response) {
  const parsed = providerSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.flatten() });
    return;
  }
  const { name, ...rest } = parsed.data;
  const doc = readProviders();
  doc.providers = doc.providers || {};
  doc.providers[name] = rest;
  writeProviders(doc);
  res.json({ ok: true, providers: doc.providers });
}

/** DELETE /v1/providers?name=... — remove a provider. */
export function deleteProvider(req: Request, res: Response) {
  const name = req.query.name as string | undefined;
  if (!name) {
    res.status(400).json({ error: "query param 'name' required" });
    return;
  }
  const doc = readProviders();
  doc.providers = doc.providers || {};
  if (!(name in doc.providers)) {
    res.status(404).json({ error: `provider '${name}' not found` });
    return;
  }
  delete doc.providers[name];
  writeProviders(doc);
  res.json({ ok: true, providers: doc.providers });
}
