import express from "express";
import { openapiDocument, scalarReferenceHtml } from "./openapi.js";
import { chatCompletions, listModels } from "./routes/chat.js";
import {
  deleteProvider,
  listProviders,
  upsertProvider,
} from "./routes/providers.js";

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "4mb" }));

const PORT = Number(process.env.TS_GATEWAY_PORT || 9191);

app.get("/openapi.json", (_req, res) => res.json(openapiDocument));
app.get("/docs", (_req, res) => res.type("html").send(scalarReferenceHtml()));

app.get("/v1/models", listModels);
app.post("/v1/chat/completions", chatCompletions);

app.get("/v1/providers", listProviders);
app.post("/v1/providers", upsertProvider);
app.delete("/v1/providers", deleteProvider);

app.get("/healthz", (_req, res) => res.json({ status: "ok" }));

app.listen(PORT, () => {
  console.log(`[agentic-circuit-gateway] listening on :${PORT}`);
  console.log(`[agentic-circuit-gateway] API reference: http://127.0.0.1:${PORT}/docs`);
});
