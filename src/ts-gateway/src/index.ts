import express from "express";
import { chatCompletions } from "./routes/chat.js";
import {
  deleteProvider,
  listProviders,
  upsertProvider,
} from "./routes/providers.js";

const app = express();
app.use(express.json({ limit: "4mb" }));

const PORT = Number(process.env.TS_GATEWAY_PORT || 9191);

// OpenAI-compatible chat endpoint consumed by OpenWebUI.
app.post("/v1/chat/completions", chatCompletions);

// Provider management (driven from OpenWebUI UI / admin).
app.get("/v1/providers", listProviders);
app.post("/v1/providers", upsertProvider);
app.delete("/v1/providers", deleteProvider);

app.get("/healthz", (_req, res) => res.json({ status: "ok" }));

app.listen(PORT, () => {
  console.log(`[ts-gateway] listening on :${PORT}`);
});
