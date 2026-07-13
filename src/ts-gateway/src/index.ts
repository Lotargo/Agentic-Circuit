import express from "express";
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

app.get("/v1/models", listModels);
app.post("/v1/chat/completions", chatCompletions);

app.get("/v1/providers", listProviders);
app.post("/v1/providers", upsertProvider);
app.delete("/v1/providers", deleteProvider);

app.get("/healthz", (_req, res) => res.json({ status: "ok" }));

app.listen(PORT, () => {
  console.log(`[ts-gateway] listening on :${PORT}`);
});
