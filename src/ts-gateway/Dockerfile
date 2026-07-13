# TS OpenAI-compatible gateway
# Build context is the repo root (see docker-compose.yml) so ../config is available.
FROM node:20-slim

WORKDIR /app

COPY src/ts-gateway/package.json ./
RUN npm install

COPY src/ts-gateway/tsconfig.json ./
COPY src/ts-gateway/src/ ./src/
COPY config/ ./config/

ENV TS_GATEWAY_PORT=9191
EXPOSE 9191

CMD ["npm", "run", "start"]
