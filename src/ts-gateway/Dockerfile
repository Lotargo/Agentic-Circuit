FROM node:20-slim AS build

WORKDIR /app
COPY src/ts-gateway/package.json src/ts-gateway/package-lock.json ./
RUN npm ci
COPY src/ts-gateway/tsconfig.json ./
COPY src/ts-gateway/src/ ./src/
RUN npm run build

FROM node:20-slim AS runtime

WORKDIR /app
ENV NODE_ENV=production \
    TS_GATEWAY_PORT=9191 \
    PROVIDERS_YAML=/app/config/providers.yaml

COPY src/ts-gateway/package.json src/ts-gateway/package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY --from=build /app/dist/ ./dist/
COPY config/ ./config/

EXPOSE 9191
CMD ["npm", "run", "start"]
