# Backend proxy

Thin AWS Lambda proxy that keeps the Claude and Voyage API keys out of the APK.

```
Android app ──HTTPS──▶ API Gateway (HTTP API, throttled) ──▶ Lambda (Python)
                                                              │  keys from SSM SecureString (KMS),
                                                              │  cached across warm invocations
                                                              ├──▶ api.anthropic.com/v1/messages
                                                              └──▶ api.voyageai.com/v1/embeddings
```

## Routes

| Route | Forwards to |
|---|---|
| `POST /claude/messages` | `https://api.anthropic.com/v1/messages` |
| `POST /voyage/embeddings` | `https://api.voyageai.com/v1/embeddings` |

The request body is the same JSON the app already sends to the provider; the proxy adds the auth header and applies the limits below.

## Abuse protection

- **Throttling** on the HTTP API stage (token bucket: 2 req/s, burst 5). Best-effort, not a hard cap.
- **Model allowlist**: only the configured Claude models and `voyage-4` are forwarded.
- **`max_tokens` cap**: clamped server-side (default 1024); invalid values get a 400.
- **Input limits**: request body at most 256 KB; Voyage `input` at most 100 strings of 4000 characters.
- **Field allowlist**: only the fields the app uses are forwarded; server-side tools (e.g. web search) are rejected.
- **No streaming**: keeps the proxy simple and bounded.
- **Spend limits**: set a monthly limit in the Anthropic and Voyage consoles as the real backstop.
- **Production answer (not implemented):** Play Integrity attestation so only the genuine app can call the proxy.

## Setup

1. Store the keys (enter the value interactively; it never lands in shell history):

   ```bash
   aws ssm put-parameter --name /rag-proxy/claude-api-key --type SecureString --value "$(read -rs v; echo $v)"
   aws ssm put-parameter --name /rag-proxy/voyage-api-key --type SecureString --value "$(read -rs v; echo $v)"
   ```

2. Build and deploy:

   ```bash
   sam build
   sam deploy --guided   # first time; later just `sam deploy`
   ```

   The `ProxyBaseUrl` output is the URL for the Android client. It ends with `/`, as Retrofit requires.

3. Smoke test (`PROXY` is the `ProxyBaseUrl` value, so no extra `/` before the path):

   ```bash
   curl -s "${PROXY}claude/messages" -H 'content-type: application/json' \
     -d '{"model":"claude-haiku-4-5","max_tokens":50,"messages":[{"role":"user","content":"ping"}]}'

   curl -s "${PROXY}voyage/embeddings" -H 'content-type: application/json' \
     -d '{"model":"voyage-4","input":["ping"]}'
   ```

## IAM: why two permissions

`ssm:GetParameters` (scoped to the two parameter ARNs) lets the function read the parameters; `kms:Decrypt` (restricted with `kms:ViaService` to SSM) lets SSM decrypt the SecureString on the function's behalf. When the parameter is encrypted with a customer-managed KMS key, missing the second one produces an `AccessDeniedException` that looks like an SSM problem but isn't.

## Operations

```bash
sam logs --stack-name mobile-rag-agent-proxy --tail   # Lambda logs (no request or response bodies)
sam delete                                            # remove the stack and all its resources
```

Request and response bodies are deliberately not logged, since they can contain user notes.