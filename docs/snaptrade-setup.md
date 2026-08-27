# SnapTrade setup — read-only brokerage import

Connecting a brokerage lets the sidebar fill itself: tickers, dollar holdings and
total value come straight from the user's accounts instead of being typed in.

**Read-only.** Nothing in this integration can place a trade. That is deliberate
— PortfolioAI presents itself as a research and educational tool, and routing
model-generated BUY/SELL instructions into a live account would be a different
product with different obligations.

The feature is **off unless configured**. With the env vars unset,
`GET /snaptrade/status` returns `enabled: false` and the frontend hides the
button, so nothing below is required to run the app.

---

## 1. SnapTrade credentials

Register at [snaptrade.com](https://snaptrade.com) and take the **Client ID** and
**Consumer Key** from the dashboard. Start in their sandbox — it provides test
brokerage accounts you can connect without touching real money.

```bash
SNAPTRADE_CLIENT_ID=...
SNAPTRADE_CONSUMER_KEY=...
```

> SnapTrade bills **per connected user**. Demo sessions expire after 24h and the
> cleanup deletes the SnapTrade user, not just our row — see step 4. Skipping
> that leaves orphaned connections that keep costing.

## 2. Encryption key

Each user's `userSecret` is a bearer credential to their brokerage data, so it is
encrypted before storage rather than left readable behind the Supabase
service_role key.

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

```bash
SNAPTRADE_ENCRYPTION_KEY=<the generated key>
```

Rotating this key invalidates every stored connection; users simply reconnect.

## 3. Supabase table

```sql
create table public.snaptrade_principals (
  id             uuid primary key default gen_random_uuid(),
  kind           text        not null default 'session'
                             check (kind in ('session', 'account')),
  token_hash     text        not null unique,
  st_user_id     text        not null unique,
  st_user_secret text        not null,          -- Fernet ciphertext, never plaintext
  account_id     uuid,                          -- Phase 2: real user accounts
  created_at     timestamptz not null default now(),
  expires_at     timestamptz,                   -- null for kind='account'
  last_seen_at   timestamptz
);

create index snaptrade_principals_token_hash_idx on public.snaptrade_principals (token_hash);
create index snaptrade_principals_expires_at_idx on public.snaptrade_principals (expires_at);

-- The API uses the service_role key, which bypasses RLS. Enable it anyway so a
-- leaked anon key cannot read the table.
alter table public.snaptrade_principals enable row level security;
```

Note what is **not** in this table: no positions, no holdings, no account
numbers. Positions are fetched, reconciled, returned to the browser and
discarded. The less personal financial data at rest, the fewer obligations.

## 4. Expiry cleanup

Sessions past `expires_at` are rejected at request time, but the rows and the
SnapTrade users they reference still need removing. Run periodically (a
`pg_cron` job, a scheduled GitHub Action, or manually during the beta):

```sql
select id, st_user_id from public.snaptrade_principals
where kind = 'session' and expires_at < now();
```

For each row, call `DELETE /snaptrade/session` with its token, or delete the
SnapTrade user directly through their API, then remove the row. **Deleting the
row alone leaves a billed connection behind.**

## 5. CORS

```bash
ALLOWED_ORIGINS=https://portfolioai-optimizer.vercel.app,http://localhost:5500
```

Defaults to `*`. Worth being precise about what this protects: bearer tokens live
in `sessionStorage`, which is already origin-scoped, so `*` does not by itself
let another site read a visitor's holdings. Setting an allowlist is defence in
depth and limits who can reach the API at all.

---

## Verifying

```bash
cd api
pytest -q                          # 20 tests, no SnapTrade credentials needed
curl localhost:8000/snaptrade/status
```

`{"enabled": true, "max_tickers": 15}` means the credentials are live. Then open
the app, click **Connect brokerage**, and link a sandbox account.

Things worth checking explicitly on a first real connection:

- An account with **more than 15 positions** — the modal should pre-tick the top
  15 by value, list the rest unticked, and block Import above 15.
- An account holding a **mutual fund or foreign listing** — it should appear
  under "cannot be modelled" with a reason, never silently vanish.
- **Timing.** Every imported portfolio is a unique ticker set, so it always
  misses the model cache. Expect ~40s for a 15-ticker run and confirm your
  platform's proxy does not cut the request first.
