# Dormant: In-App Support Flow (chat → Supabase + email)

Status: **dormant / parked**. The app currently handles support the simple way —
the chat assistant just gives out `hi@axelpeluso.com` as plain text. This document
captures the fuller "capture + notify" flow we built and then parked, so it can be
revived later without rediscovering the details.

## What it did

When a user reported a bug or asked for help in the floating chat:

1. Claude collected the user's **name, email, and issue** conversationally.
2. It ended its reply with a hidden control tag: `[SUPPORT_READY: name | email | issue]`.
3. The backend **hid the tag** from the streamed text and returned the parsed
   `{name, email, issue}` in the SSE `done` meta event as `support`.
4. The frontend saw `meta.support`, POSTed it to `POST /contact`, and showed
   **"✓ Message sent to the team."**
5. `/contact` stored a row in the Supabase `support` table (durable record) and
   sent an email notification (best-effort, in a background task).

## Why it was parked

- **Railway blocks outbound SMTP** (ports 25/465/587). Raw `smtplib` to
  `smtp.zoho.com:465` hangs until timeout — this caused a 60s `/contact` hang.
- We switched email to the **Resend HTTPS API** (not blocked), but that requires a
  Resend account + API key + domain verification to send from `hi@axelpeluso.com`.
- For the demo we decided the setup overhead wasn't worth it and reverted to simply
  displaying the support email in chat.

The **Supabase row is the reliable part**; email is only the notification layer. If
revived, you can rely on the DB even if email delivery is flaky.

---

## Reactivation checklist

1. **Supabase**: create the `support` table (SQL below).
2. **Railway env vars**: `RESEND_API_KEY`, `SUPPORT_EMAIL_TO` (recipient),
   optionally `SUPPORT_EMAIL_FROM`.
3. **Resend**: sign up, create an API key. To send *from* `hi@axelpeluso.com`,
   verify the `axelpeluso.com` domain (SPF/DKIM DNS records). Until then,
   `SUPPORT_EMAIL_FROM=onboarding@resend.dev` works to the account owner's address.
4. **Backend**: re-add the `[SUPPORT_READY]` instruction to `CHAT_SYSTEM_PROMPT`
   (the rest of the backend plumbing may still be present — see below).
5. **Frontend**: ensure the `meta.support` → `/contact` block is present in `sendChat`.

---

## Supabase table

```sql
create table support (
  id uuid default gen_random_uuid() primary key,
  name text,
  email text,
  message text,
  source text default 'chat',
  created_at timestamp default now()
);

GRANT SELECT, INSERT ON public.support TO service_role;
```

## Railway env vars

| Var | Purpose | Notes |
|-----|---------|-------|
| `RESEND_API_KEY` | Resend HTTP API key | `re_...` |
| `SUPPORT_EMAIL_TO` | Recipient inbox | e.g. `hi@axelpeluso.com` |
| `SUPPORT_EMAIL_FROM` | Sender address | `onboarding@resend.dev` until domain verified, then `hi@axelpeluso.com` |

> Do **not** use raw SMTP / Zoho on Railway — the port is blocked. `ZOHO_APP_PASSWORD`
> is obsolete.

---

## Backend code (`api/main.py`)

### Imports
```python
import requests            # for Resend HTTP API
from fastapi import BackgroundTasks   # add to the fastapi import line
```

### Chat system prompt instruction (add to `CHAT_SYSTEM_PROMPT`)
```
"If the user reports a bug, has a question, or wants to contact support: "
"acknowledge warmly, ask for their name and email, summarize their issue, "
"then end your response with exactly [SUPPORT_READY: <name> | <email> | "
"<issue summary>] on a new line so the system can log it and notify the team. "
"Only include that tag once you actually have their name and email; if either "
"is missing, just ask for it and do not emit the tag yet. The tag is hidden "
"from the user.\n\n"
```

### Control marker (so the tag is stripped from the stream)
```python
CONTROL_MARKERS = ("[REOPTIMIZE]", "[ACTION]", "[SUPPORT_READY")
```

### Tag parser
```python
def _parse_support(text: str):
    """Extract {name, email, issue} from a [SUPPORT_READY: name | email | issue] tag."""
    m = re.search(r"\[SUPPORT_READY:([^\]]*)\]", text, re.DOTALL)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split("|")]
    return {
        "name" : parts[0] if len(parts) > 0 else "",
        "email": parts[1] if len(parts) > 1 else "",
        "issue": parts[2] if len(parts) > 2 else "",
    }
```

### Wire into the `/chat` SSE meta event
```python
reoptimize, action = _parse_actions(full)
support = _parse_support(full)
meta = {"done": True, "reoptimize": reoptimize}
if action is not None:
    meta["action"] = action
if support is not None:
    meta["support"] = support
yield f"data: {json.dumps(meta)}\n\n"
yield "data: [DONE]\n\n"
```

### Request model
```python
class ContactRequest(BaseModel):
    name    : str = ""
    email   : str = ""
    message : str
    source  : str = "chat"
```

### Email helper (Resend HTTP API — NOT smtplib)
```python
def send_support_email(name, email, message, source):
    """Send a support notification via the Resend HTTP API (Railway blocks outbound
    SMTP, so raw smtplib hangs). No-op if RESEND_API_KEY is unset. Self-contained
    (swallows its own errors) so it is safe to run as a background task."""
    api_key = os.getenv("RESEND_API_KEY")
    to_addr = os.getenv("SUPPORT_EMAIL_TO")
    if not api_key or not to_addr:
        return

    sender = os.getenv("SUPPORT_EMAIL_FROM") or "onboarding@resend.dev"

    body = f"""New support request from PortfolioAI

Name:    {name or 'Not provided'}
Email:   {email or 'Not provided'}
Source:  {source}

Message:
{message}

---
Reply directly to {email} to follow up.
"""
    payload = {
        "from"    : f"PortfolioAI <{sender}>",
        "to"      : [to_addr],
        "subject" : f"[PortfolioAI Support] {message[:50]}...",
        "text"    : body,
        "reply_to": email or sender,
    }
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=15,
        )
        if r.status_code >= 400:
            logging.warning(f"Resend email failed {r.status_code}: {r.text[:300]}")
    except Exception as e:
        logging.warning(f"Support email failed: {e}")
```

### Endpoint (email runs in the background so a slow API never blocks the response)
```python
@app.post("/contact")
def contact(request: ContactRequest, background_tasks: BackgroundTasks):
    """Log a support request to Supabase and notify the team by email. Best-effort:
    the Supabase row is the durable record; the email is sent in the background so a
    slow or blocked provider never delays (or hangs) the response."""
    try:
        client = _supabase_client()
        client.table("support").insert({
            "name"   : request.name,
            "email"  : request.email,
            "message": request.message,
            "source" : request.source,
        }).execute()
    except Exception as e:
        logging.warning(f"Supabase contact insert failed: {e}")

    background_tasks.add_task(
        send_support_email, request.name, request.email, request.message, request.source
    )
    return {"success": True}
```

---

## Frontend code (`frontend/index.html`, inside `sendChat`)

### Declare the state variable
```js
let visible = '', reoptimize = false, action = null, support = null, gotText = false;
```

### Read it from the SSE meta event
```js
if (obj.done) { reoptimize = !!obj.reoptimize; action = obj.action || null; support = obj.support || null; }
```

### After the message finishes streaming, POST it
```js
// Support request captured — log it and notify the team
if (support && (support.email || support.name)) {
  try {
    await fetch(`${API}/contact`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: support.name || '', email: support.email || '', message: support.issue || finalText, source: 'chat' })
    });
    addBubble('chat-sys', '✓ Message sent to the team.');
  } catch (_) {
    addBubble('chat-sys', '⚠ Could not send just now — please try again.');
  }
}
```

---

## Gotchas / lessons learned

- **Railway blocks SMTP.** Use an HTTP email API (Resend, SendGrid, Postmark) over
  443, never `smtplib`. Symptom of a blocked port: the request hangs ~60s then times out.
- **Never block the response on email.** Do the durable write (Supabase) synchronously,
  send the email in a `BackgroundTasks` task, and give the HTTP call a timeout.
- **Resend `from` must be a verified domain.** `onboarding@resend.dev` works out of the
  box but (free tier) only delivers to the Resend account owner's address.
- **Control tags must be in `CONTROL_MARKERS`** or they leak into the visible chat
  stream. `_visible_cut` holds back any trailing text that starts one of these markers.
- Alternative to app-side email entirely: a **Supabase Database Webhook / Edge Function**
  that fires on insert into `support` and sends the email — keeps SMTP concerns out of
  the API service.
