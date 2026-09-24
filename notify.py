"""Scan results ko Telegram / WhatsApp par bhejna.
Setup (Streamlit Secrets mein daalo):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   — @BotFather se bot banao, phir apna chat_id lo
  WHATSAPP_TOKEN, WHATSAPP_PHONE_ID, WHATSAPP_TO  — Meta WhatsApp Cloud API (advanced setup)
Dhyan rahe: yeh sirf jab app khuli ho aur button dabao (ya auto-refresh chal raha ho) tab hi
bhejta hai — Streamlit free app background mein apne aap 24x7 nahi chalti."""
import streamlit as st

try:
    import requests
except Exception:
    requests = None


def secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None


def telegram_configured():
    return bool(secret("TELEGRAM_BOT_TOKEN") and secret("TELEGRAM_CHAT_ID"))


def whatsapp_configured():
    return bool(secret("WHATSAPP_TOKEN") and secret("WHATSAPP_PHONE_ID") and secret("WHATSAPP_TO"))


def send_telegram(text):
    if requests is None:
        return False, "requests library nahi mili"
    token, chat_id = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return False, "Telegram secrets set nahi hain"
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": text[:4000]}, timeout=10)
        ok = r.status_code == 200 and r.json().get("ok")
        return ok, "" if ok else r.text[:200]
    except Exception as e:
        return False, str(e)


def send_whatsapp(text):
    if requests is None:
        return False, "requests library nahi mili"
    token, phone_id, to = secret("WHATSAPP_TOKEN"), secret("WHATSAPP_PHONE_ID"), secret("WHATSAPP_TO")
    if not (token and phone_id and to):
        return False, "WhatsApp secrets set nahi hain"
    try:
        r = requests.post(
            f"https://graph.facebook.com/v20.0/{phone_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"messaging_product": "whatsapp", "to": to, "type": "text",
                 "text": {"body": text[:4000]}}, timeout=10)
        ok = r.status_code == 200
        return ok, "" if ok else r.text[:200]
    except Exception as e:
        return False, str(e)


def format_results(scan_name, table, limit=25):
    if table is None or len(table) == 0:
        return f"📊 {scan_name}\n\nAbhi koi stock match nahi karta."
    lines = [f"📊 {scan_name} — {len(table)} stocks\n"]
    for sym, row in table.head(limit).iterrows():
        chg = row.get("chg_pct", 0)
        lines.append(f"{sym}: ₹{row.get('close', 0):.2f} ({chg:+.2f}%)")
    if len(table) > limit:
        lines.append(f"...+{len(table) - limit} aur")
    return "\n".join(lines)


def notify_panel(scan_name, table):
    tg, wa = telegram_configured(), whatsapp_configured()
    if not (tg or wa):
        with st.expander("🔔 WhatsApp / Telegram notification setup karo"):
            st.markdown("""
**Telegram (2 minute mein ban jaata hai, free):**
1. Telegram mein **@BotFather** ko message karo → `/newbot` → naam do → aapko ek **token** milega.
2. Apne bot ko ek message bhejo (kuch bhi), phir browser mein kholo:
   `https://api.telegram.org/bot<TOKEN>/getUpdates` — usme `"chat":{"id":...}` milega, wahi **chat_id** hai.
3. Streamlit **Settings → Secrets** mein daalo:
   ```
   TELEGRAM_BOT_TOKEN = "123456:ABC..."
   TELEGRAM_CHAT_ID = "123456789"
   ```

**WhatsApp (advanced, Meta Business account chahiye):**
Meta for Developers → WhatsApp product → access token + phone number ID lo, Secrets mein daalo:
```
WHATSAPP_TOKEN = "..."
WHATSAPP_PHONE_ID = "..."
WHATSAPP_TO = "91XXXXXXXXXX"
```
""")
        return
    cols = st.columns(2)
    if tg and cols[0].button("📨 Telegram par bhejo"):
        ok, err = send_telegram(format_results(scan_name, table))
        (st.success if ok else st.error)("Bheja gaya ✅" if ok else f"Fail: {err}")
    if wa and cols[1].button("📨 WhatsApp par bhejo"):
        ok, err = send_whatsapp(format_results(scan_name, table))
        (st.success if ok else st.error)("Bheja gaya ✅" if ok else f"Fail: {err}")
