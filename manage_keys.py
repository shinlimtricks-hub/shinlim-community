#!/usr/bin/env python3
"""
BenTech VoiceAI — API Key Manager
===================================
A standalone CLI for creating, listing, and revoking API keys
without needing the server to be running.

USAGE:
  python manage_keys.py create --name "Acme Corp" --tier pro
  python manage_keys.py create --name "Internal dev" --tier admin
  python manage_keys.py create --name "Free user" --tier basic

  python manage_keys.py list
  python manage_keys.py list --tier pro
  python manage_keys.py list --active-only

  python manage_keys.py revoke btk_abc123...
  python manage_keys.py info   btk_abc123...

  python manage_keys.py stats

TIERS:
  admin   unlimited requests · all routes · key management
  pro     500 requests/day  · voice upload allowed
  basic   100 requests/day  · no upload

KEYS FILE:
  Reads/writes keys.json in the current directory.
  Keep this file out of git: add 'keys.json' to .gitignore
"""

import argparse
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

KEYS_FILE = Path("keys.json")

TIER_LIMITS = {
    "admin": None,
    "pro":   500,
    "basic": 100,
}

# ANSI colours
R  = "\033[91m"
G  = "\033[92m"
Y  = "\033[93m"
B  = "\033[94m"
M  = "\033[95m"
C  = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def load_keys() -> dict:
    if not KEYS_FILE.exists():
        return {}
    try:
        return json.loads(KEYS_FILE.read_text())
    except Exception as e:
        print(f"{R}✗ Failed to read {KEYS_FILE}: {e}{RESET}")
        sys.exit(1)


def save_keys(keys: dict) -> None:
    KEYS_FILE.write_text(json.dumps(keys, indent=2))


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def tier_color(tier: str) -> str:
    return {
        "admin": M + BOLD,
        "pro":   Y,
        "basic": C,
    }.get(tier, DIM)


def truncate_key(key: str) -> str:
    """Show first 12 + last 4 chars for display."""
    if len(key) > 20:
        return key[:16] + "…" + key[-4:]
    return key


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_create(args):
    name = args.name.strip()
    tier = args.tier.lower().strip()

    if not name:
        print(f"{R}✗ --name is required{RESET}")
        sys.exit(1)

    if tier not in TIER_LIMITS:
        print(f"{R}✗ Unknown tier '{tier}'. Choose from: {list(TIER_LIMITS)}{RESET}")
        sys.exit(1)

    new_key = "btk_" + secrets.token_hex(24)
    now     = datetime.now(timezone.utc).isoformat()

    record = {
        "key":         new_key,
        "name":        name,
        "tier":        tier,
        "created_at":  now,
        "last_used":   None,
        "revoked":     False,
        "usage_today": 0,
        "usage_date":  "",
        "total_usage": 0,
    }

    keys = load_keys()
    keys[new_key] = record
    save_keys(keys)

    limit = TIER_LIMITS[tier]
    limit_str = f"{limit}/day" if limit else "unlimited"

    print()
    print(f"  {G}{BOLD}✓ API key created{RESET}")
    print()
    print(f"  {DIM}Name  {RESET} {name}")
    print(f"  {DIM}Tier  {RESET} {tier_color(tier)}{tier}{RESET}  ({limit_str})")
    print(f"  {DIM}Key   {RESET} {BOLD}{new_key}{RESET}")
    print()
    print(f"  {DIM}Add to Authorization header:{RESET}")
    print(f"  {C}Authorization: Bearer {new_key}{RESET}")
    print()
    print(f"  {Y}⚠  Store this key securely — it won't be shown again.{RESET}")
    print()


def cmd_list(args):
    keys = load_keys()
    if not keys:
        print(f"\n  {DIM}No keys found in {KEYS_FILE}{RESET}\n")
        return

    # Filter
    items = list(keys.values())
    if args.tier:
        items = [k for k in items if k["tier"] == args.tier]
    if args.active_only:
        items = [k for k in items if not k["revoked"]]

    print()
    print(f"  {BOLD}{'NAME':<24}{'TIER':<10}{'USAGE':<14}{'LAST USED':<22}KEY{RESET}")
    print(f"  {'─'*24}{'─'*10}{'─'*14}{'─'*22}{'─'*20}")

    for k in sorted(items, key=lambda x: x["created_at"], reverse=True):
        revoked_tag = f" {R}[revoked]{RESET}" if k["revoked"] else ""
        tier_str    = f"{tier_color(k['tier'])}{k['tier']}{RESET}"
        limit       = TIER_LIMITS.get(k["tier"])
        usage_str   = f"{k['usage_today']}/{limit or '∞'} today"
        last_used   = (k["last_used"] or "never")[:19].replace("T", " ")
        key_display = DIM + truncate_key(k["key"]) + RESET

        name_col = k["name"][:22] + ("…" if len(k["name"]) > 22 else "")
        print(f"  {name_col:<24}{tier_str:<18}{usage_str:<14}{last_used:<22}{key_display}{revoked_tag}")

    active  = sum(1 for k in items if not k["revoked"])
    revoked = sum(1 for k in items if k["revoked"])
    print()
    print(f"  {DIM}{len(items)} key(s) shown  ·  {active} active  ·  {revoked} revoked{RESET}")
    print()


def cmd_revoke(args):
    target = args.key.strip()
    keys   = load_keys()

    if target not in keys:
        print(f"\n  {R}✗ Key not found: {target[:20]}…{RESET}\n")
        sys.exit(1)

    k = keys[target]
    if k["revoked"]:
        print(f"\n  {Y}⚠  Key '{k['name']}' is already revoked.{RESET}\n")
        return

    # Confirm
    print(f"\n  Revoke key for {BOLD}{k['name']}{RESET} (tier: {k['tier']})?")
    confirm = input(f"  Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print(f"  {DIM}Cancelled.{RESET}\n")
        return

    k["revoked"] = True
    keys[target] = k
    save_keys(keys)
    print(f"\n  {G}✓ Key for '{k['name']}' has been revoked.{RESET}\n")


def cmd_info(args):
    target = args.key.strip()
    keys   = load_keys()

    k = keys.get(target)
    if not k:
        print(f"\n  {R}✗ Key not found{RESET}\n")
        sys.exit(1)

    limit    = TIER_LIMITS.get(k["tier"])
    limit_str = f"{limit}/day" if limit else "unlimited"
    status   = f"{R}REVOKED{RESET}" if k["revoked"] else f"{G}active{RESET}"

    print()
    print(f"  {BOLD}Key info{RESET}")
    print(f"  {'─'*40}")
    print(f"  {DIM}Name        {RESET} {k['name']}")
    print(f"  {DIM}Tier        {RESET} {tier_color(k['tier'])}{k['tier']}{RESET}  ({limit_str})")
    print(f"  {DIM}Status      {RESET} {status}")
    print(f"  {DIM}Key         {RESET} {k['key']}")
    print(f"  {DIM}Created     {RESET} {k['created_at'][:19].replace('T',' ')} UTC")
    print(f"  {DIM}Last used   {RESET} {(k['last_used'] or 'never')[:19].replace('T',' ')}")
    print(f"  {DIM}Today       {RESET} {k['usage_today']} requests")
    print(f"  {DIM}Total       {RESET} {k['total_usage']} requests")
    print()


def cmd_stats(args):
    keys  = load_keys()
    today = today_utc()

    active    = [k for k in keys.values() if not k["revoked"]]
    revoked   = [k for k in keys.values() if k["revoked"]]
    req_today = sum(k["usage_today"] for k in active if k.get("usage_date") == today)
    req_total = sum(k["total_usage"] for k in active)

    by_tier = {}
    for k in active:
        t = k["tier"]
        by_tier[t] = by_tier.get(t, 0) + 1

    print()
    print(f"  {BOLD}BenTech VoiceAI — Key Stats{RESET}")
    print(f"  {'─'*36}")
    print(f"  {DIM}Active keys   {RESET} {len(active)}")
    print(f"  {DIM}Revoked keys  {RESET} {len(revoked)}")
    print(f"  {DIM}Requests today{RESET} {req_today}")
    print(f"  {DIM}Requests total{RESET} {req_total}")
    print()
    print(f"  {DIM}By tier:{RESET}")
    for tier, count in sorted(by_tier.items()):
        print(f"    {tier_color(tier)}{tier:<10}{RESET} {count} key(s)")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="manage_keys",
        description="BenTech VoiceAI — API key manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new API key")
    p_create.add_argument("--name", required=True, help="Human-readable label for this key")
    p_create.add_argument("--tier", default="basic", choices=list(TIER_LIMITS),
                          help="Tier: admin | pro | basic (default: basic)")

    # list
    p_list = sub.add_parser("list", help="List all keys")
    p_list.add_argument("--tier", help="Filter by tier")
    p_list.add_argument("--active-only", action="store_true", help="Exclude revoked keys")

    # revoke
    p_revoke = sub.add_parser("revoke", help="Revoke a key")
    p_revoke.add_argument("key", help="Full key string (btk_...)")

    # info
    p_info = sub.add_parser("info", help="Show details for a key")
    p_info.add_argument("key", help="Full key string (btk_...)")

    # stats
    sub.add_parser("stats", help="Show aggregate usage stats")

    args = parser.parse_args()

    dispatch = {
        "create": cmd_create,
        "list":   cmd_list,
        "revoke": cmd_revoke,
        "info":   cmd_info,
        "stats":  cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
