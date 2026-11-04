# Telegram stock scanner: setup guide

The scanner runs on GitHub Actions. Every Sunday evening it chooses the week's top 3 picks and sends them to you on Telegram. Every weekday after the US market closes it updates a paper account and sends a short summary. Do these steps once, in order.

## Paper only

No orders are ever placed. The scanner only suggests and keeps a pretend record. To act on a pick, use the Pluang app yourself, or ask Claude, during US market hours.

## Setup

### 1. Create the Telegram bot

1. In Telegram, open a chat with `@BotFather` and send `/newbot`.
2. Choose a name and a username (it must end in `bot`).
3. BotFather replies with a token. This is `TELEGRAM_BOT_TOKEN`. Treat it as a password.

### 2. Get your chat id

1. Open a chat with your new bot and send it any message, such as "hi".
2. In a browser, open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`, replacing `<YOUR_TOKEN>` with the token from step 1. Remember the token is a secret, so do this on your own device and do not share the page or a screenshot.
3. Find `"chat":{"id": ...}` in the reply. That number is `TELEGRAM_CHAT_ID`.

### 3. Google Cloud service account

1. Go to https://console.cloud.google.com and create a project.
2. In "APIs and services", enable the Google Drive API and the Google Sheets API.
3. In "IAM and admin", then "Service accounts", create a service account.
4. Open it, go to "Keys", then "Add key", then "Create new key", and choose JSON. A key file downloads. This file is a secret: do not commit it or share it.
5. Note the service account's email address (it looks like `name@project.iam.gserviceaccount.com`).

### 4. Share the Drive folder and a Google Sheet

1. Open the Drive folder that will hold the archive. Its address looks like `https://drive.google.com/drive/folders/<your folder id>`: the folder id is the last part of that address, after `/folders/`. This is `DRIVE_FOLDER_ID`. Share the folder with the service account's email as Editor. A shared drive is the safest place for this folder, because service accounts have no storage of their own.
2. Create a new Google Sheet and share it with the same email as Editor.
3. Copy the sheet id from its URL: it is the long text between `/d/` and `/edit`. This is `SHEET_ID`.
4. You can keep a tab called "Notes" for your own notes. The job only writes the tabs Picks, Open, WinLoss and Scores.

### 5. Add the repository secrets

In GitHub: Settings, then "Secrets and variables", then Actions, then "New repository secret". Add all five:

| Secret | What to put in it |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | The token from step 1 |
| `TELEGRAM_CHAT_ID` | The chat id from step 2 |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | The whole contents of the JSON key file from step 3 |
| `DRIVE_FOLDER_ID` | `<your folder id>`: the folder id from step 4 |
| `SHEET_ID` | The sheet id from step 4 |

### 6. Set your account size

Edit `scanner.yaml` and set `paper_account.start_equity` to your real account size, so position sizes make sense. Commit and push the change.

### 7. Check Drive access

Do this early, before you rely on the schedule. On your computer, set `GOOGLE_SERVICE_ACCOUNT_JSON` and `DRIVE_FOLDER_ID` in your shell (`check-drive` needs only these two), then run:

```
uv run python -m scanner check-drive
```

If you get a quota error, the folder must live in your own Drive (shared with the service account) or in a shared drive. Service accounts have no storage of their own.

### 8. Try a dry run

```
uv run python -m scanner weekly --dry-run
```

This writes to `.dryrun/` and prints to the console. It does not touch Telegram or Google, and it leaves `scanner_state/` alone. It still downloads live data from Yahoo Finance and Wikipedia, so it needs an internet connection and takes a few minutes.

### 9. Allow the workflows to save state

In GitHub, open Settings, then Actions, then General. Under "Workflow permissions", choose "Read and write permissions" and save. The jobs push `scanner_state/` back to `main` after each run, so also make sure `main` has no branch protection rule that blocks this push by the GitHub Actions bot.

### 10. Run it for real

In GitHub, open the Actions tab, choose "scanner-weekly", and press "Run workflow". Check Telegram for the message. After that, leave it alone: the schedule takes over (weekly on Sunday 22:00 UTC, daily on weekdays at 21:30 UTC).

## Commands

Send these to the bot in Telegram. They take effect from the next run: the next daily or weekly job reads them first, before anything else. That is usually within about a day (the Sunday weekly, then the Monday evening daily). After Friday evening it can take up to about 2 days, over the weekend, until the Sunday weekly.

- `/profile balanced`, `/profile value` or `/profile momentum`: switch the scoring profile.
- `/toggle <indicator>`: turn one indicator on or off.
- `/took <TICKER>`: tell the scanner you took the pick, so it tracks it as yours.

## What the messages mean

- Weekly message: this week's top 3 picks (fewer if not enough stocks pass the checks), each with the suggested entry, stop, size and the reasons behind the score. The paper entry fills at the next Monday US open.
- Daily message: what happened to open paper positions (stops, exits, fills) and the paper account's state.
- Error alert: something failed. The state is still saved, and the next run tries again.

## State files

The job keeps its record in `scanner_state/` and commits it back to the repository after each run (message `scanner: update state`). Nothing else is committed. The big data archive lives on Drive, not in the repository.

## Reset

To start the paper record over, delete both `scanner_state/ledger.csv` and `scanner_state/state.json` (peak equity, the weekly-run marker and any messages still waiting to be re-sent, which are then dropped), then commit. Only do this if you really want to throw away the history.

Deleting only the ledger leaves a stale peak equity behind, which can pause buys.

Keep the other files: `settings.json` (your profile and indicator choices), `telegram_offset.json` (so old commands are not replayed) and `manifest.csv` (the archive history).

## Known limits

- GitHub scheduled runs can be delayed by several minutes.
- On US exchange holidays the weekday run still happens but finds no new price bar, so nothing changes.
- The daily run is timed after the US close on purpose, so it never acts on a half-finished day. If you start it by hand while the US market is open (Monday to Friday, 09:30 to 16:15 New York time), it stops with an error instead of running.
- New buys pause once paper equity is 20% below its peak, and the pause stays until you reset (see Reset). It does not lift by itself after positions are closed.
- GitHub turns off scheduled workflows after 60 days without repository activity. The state commits after each run normally count as activity, but if the schedule stops, re-enable it in the Actions tab.
- `config_version` covers only the profile, the indicator toggles and the weights. Editing `sizing:` or `gates:` in `scanner.yaml` does not start a new version, so results before and after such an edit are counted together.
- The stop and quantity are worked out from the pick-day close, but the paper entry fills at the next open, so the real distance to the stop can differ a little.
- Prices are adjusted for dividends, so a stop can shift slightly relative to past prices.
- Yahoo Finance may refuse requests when too many arrive at once ("Too Many Requests"). The scanner waits a little between stocks and retries a refused request a few times. If Yahoo keeps refusing, it stops asking for the rest of that batch, and later batches in the same run no longer wait to retry. All retry waiting in one run is capped at 8 minutes in total; after that each stock is asked once only. Earnings dates are fetched first, so a run that cannot get them stops early.
- The weekly run fetches fundamentals, analyst data and earnings dates only for stocks that already passed the price checks (trend, momentum, fresh data, not already held). The archived `fundamentals`, `analyst` and `earnings` files cover those stocks only.
- If earnings dates cannot be fetched for most of those stocks (more than 70%, when at least 10 stocks are checked), the weekly run stops with an error alert and sends no picks, because the earnings check would be blind. You can run it again later the same day. If more than 30% are missing, the picks are still sent with a DATA WARNING line.
- A pick marked "earnings date unknown" was not checked against its earnings date. Check it yourself before acting.
- Each run prints `elapsed <N>s` at the end. The workflows stop after 30 minutes, so compare against that if Yahoo is slow.
- The archived `scores` file holds the scored candidates only. Stocks that failed a check (liquidity, trend, earnings, already held) are listed with their reason in `rejects`.
