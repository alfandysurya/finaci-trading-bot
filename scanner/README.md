# Telegram stock scanner: setup guide

The scanner runs on GitHub Actions. Every Sunday evening it picks a stock and sends it to you on Telegram. Every weekday after the US market closes it updates a paper account and sends a short summary. Do these steps once, in order.

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

1. Open the Drive archive folder: https://drive.google.com/drive/folders/1WMlMlhfo9qGcnoAWESNs3WIp6bhv6qXH
   Its id is `1WMlMlhfo9qGcnoAWESNs3WIp6bhv6qXH`. Share the folder with the service account's email as Editor.
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
| `DRIVE_FOLDER_ID` | `1WMlMlhfo9qGcnoAWESNs3WIp6bhv6qXH` |
| `SHEET_ID` | The sheet id from step 4 |

### 6. Set your account size

Edit `scanner.yaml` and set `paper_account.start_equity` to your real account size, so position sizes make sense. Commit and push the change.

### 7. Check Drive access

On your computer, set the five environment variables in your shell, then run:

```
uv run python -m scanner check-drive
```

If you get a quota error, the folder must live in your own Drive (shared with the service account) or in a shared drive. Service accounts have no storage of their own.

### 8. Try a dry run

```
uv run python -m scanner weekly --dry-run
```

This writes to `.dryrun/` and prints to the console. It does not touch Telegram or Google.

### 9. Run it for real

In GitHub, open the Actions tab, choose "scanner-weekly", and press "Run workflow". Check Telegram for the message. After that, leave it alone: the schedule takes over (weekly on Sunday 22:00 UTC, daily on weekdays at 21:30 UTC).

## Commands

Send these to the bot in Telegram. They are processed at the next run, which can be up to a day later.

- `/profile balanced`, `/profile value` or `/profile momentum`: switch the scoring profile.
- `/toggle <indicator>`: turn one indicator on or off.
- `/took <TICKER>`: tell the scanner you took the pick, so it tracks it as yours.

## What the messages mean

- Weekly message: this week's pick, with the suggested entry, stop, size and the reasons behind the score. The paper entry fills at the next Monday US open.
- Daily message: what happened to open paper positions (stops, exits, fills) and the paper account's state.
- Error alert: something failed. The state is still saved, and the next run tries again.

## State files

The job keeps its record in `scanner_state/` and commits it back to the repository after each run (message `scanner: update state`). Nothing else is committed. The big data archive lives on Drive, not in the repository.

## Reset

To start the record over, delete `scanner_state/ledger.csv` and commit. Only do this if you really want to throw away the history.

## Known limits

- GitHub scheduled runs can be delayed by several minutes.
- On US exchange holidays the weekday run still happens but finds no new price bar, so nothing changes.
- The daily run is timed after the US close on purpose, so it never acts on a half-finished day.
