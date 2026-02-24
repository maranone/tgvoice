Telegram bot that acts as a voice/text interface to Claude or Codex. You send it a voice message or text from
  your phone, it transcribes the audio (Whisper), passes it to Claude/Codex CLI, and sends the response back as both
  text and a spoken voice message (Kokoro TTS). Runs locally on your machine, no cloud APIs except Telegram itself.

1. Create the bot
  - Open Telegram, search for @BotFather
  - Send /newbot
  - Choose a name (display name): e.g. tgvoice
  - Choose a username (must end in bot): e.g. tgvoice_bot
  - BotFather gives you a token like 7484729301:ABBcDefGhIjKlMnOpQrStUvWxYz
  - Copy it — this goes into the wizard when you first run the bot

  2. Get your chat ID (for the owner restriction)
  - Search for @userinfobot on Telegram
  - Send it any message
  - It replies with your chat ID, e.g. 123456789
  - This is the owner ID — only your account can talk to the bot

  3. Run the bot for the first time
  python bot.py
  The wizard asks for the token and chat ID, downloads models, then starts.

  4. Find your bot on Telegram
  - Search for the username you picked (e.g. @tgvoice_bot)
  - Send /start
  - Done
