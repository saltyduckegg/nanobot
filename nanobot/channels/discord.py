"""Discord channel implementation using discord.py."""

import asyncio
from pathlib import Path
from typing import Any

import discord
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import DiscordConfig


class DiscordChannel(BaseChannel):
    """
    Discord channel.
    """

    name = "discord"

    def __init__(self, config: DiscordConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: DiscordConfig = config
        self._client: discord.Client | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Setup discord intents
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True  # Required for reading message content

        self._client = discord.Client(intents=intents)

        # Register event handlers
        @self._client.event
        async def on_ready():
            logger.info(f"Discord bot connected as {self._client.user}")

        @self._client.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self._client.user:
                return

            await self._on_message(message)

    async def start(self) -> None:
        """Start the Discord bot."""
        if not self.config.token:
            logger.error("Discord bot token not configured")
            return

        self._running = True
        logger.info("Starting Discord bot...")

        try:
            await self._client.start(self.config.token)
        except Exception as e:
            logger.error(f"Discord bot failed to start: {e}")
            self._running = False

    async def stop(self) -> None:
        """Stop the Discord bot."""
        self._running = False
        if self._client:
            logger.info("Stopping Discord bot...")
            await self._client.close()

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Discord."""
        if not self._client or not self._client.is_ready():
            logger.warning("Discord bot not ready")
            return

        try:
            chat_id = int(msg.chat_id)
            channel = self._client.get_channel(chat_id)

            if not channel:
                # Try fetching if not in cache
                try:
                    channel = await self._client.fetch_channel(chat_id)
                except Exception:
                    logger.error(f"Could not find Discord channel {chat_id}")
                    return

            if hasattr(channel, "send"):
                # Discord handles markdown natively, so we pass content directly
                await channel.send(msg.content)
            else:
                logger.error(f"Channel {chat_id} does not support sending messages")

        except ValueError:
            logger.error(f"Invalid chat_id: {msg.chat_id}")
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")

    async def _on_message(self, message: discord.Message) -> None:
        """Handle incoming Discord message."""
        sender_id = str(message.author.id)
        # Append username for allowlist compatibility
        if message.author.name:
            sender_id = f"{sender_id}|{message.author.name}"

        chat_id = str(message.channel.id)

        content_parts = []
        media_paths = []

        if message.content:
            content_parts.append(message.content)

        # Handle attachments
        for attachment in message.attachments:
            try:
                # Determine extension
                import mimetypes
                ext = ""
                if attachment.content_type:
                    ext = mimetypes.guess_extension(attachment.content_type) or ""
                if not ext and attachment.filename:
                    ext = Path(attachment.filename).suffix

                # Save to workspace/media/
                media_dir = Path.home() / ".nanobot" / "media"
                media_dir.mkdir(parents=True, exist_ok=True)

                file_path = media_dir / f"{attachment.id}{ext}"
                await attachment.save(file_path)

                media_paths.append(str(file_path))
                content_parts.append(f"[file: {file_path}]")
                logger.debug(f"Downloaded attachment to {file_path}")
            except Exception as e:
                logger.error(f"Failed to download attachment: {e}")
                content_parts.append(f"[file: download failed]")

        content = "\n".join(content_parts) if content_parts else "[empty message]"

        logger.debug(f"Discord message from {sender_id}: {content[:50]}...")

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media_paths,
            metadata={
                "message_id": message.id,
                "user_id": message.author.id,
                "username": message.author.name,
                "discriminator": message.author.discriminator,
                "guild_id": message.guild.id if message.guild else None,
                "channel_name": getattr(message.channel, "name", "dm")
            }
        )
