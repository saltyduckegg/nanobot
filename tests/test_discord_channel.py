import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from nanobot.channels.discord import DiscordChannel
from nanobot.config.schema import DiscordConfig
from nanobot.bus.queue import MessageBus
from nanobot.bus.events import OutboundMessage

@pytest.fixture
def mock_config():
    config = MagicMock(spec=DiscordConfig)
    config.enabled = True
    config.token = "fake_token"
    config.allow_from = []
    return config

@pytest.fixture
def mock_bus():
    return MagicMock(spec=MessageBus)

@pytest.fixture
def mock_discord_client():
    with patch("nanobot.channels.discord.discord.Client") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.start = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_instance.wait_until_ready = AsyncMock()
        mock_instance.is_ready.return_value = True
        yield mock_instance

def test_init(mock_config, mock_bus):
    with patch("nanobot.channels.discord.discord.Client"):
        channel = DiscordChannel(mock_config, mock_bus)
        assert channel.name == "discord"
        assert channel.config == mock_config

@pytest.mark.asyncio
async def test_start(mock_config, mock_bus, mock_discord_client):
    channel = DiscordChannel(mock_config, mock_bus)

    # We need to mock asyncio.create_task or run to not block if start awaits forever
    # But channel.start() awaits client.start() which we mocked.

    await channel.start()

    mock_discord_client.start.assert_called_with("fake_token")
    assert channel.is_running

@pytest.mark.asyncio
async def test_stop(mock_config, mock_bus, mock_discord_client):
    channel = DiscordChannel(mock_config, mock_bus)
    channel._running = True

    await channel.stop()

    mock_discord_client.close.assert_called_once()
    assert not channel.is_running

@pytest.mark.asyncio
async def test_send(mock_config, mock_bus, mock_discord_client):
    channel = DiscordChannel(mock_config, mock_bus)
    channel._client = mock_discord_client

    mock_channel = AsyncMock()
    mock_discord_client.get_channel.return_value = mock_channel

    msg = OutboundMessage(
        channel="discord",
        chat_id="123456",
        content="Hello world"
    )

    await channel.send(msg)

    mock_discord_client.get_channel.assert_called_with(123456)
    mock_channel.send.assert_called_with("Hello world")

@pytest.mark.asyncio
async def test_on_message(mock_config, mock_bus, mock_discord_client):
    channel = DiscordChannel(mock_config, mock_bus)
    channel._client = mock_discord_client

    mock_message = MagicMock()
    mock_message.author.id = 123
    mock_message.author.name = "user"
    mock_message.author.discriminator = "0000"
    mock_message.channel.id = 456
    mock_message.channel.name = "general"
    mock_message.content = "hello"
    mock_message.attachments = []
    mock_message.guild.id = 789

    # Mock bus.publish_inbound
    mock_bus.publish_inbound = AsyncMock()

    await channel._on_message(mock_message)

    mock_bus.publish_inbound.assert_called_once()
    call_args = mock_bus.publish_inbound.call_args[0][0]

    assert call_args.channel == "discord"
    assert call_args.sender_id == "123|user"
    assert call_args.chat_id == "456"
    assert call_args.content == "hello"
