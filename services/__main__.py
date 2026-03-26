"""Allow running as: python3 -m services"""
from services.bot import main
import asyncio

asyncio.run(main())
