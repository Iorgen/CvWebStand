import json

import aioredis


class ClientRedis:
    redis = None

    def __init__(
            self,
            redis_host: str):
        self.redis_host = redis_host

    async def redis_connection(self):
        if not self.redis:
            self.redis = await aioredis.create_redis_pool(
                    'redis://{}'.format(self.redis_host))

    async def set_on_redis(
            self,
            key: str,
            value: str,
            ttl: int = None):
        await self.redis_connection()

        await self.redis.set(key=key, value=value, expire=ttl)
        return True

    async def get_from_redis(
            self,
            key: str,
            with_delete: bool = False):
        await self.redis_connection()

        result = await self.redis.get(key=key)
        if with_delete is True:
            await self.redis.delete(key=key)
        return json.loads(result) if result else None
