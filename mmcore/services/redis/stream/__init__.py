import redis
import redis_om


class StreamConnector:
    group = None

    def __init__(self, name, redis_conn: redis.Redis, **kwargs):
        super().__init__()

        self.__dict__ |= kwargs
        self.name = name
        self.conn = redis_conn
        self.last_pub=None
        self.last_sub = None
        redis_om.get_redis_connection()

    def write_to_stream(self, data, **kwargs):
        last_pub_id = self.xadd(self.name, data, **kwargs)
        return last_pub_id

    def read_from_stream(self, data, min="-", max="+", count=1, **kwargs):
        self.last_sub = self.xrevrange( min=min, max=max, count=count, **kwargs)

    def create_group(self, group_name=None, id="*", **kwargs):
        group = self.group if group_name is None else group_name
        ans = self.conn.xgroup_create(self.name, group, id=id, **kwargs)
        if not ans:
            raise ans

    def xadd(self, item, fields, **kwargs):
        if item is None:
            item = "*"

        return self.conn.xadd(self.name, fields, id=item, **kwargs)

    def xrange(self, min="-", max="+", count=1):
        return self.conn.xrange(self.name, min=min, max=max, count=count)

    def xrevrange(self, min="-", max="+", count=1):
        return self.conn.xrevrange(self.name, min=min, max=max, count=count)

    def get_last(self):
        return self.xrevrange()

    def xread(self, *streams, count=1, **kwargs):
        return self.conn.xread(streams, count=count, **kwargs)

    def xlen(self):
        self.conn.xlen(self.name)

    def __getitem__(self, item):
        min=str(int(item))
        max=str(int(item) + 1)

        return self.xrange(count=1, min=min, max=max)
    def keys(self):
        return [tg.decode().split("-")[0] for tg in list(zip(*self.xrange(count=)))[0]]

    def __len__(self):
        return self.xlen()
