import time

from mmcore.base import AGroup
from mmcore.base.userdata.entry import Entry, EntryEndpoint, EntryProtocol, add_entry_safe, add_props_update_support


class ViewerGroup(AGroup):

    def __new__(cls, *args, name="ViewerGroup", user_data_extras=None, entries_support=True, props_update_support=True,
                **kwargs):
        if user_data_extras is None:
            user_data_extras = dict()
        obj = super().__new__(cls, *args, name=name, _user_data_extras=user_data_extras, **kwargs)
        add_props_update_support(obj)
        return obj

    @property
    def object_url(self):

        return self.__class__.__gui_controls__.config.address + self.__class__.__gui_controls__.config.api_prefix

    def add_entry(self, entry: Entry):
        add_entry_safe(self._user_data_extras['entries'], entry)

    def add_entries_support(self):
        if not self.has_entries:
            self.add_user_data_extra("entries", [])

    def add_user_data_extra(self, key, value):
        if key not in self._user_data_extras:
            self._user_data_extras[key] = value
        else:
            pass

    @property
    def entries(self):
        return self._user_data_extras.get('entries', None)

    @entries.setter
    def entries(self, v):
        self._user_data_extras['entries'] = v

    def add_props_update_support(self):
        if not self.has_entries:
            self.add_entries_support()

        self.add_entry(Entry(name="update-props",
                             endpoint=EntryEndpoint(protocol=EntryProtocol.REST,
                                                    url=self.object_url + f"props-update/{self.uuid}")))

    def props_update(self, uuids: list[str], props: dict):
        # recompute_mask = False
        print(props)
        s = time.time()
        # if self.mask_name in props.keys():
        # recompute_mask = True

        ans = super().props_update(uuids, props)
        # if recompute_mask:
        # self.recompute_mask()
        self.make_cache()
        m, sec = divmod(time.time() - s, 60)
        print(f'updated at {m} min, {sec} sec')
        return ans
