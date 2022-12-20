#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
class BaseClientDescriptor(object):

    def __set_name__(self, owner, name):
        self.bucket = owner.bucket
        self.prefix = owner.prefix

        self.name = name

    def __get__(self, instance, owner=None):
        return instance.__client__.get_object(Bucket=instance.bucket, Key=f"{instance.prefix}{self.name}")

    def __set__(self, instance, value):
        instance.__client__.put_object(Bucket=instance.bucket, Key=f"{instance.prefix}{self.name}", Body=value)


class HookDescriptor(BaseClientDescriptor):

    def __set__(self, instance, value):
        super(HookDescriptor, self).__set__(instance, instance.__sethook__(value))

    def __get__(self, instance, owner=None):
        return instance.__gethook__(super(HookDescriptor, self).__get__(instance, owner))


class ClientDescriptor(object):
    def __set_name__(self, owner, name):
        self.bucket = owner.bucket
        self.prefix = owner.prefix
        self.name = name

    def __get__(self, instance, owner=None):
        print(instance, owner)
        return instance.__gethook__(instance.client.get_object(Bucket=self.bucket, Key=f"{self.prefix}{self.name}"))

    def __set__(self, instance, value):
        print(instance, value)
        instance.client.put_object(Bucket=self.bucket, Key=f"{self.prefix}{self.name}{instance.suffix}",
                                   Body=instance.__sethook__(value))
