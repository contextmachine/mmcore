import unittest


class TestImports(unittest.TestCase):
    def import_case(self):
        try:
            from mmcore.base.basic import Object3D
            from mmcore.base.geom import GqlGeometry
            from mmcore.base.utils import export_edgedata_to_json
            assert Object3D
            assert GqlGeometry
            assert export_edgedata_to_json

        except ImportError as err:
            self.fail(err)


TestImports('import_case')
