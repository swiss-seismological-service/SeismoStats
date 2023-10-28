import pytest

from seismostats.utils import require_cols

REQUIRED = ['longitude', 'latitude', 'depth',
                         'time', 'magnitude']


class TestCatalog:
    columns = ['name', 'magnitude']

    @require_cols(require=REQUIRED)
    def require(self):
        pass

    @require_cols(require=['name'])
    def require_spec(self):
        pass

    @require_cols(require=REQUIRED, exclude=['magnitude'])
    def require_exclude(self):
        pass

    def test_require(self):
        pytest.raises(AttributeError, self.require)

    def test_require_succeed(self):
        self.columns = REQUIRED
        self.require()

    def test_require_exclude(self):
        self.columns = REQUIRED
        self.columns.remove('magnitude')
        self.require_exclude()

    def test_require_spec(self):
        self.columns = ['name']
        self.require_spec()
