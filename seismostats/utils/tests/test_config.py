import pytest

from seismostats.utils import get_option, set_option
from seismostats.utils._config import __options


def test_config():
    warnings = get_option('warnings')
    assert warnings is __options['warnings']

    set_option('warnings', False)
    warnings = get_option('warnings')
    assert warnings is False

    with pytest.raises(KeyError):
        set_option('unknown', 'value')
