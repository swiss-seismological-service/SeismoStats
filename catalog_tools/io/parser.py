import xml.sax
from datetime import datetime


def _get_realvalue(key: str, value: str) -> dict:
    real_values = {'value': '',
                   'uncertainty': '_uncertainty',
                   'lowerUncertainty': '_lowerUncertainty',
                   'upperUncertainty': '_upperUncertainty',
                   'confidenceLevel': '_confidenceLevel'}
    return {f'{key}{k}': f'{value}{v}' for k, v in real_values.items()}


EVENT_MAPPINGS = {'publicID': 'eventid',
                  'type': 'event_type'}

ORIGIN_MAPPINGS = {
    **_get_realvalue('origintime', 'time'),
    **_get_realvalue('originlatitude', 'latitude'),
    **_get_realvalue('originlongitude', 'longitude'),
    **_get_realvalue('origindepth', 'depth')
}

MAGNITUDE_MAPPINGS = {
    **_get_realvalue('magnitudemag', 'magnitude'),
    'magnitudetype': 'magnitude_type',
    'magnitudeevaluationMode': 'evaluationMode',
}

DUMMY_MAGNITUDE = {
    'magnitudemagvalue': None,
    'magnitudetype': None,
    'magnitudeevaluationMode': None}

DUMMY_ORIGIN = {
    'origintimevalue': None,
    'originlatitudevalue': None,
    'originlongitudevalue': None,
    'origindepthvalue': None
}


def _get_preferred_magnitude(magnitudes: list, id: str | None) \
        -> tuple[dict, list]:
    preferred = next((m for m in magnitudes if id
                     == m['magnitudepublicID']), DUMMY_MAGNITUDE)

    magnitudes = [m for m in magnitudes if m['magnitudetype']
                  != preferred['magnitudetype'] or id == m['magnitudepublicID']]

    return preferred, magnitudes


def _get_preferred_origin(origins: list, id: str):
    return next((o for o in origins if id == o['originpublicID']), DUMMY_ORIGIN)


def _select_secondary_magnitudes(magnitudes: list):
    """
    Check the magnitudes for multiple magnitudes of the same type and
    select the one with the highest version number and creation time.
    """
    magnitude_types = set(m['magnitudetype'] for m in magnitudes)

    if len(magnitude_types) == len(magnitudes):
        return magnitudes

    selection = []

    for mt in magnitude_types:
        mags = [m for m in magnitudes if m['magnitudetype'] == mt]

        if len(mags) == 1:
            selection.extend(mags)
            continue

        key1 = [False, 'magnitudecreationInfoversion']
        key2 = [False, 'magnitudecreationInfocreationTime']
        key1[0] = all(key1[1] in m for m in mags)
        key2[0] = all(key2[1] in m for m in mags)
        mags = sorted(mags, key=lambda x: (
            float(x[key1[1]]) if key1[0] else None,
            datetime.strptime(x[key2[1]][:19], '%Y-%m-%dT%H:%M:%S')
            if key2[0] else None),
            reverse=True)

        selection.append(mags[0])

    return selection


def _extract_origin(origin: dict) -> dict:
    origin_dict = {}
    for key, value in ORIGIN_MAPPINGS.items():
        if key in origin:
            origin_dict[value] = origin[key]
    return origin_dict


def _extract_magnitude(magnitude: dict) -> dict:
    magnitude_dict = {}
    for key, value in MAGNITUDE_MAPPINGS.items():
        if key in magnitude:
            magnitude_dict[value] = magnitude[key]
    return magnitude_dict


def _extract_secondary_magnitudes(magnitudes: list) -> dict:
    magnitude_dict = {}
    for magnitude in magnitudes:
        mappings = _get_realvalue(
            'magnitudemag', f'magnitude_{magnitude["magnitudetype"]}')
        for key, value in mappings.items():
            if key in magnitude:
                magnitude_dict[value] = magnitude[key]
    return magnitude_dict


def _parse_to_dict(event: dict, origins: list, magnitudes: list,
                   includeallmagnitudes: bool = True) -> dict:
    """
    Parse earthquake event information dictionaries as produced by the
    QuakeMLHandler and return a dictionary of event parameters.

    Args:
        event : dict
            A dictionary representing the earthquake event.
        origins : list
            A list of dictionaries representing the earthquake origins.
        magnitudes : list
            A list of dictionaries representing the earthquake magnitudes.
        includeallmagnitudes : bool, optional
            If True, include all magnitudes in the output dictionary.
            Otherwise, only include the preferred magnitude.

    Returns:
        dict
            A dictionary of earthquake event parameters.
    """
    preferred_origin = \
        _get_preferred_origin(origins,
                              event.get('preferredOriginID', None))

    preferred_magnitude, magnitudes = \
        _get_preferred_magnitude(magnitudes,
                                 event.get('preferredMagnitudeID', None))

    if magnitudes and includeallmagnitudes:
        magnitudes = _select_secondary_magnitudes(magnitudes)
    else:
        magnitudes = []

    event_params = \
        {value: event.get(key, None) for key, value in EVENT_MAPPINGS.items()}

    return event_params | \
        _extract_origin(preferred_origin) | \
        _extract_magnitude(preferred_magnitude) | \
        _extract_secondary_magnitudes(magnitudes)


class QuakeMLHandler(xml.sax.ContentHandler):
    """
    A SAX ContentHandler that is used to parse QuakeML files and extract
    earthquake event information.

    Args:
        catalog : Catalog
            A Catalog object to store the extracted earthquake events.
        includeallmagnitudes : bool, optional
            If True, include all magnitudes in the catalog. Otherwise,
            only include the preferred magnitude.
    Notes:
        This class is a SAX ContentHandler, and is used in conjunction
        with an xml.sax parser to extract earthquake event information
        from QuakeML files.
    """

    def __init__(self, catalog, includeallmagnitudes=True):
        self.catalog = catalog
        self.includeallmagnitudes = includeallmagnitudes
        self.event = []
        self.origin = []
        self.magnitude = []

        self.parent = ''
        self.location = ''

    def setter(self, key, value, additional_key=''):
        if self.location in getattr(self, key)[-1]:
            getattr(self, key)[-1][self.location + additional_key] += value
        else:
            getattr(self, key)[-1][self.location + additional_key] = value

    def startElement(self, tagName, attrs):
        if tagName in ['event', 'origin', 'magnitude']:

            self.parent = tagName
            self.location = tagName if tagName != 'event' else ''
            setattr(self, tagName, getattr(self, tagName) + [{}])

            if 'publicID' in attrs:
                self.setter(self.parent, attrs['publicID'], 'publicID')

        elif self.parent != '':
            self.location += tagName

    def endElement(self, tagName):
        if tagName == 'event':
            self.catalog.append(_parse_to_dict(
                self.event[-1], self.origin, self.magnitude,
                includeallmagnitudes=self.includeallmagnitudes))
            self.parent = ''
            self.location = ''
            self.event = []
            self.origin = []
            self.magnitude = []

        elif tagName in ['origin', 'magnitude']:
            self.parent = 'event'

        if self.parent != '':
            self.location = self.location[:-len(tagName)]

    def characters(self, chars):
        if chars.strip() and self.parent:
            self.setter(self.parent, chars.strip())

    def startDocument(self):
        pass

    def endDocument(self):
        pass
