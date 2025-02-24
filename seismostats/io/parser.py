import xml.sax
from datetime import datetime
from xml.sax._exceptions import SAXParseException

from requests import Response


def _get_realvalue(key: str, value: str) -> dict:
    real_values = {'value': '',
                   'uncertainty': '_uncertainty',
                   'lowerUncertainty': '_lowerUncertainty',
                   'upperUncertainty': '_upperUncertainty',
                   'confidenceLevel': '_confidenceLevel'}
    return {f'{key}{k}': f'{value}{v}' for k, v in real_values.items()}


EVENT_MAPPINGS = {'publicID': 'eventID',
                  'type': 'event_type'}

ORIGIN_MAPPINGS = {
    **_get_realvalue('origintime', 'time'),
    **_get_realvalue('originlatitude', 'latitude'),
    **_get_realvalue('originlongitude', 'longitude'),
    **_get_realvalue('origindepth', 'depth'),
    'originevaluationMode': 'evaluationmode',
    'originpublicID': 'originID',
}

QUALITY_MAPPINGS = {
    'originqualityassociatedPhaseCount': 'associatedphasecount',
    'originqualityusedPhaseCount': 'usedphasecount',
    'originqualityassociatedStationCount': 'associatedstationcount',
    'originqualityusedStationCount': 'usedstationcount',
    'originqualitydepthPhaseCount': 'depthphasecount',
    'originqualitystandardError': 'standarderror',
    'originqualityazimuthalGap': 'azimuthalgap',
    'originqualitysecondaryAzimuthalGap': 'secondaryazimuthalgap',
    'originqualitymaximumDistance': 'maximumdistance',
    'originqualityminimumDistance': 'minimumdistance',
    'originqualitymedianDistance': 'mediandistance'
}

MAGNITUDE_MAPPINGS = {
    **_get_realvalue('magnitudemag', 'magnitude'),
    'magnitudetype': 'magnitude_type',
    'magnitudepublicID': 'magnitudeID',
}

DUMMY_MAGNITUDE = {
    'magnitudemagvalue': None,
    'magnitudetype': None}

DUMMY_ORIGIN = {
    'origintimevalue': None,
    'originlatitudevalue': None,
    'originlongitudevalue': None,
    'origindepthvalue': None,
    'originevaluationMode': None
}


def SECONDARY_MAGNITUDE_MAPPINGS(type):
    return {
        **_get_realvalue('magnitudemag', f'magnitude_{type}'),
        'magnitudepublicID': f'magnitude_{type}_magnitudeID'}


def _select_magnitude_by_id(magnitudes: list, id: str | None) \
        -> tuple[dict, list]:
    preferred = next((m for m in magnitudes if id
                     == m['magnitudepublicID']), DUMMY_MAGNITUDE)

    magnitudes = [m for m in magnitudes if m['magnitudetype']
                  != preferred['magnitudetype'] or id == m['magnitudepublicID']]

    return preferred, magnitudes


def _select_origin_by_id(origins: list, id: str) -> tuple[dict, list]:
    preferred = next((o for o in origins if id
                     == o['originpublicID']), DUMMY_ORIGIN)
    origins = [o for o in origins if id != o['originpublicID']]
    return preferred, origins


def _select_secondary_magnitudes(magnitudes: list) -> list:
    """
    Check the magnitudes for multiple magnitudes of the same type and
    select the one with the highest version number and creation time.

    Args:
        magnitudes:    The magnitudes to select from and check for multiple
                    versions.

    Returns:
        selected:      The selected magnitudes.
    """
    magnitude_types = set(m['magnitudetype'] for m in magnitudes)

    # only one magnitude per type, return all magnitudes
    if len(magnitude_types) == len(magnitudes):
        return magnitudes

    selection = []

    for mt in magnitude_types:
        mags = [m for m in magnitudes if m['magnitudetype'] == mt]

        if len(mags) == 1:
            selection.extend(mags)
            continue

        # sort mags array by version number and creation time
        key1 = [False, 'magnitudecreationInfoversion']
        key2 = [False, 'magnitudecreationInfocreationTime']
        key1[0] = all(key1[1] in m for m in mags)
        key2[0] = all(key2[1] in m for m in mags)
        mags = sorted(mags, key=lambda x: (
            float(x[key1[1]]) if key1[0] else None,
            datetime.strptime(x[key2[1]][:19], '%Y-%m-%dT%H:%M:%S')
            if key2[0] else None),
            reverse=True)

        # select magnitude with highest version number and creation time
        selection.append(mags[0])

    return selection


def _extract_origin(origin: dict, include_quality: bool) -> dict:
    origin_dict = {}
    for key, value in ORIGIN_MAPPINGS.items():
        if key in origin:
            origin_dict[value] = origin[key]
    if include_quality:
        for key, value in QUALITY_MAPPINGS.items():
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
        mappings = SECONDARY_MAGNITUDE_MAPPINGS(magnitude['magnitudetype'])
        for key, value in mappings.items():
            if key in magnitude:
                magnitude_dict[value] = magnitude[key]
    return magnitude_dict


def _parse_to_dict(event: dict,
                   origins: list[dict],
                   magnitudes: list[dict],
                   include_all_magnitudes: bool = True,
                   include_quality: bool = True) -> dict:
    """
    Parse earthquake event information dictionaries as produced by the
    QuakeMLHandler and return a dictionary of event parameters.

    Args:
        event:          The earthquake event.
        origins:        The earthquake origins.
        magnitudes:     The earthquake magnitudes.
        include_all_magnitudes:     If True, include all magnitudes in the
                    output dictionary. Otherwise, only include
                    the preferred magnitude.
        include_quality:   If True, include quality information in the output
                    dictionary.

    Returns:
        event_params:   Full dictionary of earthquake event parameters.
    """
    preferred_origin, _ = \
        _select_origin_by_id(origins,
                             event.get('preferredOriginID', None))

    preferred_magnitude, magnitudes = \
        _select_magnitude_by_id(magnitudes,
                                event.get('preferredMagnitudeID', None))

    if magnitudes and include_all_magnitudes:
        magnitudes = _select_secondary_magnitudes(magnitudes)
    else:
        magnitudes = []

    event_params = \
        {value: event.get(key, None) for key, value in EVENT_MAPPINGS.items()}

    return event_params | \
        _extract_origin(preferred_origin, include_quality) | \
        _extract_magnitude(preferred_magnitude) | \
        _extract_secondary_magnitudes(magnitudes)


class QuakeMLHandler(xml.sax.ContentHandler):
    """
    A SAX ContentHandler that is used to parse QuakeML files and extract
    earthquake event information.

    Args:
        catalog:        Object to store the extracted earthquake events.
        include_all_magnitudes: If True, include all magnitudes in the catalog.
                    Otherwise, only include the preferred magnitude.
    Notes:
        This class is a SAX ContentHandler, and is used in conjunction
        with an xml.sax parser to extract earthquake event information
        from QuakeML files.
    """

    def __init__(self,
                 catalog,
                 include_all_magnitudes=True,
                 include_quality=True):

        self.catalog = catalog
        self.include_all_magnitudes = include_all_magnitudes
        self.include_quality = include_quality
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
                include_all_magnitudes=self.include_all_magnitudes,
                include_quality=self.include_quality))
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


def parse_quakeml_file(file_path: str,
                       include_all_magnitudes: bool = True,
                       include_quality: bool = True) -> list[dict]:
    """
    Parse a QuakeML file and return a list of earthquake event information
    dictionaries.

    Args:
        file_path :     Path to the QuakeML file.

    Returns:
        events:         A list of earthquake event information dictionaries.
    """
    data = []
    handler = QuakeMLHandler(data, include_all_magnitudes, include_quality)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    try:
        parser.parse(file_path)
    except SAXParseException as e:
        if 'no element found' in str(e):
            return data
        raise e
    return data


def parse_quakeml(quakeml: str,
                  include_all_magnitudes: bool = True,
                  include_quality: bool = True) -> list[dict]:
    """
    Parse a QuakeML string and return a list of earthquake event information
    dictionaries.

    Args:
        quakeml :   A QuakeML string.

    Returns:
        events:     A list of earthquake event information dictionaries.
    """
    data = []

    if quakeml == '':
        return data

    handler = QuakeMLHandler(data, include_all_magnitudes, include_quality)
    xml.sax.parseString(quakeml, handler)
    return data


def parse_quakeml_response(response: Response,
                           include_all_magnitudes: bool = True,
                           include_quality: bool = True) -> list[dict]:
    """
    Parse a QuakeML response and return a list of earthquake event information
    dictionaries.

    Args:
        response:   A response object from a QuakeML request.

    Returns:
        events:     A list of earthquake event information dictionaries.
    """
    response.raw.decode_content = True  # if content-encoding is used decode
    data = []
    handler = QuakeMLHandler(data, include_all_magnitudes, include_quality)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    try:
        parser.parse(response.raw)
    except SAXParseException as e:
        if 'no element found' in str(e):
            return data
        raise e
    return data
