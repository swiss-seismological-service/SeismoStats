import xml.sax
from datetime import datetime
from xml.sax import handler, make_parser

import pandas as pd
import requests


def get_realvalue(key, value):
    real_values = ['value', 'uncertainty',
                   'lowerUncertainty', 'upperUncertainty',
                   'confidenceLevel']
    return {f'{key}{v}': f'{value}_{v}' for v in real_values}


EVENT_MAPPINGS = {
    'publicID': 'eventid'
}

ORIGIN_MAPPINGS = {
    **get_realvalue('origintime', 'time'),
    **get_realvalue('originlatitude', 'latitude'),
    **get_realvalue('originlongitude', 'longitude'),
    **get_realvalue('origindepth', 'depth')
}

MAGNITUDE_MAPPINGS = {
    **get_realvalue('magnitudemag', 'magnitude'),
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


def get_preferred_magnitude(magnitudes, id):
    preferred = next((m for m in magnitudes if id
                     == m['magnitudepublicID']), DUMMY_MAGNITUDE)

    magnitudes = [m for m in magnitudes if m['magnitudetype']
                  != preferred['magnitudetype'] or id == m['magnitudepublicID']]

    return preferred, magnitudes


def get_preferred_origin(origins, id):
    return next((o for o in origins if id == o['originpublicID']), DUMMY_ORIGIN)


def select_secondary_magnitudes(magnitudes):
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


def extract_origin(origin):
    origin_dict = {}
    for key, value in ORIGIN_MAPPINGS.items():
        if key in origin:
            origin_dict[value] = origin[key]
    return origin_dict


def extract_magnitude(magnitude):
    magnitude_dict = {}
    for key, value in MAGNITUDE_MAPPINGS.items():
        if key in magnitude:
            magnitude_dict[value] = magnitude[key]
    return magnitude_dict


def extract_secondary_magnitudes(magnitudes):
    magnitude_dict = {}
    for magnitude in magnitudes:
        mappings = get_realvalue(
            'magnitudemag', f'magnitude_{magnitude["magnitudetype"]}')
        for key, value in mappings.items():
            if key in magnitude:
                magnitude_dict[value] = magnitude[key]
    return magnitude_dict


def parse_to_dict(event, origins, magnitudes):
    preferred_origin = next(
        (o for o in origins
         if o['originpublicID'] == event['preferredOriginID']),
        None)

    preferred_magnitude, magnitudes = \
        get_preferred_magnitude(magnitudes, event['preferredMagnitudeID'])

    if magnitudes:
        magnitudes = select_secondary_magnitudes(magnitudes)

    return extract_origin(preferred_origin) | \
        extract_magnitude(preferred_magnitude) | \
        extract_secondary_magnitudes(magnitudes)

# define a Custom ContentHandler class that extends ContenHandler


class CustomContentHandler(xml.sax.ContentHandler):
    def __init__(self, catalog):
        self.catalog = catalog

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
            self.catalog.append(parse_to_dict(
                self.event[-1], self.origin, self.magnitude))
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


start_cat = "2018-01-01T00:00:00"
end_cat = "2019-01-01T00:00:00"

URL = f'https://service.scedc.caltech.edu/fdsnws/event/1/query?starttime={start_cat}&endtime={end_cat}&minmagnitude=4.0&minlatitude=10&minlongitude=-124&maxlatitude=35&maxlongitude=-80'  # &includeallmagnitudes=true'  # noqa
URL2 = f'http://arclink.ethz.ch/fdsnws/event/1/query?starttime={start_cat}&endtime={end_cat}&minmagnitude=2.0&minlatitude=45&minlongitude=5&maxlatitude=48&maxlongitude=11'  # &includeallmagnitudes=true'  # noqa


def main():
    catalog = []

    parser = make_parser()
    parser.setFeature(handler.feature_namespaces, False)
    parser.setContentHandler(CustomContentHandler(catalog))

    r = requests.get(URL2, stream=True)

    r.raw.decode_content = True  # if content-encoding is used decode
    parser.parse(r.raw)
    print(len(pd.DataFrame.from_dict(catalog)))


if __name__ == '__main__':
    main()
