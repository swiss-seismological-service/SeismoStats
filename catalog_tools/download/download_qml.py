import xml.sax
from datetime import datetime
from pprint import pprint
from time import perf_counter
from xml.sax import handler, make_parser

import pandas as pd
import requests
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


class Event:
    def __init__(self):
        self.data = {}
        self.data['origins'] = {}
        self.data['magnitudes'] = {}

        self.real_values = ['value', 'uncertainty',
                            'lowerUncertainty', 'upperUncertainty',
                            'confidenceLevel']

        self.event_mappings = {
            'publicID': 'eventid'
        }
        self.origin_mappings = {
            **self.get_realvalue('origintime', 'time'),
            **self.get_realvalue('originlatitude', 'latitude'),
            **self.get_realvalue('originlongitude', 'longitude'),
            **self.get_realvalue('origindepth', 'depth')
        }

        self.magnitude_mappings = {
            **self.get_realvalue('magnitudemag', 'magnitude'),
            'magnitudetype': 'type',
            'magnitudeevaluationMode': 'evaluationMode',
        }

    def clean_magnitudes(self):
        cleaned_mags = {}
        mag_types = set(m['magnitudetype']
                        for m in self.data['magnitudes'].values())
        for mt in mag_types:
            mags = [m for m in self.data['magnitudes'].values()
                    if m['magnitudetype'] == mt]
            if len(mags) > 1:
                pref = next((m for m in mags if self.data['preferredMagnitudeID']
                            == m['magnitudepublicID']), None)
                if pref is None:
                    key1 = None
                    key2 = None
                    if all('magnitudecreationInfoversion' in m for m in mags):
                        key1 = 'magnitudecreationInfoversion'
                    if all('magnitudecreationInfocreationTime' in m for m in mags):
                        key2 = 'magnitudecreationInfocreationTime'
                    mags = sorted(mags, key=lambda x: (
                        int(x[key1]) if key1 else None, datetime.strptime(x[key2][:19], '%Y-%m-%dT%H:%M:%S' if key2 else None)), reverse=True)
                    pprint(mags)
                    pprint(mags[0])
                    pref = mags[0]
                    raise Exception()
                cleaned_mags[pref['magnitudepublicID']] = pref
            else:
                cleaned_mags[mags[0]['magnitudepublicID']] = mags[0]
        self.data['magnitudes'] = cleaned_mags

    def to_dict(self):
        self.clean_magnitudes()
        result = {}
        for key in self.event_mappings:
            if key in self.data:
                result[self.event_mappings[key]] = self.data[key]
        if self.data['preferredOriginID'] in self.data['origins']:
            for key in self.origin_mappings:
                if key in \
                        self.data['origins'][
                            self.data['preferredOriginID']]:
                    result[self.origin_mappings[key]] = self.data['origins'][
                        self.data['preferredOriginID']][key]
        if self.data['preferredMagnitudeID'] in self.data['magnitudes']:
            for key in self.magnitude_mappings:
                if key in self.data['magnitudes'][
                        self.data['preferredMagnitudeID']]:
                    result[self.magnitude_mappings[key]] = \
                        self.data['magnitudes'][
                        self.data['preferredMagnitudeID']][key]

        return result

    def get_realvalue(self, key, value):
        return {f'{key}{v}': f'{value}_{v}' for v in self.real_values}


# define a Custom ContentHandler class that extends ContenHandler
class CustomContentHandler(xml.sax.ContentHandler):
    def __init__(self, catalog):
        self.catalog = catalog

        self.event_obj = Event()

        self.event = {}
        self.origin = {}
        self.magnitude = {}

        self.parent = ''
        self.location = ''

    def setter(self, key, value, additional_key=''):
        if self.location in getattr(self, key):
            getattr(self, key)[self.location + additional_key] += value
        else:
            getattr(self, key)[self.location + additional_key] = value

    def startElement(self, tagName, attrs):
        if tagName in ['event', 'origin', 'magnitude']:
            self.parent = tagName

            self.location = tagName if tagName != 'event' else ''

            if 'publicID' in attrs:
                self.setter(self.parent, attrs['publicID'], 'publicID')

        elif self.parent != '':
            self.location += tagName

    def endElement(self, tagName):
        if tagName == 'event':
            self.event_obj.data.update(self.event)
            self.catalog.append(self.event_obj.to_dict())
            self.parent = ''
            self.location = ''
            self.event = {}
            self.event_obj = Event()

        elif tagName == 'origin':
            self.event_obj.data['origins'][
                self.origin['originpublicID']] = self.origin
            self.origin = {}
            self.parent = 'event'

        elif tagName == 'magnitude':
            self.event_obj.data['magnitudes'][
                self.magnitude['magnitudepublicID']] = self.magnitude
            self.magnitude = {}
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

URL = f'https://service.scedc.caltech.edu/fdsnws/event/1/query?starttime={start_cat}&endtime={end_cat}&minmagnitude=4.0&minlatitude=10&minlongitude=-124&maxlatitude=35&maxlongitude=-80&includeallmagnitudes=true'  # noqa
URL2 = f'http://arclink.ethz.ch/fdsnws/event/1/query?starttime={start_cat}&endtime={end_cat}&minmagnitude=2.0&minlatitude=45&minlongitude=5&maxlatitude=48&maxlongitude=11&includeallmagnitudes=true'  # noqa


def main():
    start = perf_counter()
    catalog = []

    parser = make_parser()
    parser.setFeature(handler.feature_namespaces, False)
    parser.setContentHandler(CustomContentHandler(catalog))

    r = requests.get(URL, stream=True)

    r.raw.decode_content = True  # if content-encoding is used decode
    parser.parse(r.raw)
    print(pd.DataFrame.from_dict(catalog))
    print(perf_counter() - start)

    # start = perf_counter()
    # client = Client("http://arclink.ethz.ch")
    # starttime = UTCDateTime(start_cat)
    # endtime = UTCDateTime(end_cat)
    # # cat = client.get_events(starttime=starttime, endtime=endtime,
    # #                         minmagnitude=4.0, includeallmagnitudes=True,
    # #                         minlatitude=10, maxlatitude=35,
    # #                         maxlongitude=-80, minlongitude=-124)
    # cat = client.get_events(starttime=starttime, endtime=endtime,
    #                         minmagnitude=2.0, includeallmagnitudes=True,
    #                         minlatitude=45, maxlatitude=48,
    #                         maxlongitude=11, minlongitude=5)
    # print(len(cat))
    # print(perf_counter() - start)


if __name__ == '__main__':
    main()
