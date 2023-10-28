import os
import xml.sax

import numpy as np

from seismostats.io.parser import QuakeMLHandler, parse_quakeml

OUT = [{'depth': '1181.640625',
        'depth_uncertainty': '274.9552879',
        'evaluationmode': 'manual',
        'event_type': 'earthquake',
        'eventid': 'smi:ch.ethz.sed/sc20a/Event/2021zqxyri',
        'latitude': '46.05144527',
        'latitude_uncertainty': '0.1222628824',
        'longitude': '7.388024848',
        'longitude_uncertainty': '0.1007121534',
        'magnitude': '2.510115344',
        'magnitude_MLh': '2.49',
        'magnitude_MLh_magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/'
        '20220103070310.700951.80206.mlhc2mlh',
            'magnitude_MLh_uncertainty': '0.23854491',
            'magnitude_MLhc': '2.510115344',
            'magnitude_MLhc_magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/20220103070310.700951.80206',
            'magnitude_MLhc_uncertainty': '0.23854491',
            'magnitude_MLv': '2.301758471',
            'magnitude_MLv_magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/20220103070310.752473.80241',
            'magnitude_MLv_uncertainty': '0.2729312832',
            'magnitude_MWspec': '2.59',
            'magnitude_MWspec_confidenceLevel': '0.68',
            'magnitude_MWspec_magnitudeid':
            'smi:ch.ethz.sed/wfa.ethz.ch/magnitude/MWspec/IbD1W8X9zRumJCb',
            'magnitude_MWspec_uncertainty': '0.04',
            'magnitude_type': 'MLhc',
            'magnitude_uncertainty': '0.23854491',
            'magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/20220103070310.700951.80206',
            'originid':
            'smi:ch.ethz.sed/sc20ag/Origin/NLL.20220103070248.816904.80080',
            'time': '2021-12-30T07:43:14.681975Z'},
       {'depth': '3364.257812',
        'depth_uncertainty': '1036.395075',
        'evaluationmode': 'manual',
        'event_type': 'earthquake',
        'eventid': 'smi:ch.ethz.sed/sc20a/Event/2021zihlix',
        'latitude': '47.37175484',
        'latitude_uncertainty': '0.1363265577',
        'longitude': '6.917056725',
        'longitude_uncertainty': '0.1277685645',
        'magnitude': '3.539687307',
        'magnitude_MLhc': '3.539687307',
            'magnitude_MLhc_magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/20211228194308.87278.210164',
            'magnitude_MLhc_uncertainty': '0.272435385',
            'magnitude_type': 'MLhc',
            'magnitude_uncertainty': '0.272435385',
            'magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/20211228194308.87278.210164',
            'originid':
            'smi:ch.ethz.sed/sc20ag/Origin/NLL.20211228194249.917108.210045',
            'time': '2021-12-25T14:49:40.125942Z'},
       {'depth': '4423.828125',
        'depth_uncertainty': '692.7322216',
        'evaluationmode': 'manual',
        'event_type': 'earthquake',
        'eventid': 'smi:ch.ethz.sed/sc20a/Event/2021zhdzar',
        'latitude': '47.37349438',
        'latitude_uncertainty': '0.140019287',
        'longitude': '6.918607095',
        'longitude_uncertainty': '0.1174757855',
        'magnitude': None,
        'magnitude_type': None,
            'originid':
            'smi:ch.ethz.sed/sc20ag/Origin/NLL.20211227163318.114678.110414',
            'time': '2021-12-24T23:59:56.706839Z'},
       {'depth': '5136.71875',
        'depth_uncertainty': '570.791371',
        'evaluationmode': 'manual',
        'event_type': 'earthquake',
        'eventid': 'smi:ch.ethz.sed/sc20a/Event/2021zamwcn',
        'latitude': '47.07531705',
        'latitude_uncertainty': '0.1183643238',
        'longitude': '6.613590312',
        'longitude_uncertainty': '0.1467303134',
        'magnitude': '2.908839011',
        'magnitude_MLhc': '2.908839011',
            'magnitude_MLhc_magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/20211228090234.790826.111865',
            'magnitude_MLhc_uncertainty': '0.2414400072',
            'magnitude_MLv': '2.86',
            'magnitude_MLv_magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/'
                '20211228090234.790826.111865.mlhc2mlh',
            'magnitude_MLv_uncertainty': '0.2414400072',
            'magnitude_type': 'MLhc',
            'magnitude_uncertainty': '0.2414400072',
            'magnitudeid':
            'smi:ch.ethz.sed/sc20ag/Magnitude/20211228090234.790826.111865',
            'originid':
            'smi:ch.ethz.sed/sc20ag/Origin/NLL.20211228090222.85681.111790',
            'time': '2021-12-21T08:56:46.30756Z'}]


def test_quakeml_handler():
    catalog = []
    handler = QuakeMLHandler(catalog)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(current_dir, 'query.xml')
    parser.parse(xml_file)

    np.testing.assert_equal(sorted(catalog, key=lambda k: k['eventid']),
                            sorted(OUT, key=lambda k: k['eventid']))


def test_parse_quakeml():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(current_dir, 'query.xml')
    with open(xml_file, 'r') as f:
        xml_str = f.read()

    catalog = parse_quakeml(xml_str)

    np.testing.assert_equal(sorted(catalog, key=lambda k: k['eventid']),
                            sorted(OUT, key=lambda k: k['eventid']))
