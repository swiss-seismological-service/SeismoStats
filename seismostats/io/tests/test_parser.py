import os
import xml.sax
from xml.sax._exceptions import SAXParseException

import numpy as np
import pytest
import requests
import responses

from seismostats.io.parser import (QuakeMLHandler, parse_quakeml,
                                   parse_quakeml_file, parse_quakeml_response)

OUT = [
    {
        "eventID": "smi:ch.ethz.sed/sc20a/Event/2021zqxyri",
        "event_type": "earthquake",
        "time": "2021-12-30T07:43:14.681975Z",
        "latitude": "46.05144527",
        "latitude_uncertainty": "0.1222628824",
        "longitude": "7.388024848",
        "longitude_uncertainty": "0.1007121534",
        "depth": "1181.640625",
        "depth_uncertainty": "274.9552879",
        "evaluationmode": "manual",
        "originID":
        "smi:ch.ethz.sed/sc20ag/Origin/NLL.20220103070248.816904.80080",
        "associatedphasecount": "90",
        "usedphasecount": "77",
        "usedstationcount": "46",
        "standarderror": "0.1050176803",
        "azimuthalgap": "94.07142612",
        "secondaryazimuthalgap": "110.6864591",
        "maximumdistance": "0.6359060797",
        "minimumdistance": "0.0329221581",
        "mediandistance": "0.2980450328",
        "magnitude": "2.510115344",
        "magnitude_uncertainty": "0.23854491",
        "magnitude_type": "MLhc",
        "magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20220103070310.700951.80206",
        "magnitude_MLhc": "2.510115344",
        "magnitude_MLhc_uncertainty": "0.23854491",
        "magnitude_MLhc_magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20220103070310.700951.80206",
        "magnitude_MLv": "2.301758471",
        "magnitude_MLv_uncertainty": "0.2729312832",
        "magnitude_MLv_magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20220103070310.752473.80241",
        "magnitude_MLh": "2.49",
        "magnitude_MLh_uncertainty": "0.23854491",
        "magnitude_MLh_magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20220103070310.700951.80206.mlhc2mlh",
        "magnitude_MWspec": "2.59",
        "magnitude_MWspec_uncertainty": "0.04",
        "magnitude_MWspec_confidenceLevel": "0.68",
        "magnitude_MWspec_magnitudeID":
        "smi:ch.ethz.sed/wfa.ethz.ch/magnitude/MWspec/IbD1W8X9zRumJCb"
    },
    {
        "eventID": "smi:ch.ethz.sed/sc20a/Event/2021zihlix",
        "event_type": "earthquake",
        "time": "2021-12-25T14:49:40.125942Z",
        "latitude": "47.37175484",
        "latitude_uncertainty": "0.1363265577",
        "longitude": "6.917056725",
        "longitude_uncertainty": "0.1277685645",
        "depth": "3364.257812",
        "depth_uncertainty": "1036.395075",
        "evaluationmode": "manual",
        "originID":
        "smi:ch.ethz.sed/sc20ag/Origin/NLL.20211228194249.917108.210045",
        "associatedphasecount": "236",
        "usedphasecount": "44",
        "usedstationcount": "29",
        "standarderror": "0.1058308462",
        "azimuthalgap": "77.46649376",
        "secondaryazimuthalgap": "77.46649376",
        "maximumdistance": "0.5447263866",
        "minimumdistance": "0.134102326",
        "mediandistance": "0.4363983875",
        "magnitude": "3.539687307",
        "magnitude_uncertainty": "0.272435385",
        "magnitude_type": "MLhc",
        "magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20211228194308.87278.210164",
        "magnitude_MLhc": "3.539687307",
        "magnitude_MLhc_uncertainty": "0.272435385",
        "magnitude_MLhc_magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20211228194308.87278.210164"
    },
    {
        "eventID": "smi:ch.ethz.sed/sc20a/Event/2021zhdzar",
        "event_type": "earthquake",
        "time": "2021-12-24T23:59:56.706839Z",
        "latitude": "47.37349438",
        "latitude_uncertainty": "0.140019287",
        "longitude": "6.918607095",
        "longitude_uncertainty": "0.1174757855",
        "depth": "4423.828125",
        "depth_uncertainty": "692.7322216",
        "evaluationmode": "manual",
        "originID":
        "smi:ch.ethz.sed/sc20ag/Origin/NLL.20211227163318.114678.110414",
        "associatedphasecount": "390",
        "usedphasecount": "52",
        "usedstationcount": "36",
        "standarderror": "0.1059982097",
        "azimuthalgap": "77.37160287",
        "secondaryazimuthalgap": "142.3355406",
        "maximumdistance": "0.5356435142",
        "minimumdistance": "0.1340932153",
        "mediandistance": "0.4623895208",
        "magnitude": None,
        "magnitude_type": None
    },
    {
        "eventID": "smi:ch.ethz.sed/sc20a/Event/2021zamwcn",
        "event_type": "earthquake",
        "time": "2021-12-21T08:56:46.30756Z",
        "latitude": "47.07531705",
        "latitude_uncertainty": "0.1183643238",
        "longitude": "6.613590312",
        "longitude_uncertainty": "0.1467303134",
        "depth": "5136.71875",
        "depth_uncertainty": "570.791371",
        "evaluationmode": "manual",
        "originID":
        "smi:ch.ethz.sed/sc20ag/Origin/NLL.20211228090222.85681.111790",
        "associatedphasecount": "163",
        "usedphasecount": "35",
        "usedstationcount": "21",
        "standarderror": "0.1125538689",
        "azimuthalgap": "89.86646929",
        "secondaryazimuthalgap": "89.86646929",
        "maximumdistance": "0.5217167727",
        "minimumdistance": "0.05024666266",
        "mediandistance": "0.381235733",
        "magnitude": "2.908839011",
        "magnitude_uncertainty": "0.2414400072",
        "magnitude_type": "MLhc",
        "magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20211228090234.790826.111865",
        "magnitude_MLhc": "2.908839011",
        "magnitude_MLhc_uncertainty": "0.2414400072",
        "magnitude_MLhc_magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20211228090234.790826.111865",
        "magnitude_MLv": "2.86",
        "magnitude_MLv_uncertainty": "0.2414400072",
        "magnitude_MLv_magnitudeID":
        "smi:ch.ethz.sed/sc20ag/Magnitude/20211228090234.790826.111865.mlhc2mlh"
    }
]


def test_quakeml_handler():
    catalog = []
    handler = QuakeMLHandler(catalog)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(current_dir, 'query.xml')
    parser.parse(xml_file)

    np.testing.assert_equal(sorted(catalog, key=lambda k: k['eventID']),
                            sorted(OUT, key=lambda k: k['eventID']))


def test_parse_quakeml():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(current_dir, 'query.xml')
    with open(xml_file, 'r') as f:
        xml_str = f.read()

    catalog = parse_quakeml(xml_str, include_quality=True)

    np.testing.assert_equal(sorted(catalog, key=lambda k: k['eventID']),
                            sorted(OUT, key=lambda k: k['eventID']))

    catalog = parse_quakeml('')
    assert catalog == []

    catalog = parse_quakeml_file('seismostats/io/tests/empty.xml')
    assert catalog == []

    catalog = parse_quakeml_file('seismostats/io/tests/query.xml')
    assert len(catalog) > 0

    with pytest.raises(SAXParseException):
        catalog = parse_quakeml_file('seismostats/io/tests/wrong.xml')


@responses.activate
def test_parse_quakeml_response():

    # test empty
    rsp = responses.Response(
        method="GET",
        url="http://example.com/nodata",
        status=204,
        body=''
    )
    responses.add(rsp)

    resp = requests.get("http://example.com/nodata", stream=True)

    catalog = parse_quakeml_response(resp)
    assert catalog == []

    # test with data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(current_dir, 'query.xml')
    with open(xml_file, 'rb') as f:
        rsp2 = responses.Response(
            method="GET",
            url="http://example.com/data",
            status=200,
            body=f.read()
        )
        responses.add(rsp2)

        resp2 = requests.get("http://example.com/data", stream=True)

        catalog = parse_quakeml_response(resp2)

        np.testing.assert_equal(sorted(catalog, key=lambda k: k['eventID']),
                                sorted(OUT, key=lambda k: k['eventID']))

    # test invalid data
    rsp3 = responses.Response(
        method="GET",
        url="http://example.com/error",
        status=200,
        body='''</html>
                </html>'''
    )
    responses.add(rsp3)

    resp3 = requests.get("http://example.com/error", stream=True)

    with pytest.raises(SAXParseException):
        catalog = parse_quakeml_response(resp3)
