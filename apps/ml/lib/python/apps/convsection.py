__author__ = 'linked0'

import pandas as pd
import numpy as np
import json
import re
from geopy.distance import vincenty

def toJson():
    sect_dict = {}
    sections = []
    sect_data = pd.read_csv('sections.csv')
    sect_names = sect_data[sect_data.section_name.notnull()]
    sect_names_idx = list(sect_names.index)
    sect_names_idx.append(sect_names_idx[0] + sect_data.shape[0])
    prev_idx = -1

    for idx in sect_names_idx:
        if (prev_idx == -1):
            prev_idx = idx
            continue
        sect_name = sect_names.section_name[prev_idx]
        sect = create_sect(sect_name, sect_data.ix[prev_idx:idx-1])
        sections.append({"section_name": sect_name, "routes": sect})
        prev_idx = idx

    sect_dict = {"sections": sections}

    out_file = open("sections.txt", "w")
    json.dump(sect_dict, out_file, indent=4)
    print sect_dict

def create_sect(sect_name, one_sect):
    routes = []
    route_idx = 1
    for idx in one_sect.index:
        route = create_route(sect_name, route_idx, one_sect.ix[idx])
        routes.append({"route_name": make_route_name(route_idx), "nodes": route})
        route_idx += 1

    print "return create_sect"
    return routes

def create_route(sect_name, route_idx, one_route):
    # print one_route
    nodes = []
    route_prefix = sect_name + "_" + make_route_name(route_idx)
    route_name = one_route.route_name
    print "NODE: ", route_name

    if not pd.isnull(route_name):
        route_name = "_" + route_name
    else:
        route_name = ""

    # start route
    latlon = one_route.latlon
    latlonli = re.split(",", latlon)
    nodes.append({"node_name": route_prefix + route_name,
               "lat": latlonli[0],
               "lon": latlonli[1]})

    #end route
    latlon = one_route.latlonend
    latlonli = re.split(",", latlon)
    nodes.append({"node_name": route_prefix + route_name + "#last",
               "lat": latlonli[0],
               "lon": latlonli[1]})


    return nodes

def make_route_name(idx):
    return "r" + str(idx)


def subRoute(start_lat, start_lon, end_lat, end_lon, divide_meter):
    start = (start_lat, start_lon)
    end = (end_lat, end_lon)
    dist = vincenty(start, end).meters
    divide_count = int(dist/divide_meter)
    print "dist: %f, divide_count: %d" % (dist, divide_count)

    unitLat = (end_lat - start_lat) / (divide_count)
    unitLon = (end_lon - start_lon) / (divide_count)
    print "unitLat: %f, unitLon: %f" % (unitLat, unitLon)

    for i in range(divide_count):
        newLat = start_lat + (unitLat * i)
        newLat = round(newLat, 6)
        newLon = start_lon + (unitLon * i)
        newLon = round(newLon, 6)
        print "%f,%f" % (newLat, newLon)

    print "%f,%f" % (end_lat, end_lon)

def distance(start_lat, start_lon, end_lat, end_lon):
    start = (start_lat, start_lon)
    end = (end_lat, end_lon)
    dist = vincenty(start, end).meters
    print "dist: %f" % (dist)

