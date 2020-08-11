import logging
import copy
import time 
from opensfm import dataset
from opensfm import exif
from opensfm import matching
from opensfm import tracking
from opensfm import reconstructingion
from opensfm import mesh
from opensfm import types
from timeit import default_timer as timer
from opensfm.context import parallel_map
from opensfm import io
from opensfm import log
from opensfm import dense
import numpy as np


from utils import _extract_exif, detect, detecting_features_report, matching_features_report, tracks_report 
from utils import undistort_reconstructingion
dataset_path = 'data/observatory/'
log.setup()

logger = logging.getLogger('reconstructingion')
logging.getLogger("reconstructingion").setLevel(logging.INFO)

def extract_meta_data( dataset_path):
    start = time.time()
    data = dataset.DataSet(dataset_path)

    exif_overrides = {}
    if data.exif_overrides_exists():
        exif_overrides = data.load_exif_overrides()

    camera_models = {}
    for image in data.images():
        if data.exif_exists(image):
            logging.info('Cargando EXIF existente para {}'.format(image))
            d = data.load_exif(image)
        else:
            logging.info('Extrayendo EXIF para {}'.format(image))
            d = _extract_exif(image, data)

            if image in exif_overrides:
                d.update(exif_overrides[image])

            data.save_exif(image, d)

        if d['camera'] not in camera_models:
            camera = exif.camera_from_exif_metadata(d, data)
            camera_models[d['camera']] = camera

    if data.camera_models_overrides_exists():
        overrides = data.load_camera_models_overrides()
        if "all" in overrides:
            for key in camera_models:
                camera_models[key] = copy.copy(overrides["all"])
                camera_models[key].id = key
        else:
            for key, value in overrides.items():
                camera_models[key] = value
    data.save_camera_models(camera_models)

    end = time.time()
    with open(data.profile_log(), 'a') as fout:
        fout.write('extract_meta_data: {0}\n'.format(end - start))


def detecting_features(dataset_path):
    data = dataset.DataSet(dataset_path)
    images = data.images()

    arguments = [(image, data) for image in images]

    start = timer()
    processes = data.config['processes']
    parallel_map(detect, arguments, processes, 1)
    end = timer()
    with open(data.profile_log(), 'a') as fout:
        fout.write('detecting_features: {0}\n'.format(end - start))

    detecting_features_report(data, end - start)


def matching_features(dataset_path):
    data = dataset.DataSet(dataset_path)
    images = data.images()

    start = timer()
    pairs_matches, preport = matching.match_images(data, images, images)
    matching.save_matches(data, images, pairs_matches)
    end = timer()

    with open(data.profile_log(), 'a') as fout:
        fout.write('matching_features: {0}\n'.format(end - start))
    matching_features_report(data, preport, list(pairs_matches.keys()), end - start)


def creating_tracks(dataset_path):
    data = dataset.DataSet(dataset_path)

    start = timer()
    features_tracks, colors = tracking.load_features(data, data.images())
    features_end = timer()
    matches = tracking.load_matches(data, data.images())
    matches_end = timer()
    tracks_manager = tracking.creating_tracks_manager(features_tracks, colors, matches,
                                                    data.config)
    tracks_end = timer()
    data.save_tracks_manager(tracks_manager)
    end = timer()

    with open(data.profile_log(), 'a') as fout:
        fout.write('creating_tracks: {0}\n'.format(end - start))

    tracks_report(data,
                      tracks_manager,
                      features_end - start,
                      matches_end - features_end,
                      tracks_end - matches_end)
    
    
def reconstructing(dataset_path):
    start = time.time()
    data = dataset.DataSet(dataset_path)
    tracks_manager = data.load_tracks_manager()
    report, reconstructingions = reconstructingion.\
        incremental_reconstructingion(data, tracks_manager)
    end = time.time()
    with open(data.profile_log(), 'a') as fout:
        fout.write('reconstructing: {0}\n'.format(end - start))
    data.save_reconstructingion(reconstructingions)
    data.save_report(io.json_dumps(report), 'reconstructingion.json')
"""
 TO DO
"""    
    
def create_mesh(dataset_path):
    start = time.time()
    data = dataset.DataSet(dataset_path)
    tracks_manager = data.load_tracks_manager()
    reconstructingions = data.load_reconstructingion()

"""  
 TO DO
"""        
def undistort(dataset_path,reconstructingion = None,tracks = None,reconstructingion_index = 0, 
        output = 'undistorted',):
    data = dataset.DataSet(dataset_path)
    udata = dataset.UndistortedDataSet(data, output)
    reconstructingions = data.load_reconstructingion(reconstructingion)

"""
 TO DO
""" 
def computing_depthmaps(dataset_path, subfolder = 'undistorted', interactive = False):
    data = dataset.DataSet(dataset_path)
    udata = dataset.UndistortedDataSet(data, subfolder)
    data.config['interactive'] = interactive
    


def main(dataset_path):
    extract_meta_data(dataset_path)
    detecting_features(dataset_path)
    matching_features(dataset_path)
    creating_tracks(dataset_path)
    reconstructing(dataset_path)
    #create_mesh(dataset_path)
    #undistort(dataset_path)
    #computing_depthmaps(dataset_path)

main(dataset_path)





