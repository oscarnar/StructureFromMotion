import logging
from timeit import default_timer as timer

import numpy as np
import cv2
from opensfm import exif
from opensfm import bow
from opensfm import dataset
from opensfm import features
from opensfm import io
from opensfm import log
from opensfm.context import parallel_map
from opensfm import transformations as transf     
from opensfm import types
from opensfm import pysfm
from opensfm import pygeometry


logger = logging.getLogger('reconstruction')

def _extract_exif(image, data):
    d = exif.extract_exif_from_file(data.open_image_file(image))

    if d['width'] <= 0 or not data.config['use_exif_size']:
        d['height'], d['width'] = data.image_size(image)

    d['camera'] = exif.camera_id(d)

    return d


def detect(args):
    image, data = args

    log.setup()

    need_words = data.config['matcher_type'] == 'WORDS' or data.config['matching_bow_neighbors'] > 0
    has_words = not need_words or data.words_exist(image)
    has_features = data.features_exist(image)

    if has_features and has_words:
        logger.info('Skip recomputing {} features for image {}'.format(
            data.feature_type().upper(), image))
        return

    logger.info('Extracting {} features for image {}'.format(
        data.feature_type().upper(), image))

    start = timer()

    p_unmasked, f_unmasked, c_unmasked = features.extract_features(
        data.load_image(image), data.config)

    fmask = data.load_features_mask(image, p_unmasked)

    p_unsorted = p_unmasked[fmask]
    f_unsorted = f_unmasked[fmask]
    c_unsorted = c_unmasked[fmask]

    if len(p_unsorted) == 0:
        logger.warning('No features found in image {}'.format(image))
        return

    size = p_unsorted[:, 2]
    order = np.argsort(size)
    p_sorted = p_unsorted[order, :]
    f_sorted = f_unsorted[order, :]
    c_sorted = c_unsorted[order, :]
    data.save_features(image, p_sorted, f_sorted, c_sorted)

    if need_words:
        bows = bow.load_bows(data.config)
        n_closest = data.config['bow_words_to_match']
        closest_words = bows.map_to_words(
            f_sorted, n_closest, data.config['bow_matcher_type'])
        data.save_words(image, closest_words)

    end = timer()
    report = {
        "image": image,
        "num_features": len(p_sorted),
        "wall_time": end - start,
    }
    data.save_report(io.json_dumps(report), 'features/{}.json'.format(image))


def undistort_reconstruction(tracks_manager, reconstruction, data, udata):
    urec = types.Reconstruction()
    urec.points = reconstruction.points
    utracks_manager = pysfm.TracksManager()

    logger.debug('Undistorting the reconstruction')
    undistorted_shots = {}
    for shot in reconstruction.shots.values():
        if shot.camera.projection_type == 'perspective':
            camera = perspective_camera_from_perspective(shot.camera)
            subshots = [get_shot_with_different_camera(shot, camera)]
        elif shot.camera.projection_type == 'brown':
            camera = perspective_camera_from_brown(shot.camera)
            subshots = [get_shot_with_different_camera(shot, camera)]
        elif shot.camera.projection_type == 'fisheye':
            camera = perspective_camera_from_fisheye(shot.camera)
            subshots = [get_shot_with_different_camera(shot, camera)]
        elif shot.camera.projection_type in ['equirectangular', 'spherical']:
            subshot_width = int(data.config['depthmap_resolution'])
            subshots = perspective_views_of_a_panorama(shot, subshot_width)

        for subshot in subshots:
            urec.add_camera(subshot.camera)
            urec.add_shot(subshot)
            if tracks_manager:
                add_subshot_tracks(tracks_manager, utracks_manager, shot, subshot)
        undistorted_shots[shot.id] = subshots

    udata.save_undistorted_reconstruction([urec])
    if tracks_manager:
        udata.save_undistorted_tracks_manager(utracks_manager)

    arguments = []
    for shot in reconstruction.shots.values():
        arguments.append((shot, undistorted_shots[shot.id], data, udata))

    processes = data.config['processes']
    parallel_map(undistort_image_and_masks, arguments, processes)


def undistort_image_and_masks(arguments):
    shot, undistorted_shots, data, udata = arguments
    log.setup()
    logger.debug('Undistorting image {}'.format(shot.id))

    image = data.load_image(shot.id, unchanged=True, anydepth=True)
    if image is not None:
        max_size = data.config['undistorted_image_max_size']
        undistorted = undistort_image(shot, undistorted_shots, image,
                                      cv2.INTER_AREA, max_size)
        for k, v in undistorted.items():
            udata.save_undistorted_image(k, v)

    mask = data.load_mask(shot.id)
    if mask is not None:
        undistorted = undistort_image(shot, undistorted_shots, mask,
                                      cv2.INTER_NEAREST, 1e9)
        for k, v in undistorted.items():
            udata.save_undistorted_mask(k, v)

    segmentation = data.load_segmentation(shot.id)
    if segmentation is not None:
        undistorted = undistort_image(shot, undistorted_shots, segmentation,
                                      cv2.INTER_NEAREST, 1e9)
        for k, v in undistorted.items():
            udata.save_undistorted_segmentation(k, v)

    detection = data.load_detection(shot.id)
    if detection is not None:
        undistorted = undistort_image(shot, undistorted_shots, detection,
                                      cv2.INTER_NEAREST, 1e9)
        for k, v in undistorted.items():
            udata.save_undistorted_detection(k, v)


def undistort_image(shot, undistorted_shots, original, interpolation,
                    max_size):
    if original is None:
        return

    projection_type = shot.camera.projection_type
    if projection_type in ['perspective', 'brown', 'fisheye']:
        new_camera = undistorted_shots[0].camera
        height, width = original.shape[:2]
        map1, map2 = pygeometry.compute_camera_mapping(shot.camera, new_camera, width, height)
        undistorted = cv2.remap(original, map1, map2, interpolation)
        return {shot.id: scale_image(undistorted, max_size)}
    elif projection_type in ['equirectangular', 'spherical']:
        subshot_width = undistorted_shots[0].camera.width
        width = 4 * subshot_width
        height = width // 2
        image = cv2.resize(original, (width, height), interpolation=interpolation)
        mint = cv2.INTER_LINEAR if interpolation == cv2.INTER_AREA else interpolation
        res = {}
        for subshot in undistorted_shots:
            undistorted = render_perspective_view_of_a_panorama(
                image, shot, subshot, mint)
            res[subshot.id] = scale_image(undistorted, max_size)
        return res
    else:
        raise NotImplementedError(
            'Undistort not implemented for projection type: {}'.format(
                shot.camera.projection_type))




