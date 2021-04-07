r"""
A Reader simply reads data from disk and returns it _almost_ as is.
Readers should be utilized by PyTorch :class:`~torch.utils.data.Dataset`.
Too much of data pre-processing is not recommended in the reader, such as
tokenizing words to integers, embedding tokens, or passing an image through
a pre-trained CNN. Each reader must implement at least two methods:

    1. ``__len__`` to return the length of data this Reader can read.
    2. ``__getitem__`` to return data based on an index or a
        primary key (such as ``image_id``).
"""
import os
import json
import h5py
import spacy
import pickle
import numpy as np
from tqdm import tqdm
from absl import logging
from collections import defaultdict
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from typing import Any, Dict, List, Tuple, Optional

# Custom function
import constants
# from updown.utils.reader_utils_legacy import _process_detection_label
from updown.utils.pointer_preprocessing import (
    classid_to_label, tokenize_for_pointer,
    DETECTION_PLACEHOLDER_PREFIX)


# fmt: off
# List of punctuations taken from pycocoevalcap -
# these are ignored during evaluation.
PUNCTUATIONS: List[str] = [
    "''", "'", "``", "`", "(", ")", "{", "}",
    ".", "?", "!", ",", ":", "-", "--", "...", ";"
]
# fmt: on

spacy_nlp_sm = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser", "tagger"])


class ImageFeaturesReader(object):
    r"""
    A reader for H5 files containing pre-extracted image features.
    A typical image features file should have at least two H5 datasets,
    named ``image_id`` and ``features``. It may optionally have other H5
    datasets, such as ``boxes`` (for bounding box coordinates), ``width``
    and ``height`` for image size, and others. This reader only reads image
    features, because our UpDown captioner baseline does not require anything
    other than image features.

    Example of an h5 file::

        image_bottomup_features.h5
        |--- "image_id" [shape: (num_images, )]
        |--- "features" [shape: (num_images, num_boxes, feature_size)]
        +--- .attrs {"split": "coco_train2017"}

    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing image ids and features corresponding
        to one of the four splits used: "coco_train2017", "coco_val2017",
        "nocaps_val", "nocaps_test".
    """

    def __init__(self, features_h5path: str) -> None:
        features_h5 = h5py.File(features_h5path, "r")
        image_id_np = np.array(features_h5["image_id"])

        # Indices in the files to read features from,
        # a mapping of image id to index of features in H5 file.
        # Dict[int, Union[int, np.ndarray]]
        image_id_map = {image_id_np[index]: index
                        for index in range(image_id_np.shape[0])}

        # A mapping from detection-IDs to string
        with open(constants.DETECTION_LABEL_MAP_FILE) as f:
            label_map = json.load(f)

        self._map = image_id_map
        self._label_map = label_map
        self.features_h5 = features_h5
        self.features_h5path = features_h5path


    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, image_id: int) -> Tuple[np.int64,
                                                  np.ndarray,
                                                  np.ndarray,
                                                  np.ndarray,
                                                  np.ndarray,
                                                  np.ndarray]:

        # @np.vectorize
        # def _detection_id_to_label(detection_id):
        #     # Map the ID to label
        #     label_dict = self._label_map[detection_id]

        #     # Assert the ID is correct
        #     if label_dict["id"] != detection_id:
        #         raise ValueError

        #     # Process the text
        #     return _process_detection_label(label_dict["name"])

        if image_id not in self._map.keys():
            num_detections = 0
            detection_boxes = np.array([], dtype=np.float32)
            detection_scores = np.array([], dtype=np.float32)
            detection_classes = np.array([], dtype=np.uint32)
            detection_features = np.array([], dtype=np.float32)

        else:
            index = self._map[image_id]
            num_detections = self.features_h5["num_boxes"][index]
            detection_boxes = self.features_h5["boxes"][index]
            detection_scores = self.features_h5["scores"][index]
            detection_classes = self.features_h5["classes"][index]
            detection_features = self.features_h5["features"][index]

        # Map the detection classes to readable texts
        # We also have to handle the case when the num_detections is zero
        # if num_detections > 0:
        #     detection_labels = _detection_id_to_label(
        #         detection_id=detection_classes)
        # else:
        #     detection_labels = np.array([])
        detection_labels = np.array([
            f"{DETECTION_PLACEHOLDER_PREFIX}"
            f"{np.argwhere(detection_classes == c).flatten()[0]}"
            for c in detection_classes])

        return (num_detections,
                detection_boxes,
                detection_scores,
                detection_labels,
                detection_classes,
                detection_features)


class CocoCaptionsReader(object):
    r"""
    A reader for annotation files containing training captions.
    These are JSON files in COCO format.

    Parameters
    ----------
    captions_jsonpath : str
        Path to a JSON file containing training captions in
        COCO format (COCO train2017 usually).
    """

    def __init__(self,
                 captions_jsonpath: str,
                 parallel_prefer: Optional[str]=None,
                 cache_captions: bool=constants.CACHE_PROCESSED_CAPTIONS
                 ) -> None:

        if not os.path.isfile(captions_jsonpath):
            raise FileNotFoundError(f"{captions_jsonpath} not exist")

        # Saving cache files into local directory
        # instead of globally shared disk
        class_name = self.__class__.__name__.lower()
        directory, filename = os.path.split(captions_jsonpath)
        cached_caption_file = f"{filename}.{class_name}.cached"
        cached_file_location = os.path.join("data/cached_data/",cached_caption_file)

        self._captions_jsonpath = captions_jsonpath


        if os.path.exists(cached_file_location) and cache_captions is True:
            # If cache file exists, we will load from cache directly
            self._load_from_cache(cached_file_location)

        else:
            # Otherwise, we will create the captions and save them into cache
            logging.info(f"Tokenizing captions from {captions_jsonpath}...")

            # Read the data from the path
            with open(captions_jsonpath) as cap:
                captions_json: Dict[str, Any] = json.load(cap)

            # Process the captions into List of (image id, caption) tuples.
            self._captions = (
                Parallel(n_jobs=-1, prefer=parallel_prefer)(
                    delayed(self._process_single_caption_item)(caption_item)
                    for caption_item in tqdm(captions_json["annotations"])))

            if cache_captions:
                self._save_to_cache(cached_caption_file)

    def __len__(self):
        return len(self._captions)

    def __getitem__(self, index) -> Tuple[int, List[str]]:
        return self._captions[index]

    def _process_single_caption_item(self, caption_item):
        caption: str = caption_item["caption"].lower().strip()
        caption_tokens: List[str] = spacy_tokenize(caption)
        caption_tokens = [ct for ct in caption_tokens
                          if ct not in PUNCTUATIONS]

        return caption_item["image_id"], caption_tokens

    def _load_from_cache(self, cached_file):
        # Load the captions from cached file
        logging.info(f"Loading captions from {cached_file}")
        with open(cached_file, "rb") as handle:
            self._captions = pickle.load(handle)

    def _save_to_cache(self, cached_file):
        # Save captions to cached file
        logging.info(f"Caching features into {cached_file}")
        with open(cached_file, "wb") as handle:
            pickle.dump(self._captions, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


class CocoCaptionsPointerReader(CocoCaptionsReader):
    def __init__(self,
                 captions_jsonpath: str,
                 cache_captions: bool=constants.CACHE_PROCESSED_CAPTIONS
                 ) -> None:

        # Note that NOCAPS will not need this function
        if captions_jsonpath == constants.NOCAPS_ANNO_TRAIN_FNAME:
            features_h5path = constants.NOCAPS_IMG_FEATURES_TRAIN_FNAME

        if captions_jsonpath == constants.NOCAPS_ANNO_VALID_FNAME:
            features_h5path = constants.NOCAPS_IMG_FEATURES_VALID_FNAME

        self._features_reader = ImageFeaturesReader(
            features_h5path=features_h5path)

        # detection_label_map_sequences:
        # Key: Integer Image-ID
        # Value: List of detection_label_map_sequence for each image-id
        self._detection_label_map_sequences: Dict[
            int, List[List[Dict[str, str]]]] = defaultdict(list)

        # cache_captions = os.path.join("data/cached_data/",captions_jsonpath)

        super(CocoCaptionsPointerReader, self).__init__(
            captions_jsonpath=captions_jsonpath,
            parallel_prefer="threads",
            cache_captions=cache_captions)

    def _process_single_caption_item(self, caption_item):
        # Fetch the detection labels
        (num_detections, _, _, _,
         detection_classes, _) = self._features_reader[
            caption_item["image_id"]]

        if num_detections == 0:
            # np.vectorize has troubles
            # when the inputs are empty
            detection_labels = []
        else:
            detection_labels = classid_to_label(
                detection_classes[:num_detections]).tolist()

        caption: str = caption_item["caption"].lower().strip()
        (_, caption_tokens,
         _, detection_label_map_sequence) = tokenize_for_pointer(
            string=caption,
            detection_labels=detection_labels)

        image_id = caption_item["image_id"]
        caption_tokens = [ct for ct in caption_tokens
                          if ct not in PUNCTUATIONS]

        # We will cache these as well
        # Note we will ignore empty `detection_label_map_sequence`
        if detection_label_map_sequence:
            self._detection_label_map_sequences[image_id].append(
                detection_label_map_sequence)

        return image_id, caption_tokens

    def _load_from_cache(self, cached_file):
        # Load the captions from cached file
        logging.info(f"Loading captions from {cached_file}")
        with open(cached_file, "rb") as handle:
            self._captions = pickle.load(handle)

        # Load the metadata from cached file
        meta_cached_file = f"{cached_file}.meta"
        logging.info(f"Loading metadata from {meta_cached_file}")
        with open(meta_cached_file, "rb") as handle:
            self._detection_label_map_sequences = pickle.load(handle)

    def _save_to_cache(self, cached_file):
        # Save captions to cached file
        logging.info(f"Caching features into {cached_file}")
        with open(cached_file, "wb") as handle:
            pickle.dump(self._captions, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata to cached file
        meta_cached_file = f"{cached_file}.meta"
        logging.info(f"Caching metadata into {meta_cached_file}")
        with open(meta_cached_file, "wb") as handle:
            pickle.dump(self._detection_label_map_sequences, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


def spacy_tokenize(string: str) -> List[str]:
    doc = spacy_nlp_sm(string)
    return [token.text for token in doc]
