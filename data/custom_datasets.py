from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from allennlp.data import Vocabulary

from updown.config import Config
# from updown.data.readers import CocoCaptionsReader, ConstraintBoxesReader, ImageFeaturesReader
from updown.types import (
    TrainingInstance,
    TrainingBatch,
    EvaluationInstance,
    EvaluationInstanceWithConstraints,
    EvaluationBatch,
    EvaluationBatchWithConstraints,
)
from updown.utils.constraints import ConstraintFilter, FiniteStateMachineBuilder

# Custom imports
import constants
from updown.data.custom_readers import CocoCaptionsPointerReader, ImageFeaturesReader


class TrainingDataset(Dataset):
    r"""
    A PyTorch `:class:`~torch.utils.data.Dataset` providing access to COCO train2017 captions data
    for training :class:`~updown.models.updown_captioner.UpDownCaptioner`. When wrapped with a
    :class:`~torch.utils.data.DataLoader`, it provides batches of image features and tokenized
    ground truth captions.

    .. note::

        Use :mod:`collate_fn` when wrapping with a :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    captions_jsonpath: str
        Path to a JSON file containing COCO train2017 caption annotations.
    image_features_h5path: str
        Path to an H5 file containing pre-extracted features from COCO train2017 images.
    max_caption_length: int, optional (default = 20)
        Maximum length of caption sequences for language modeling. Captions longer than this will
        be truncated to maximum length.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        captions_jsonpath: str,
        image_features_h5path: str,
        max_caption_length: int = 20,
        in_memory: bool = True,
    ) -> None:
        # `in_memory` is left unused
        self._vocabulary = vocabulary
        self._image_features_reader = ImageFeaturesReader(image_features_h5path)
        self._captions_reader = CocoCaptionsPointerReader(captions_jsonpath)

        self._max_caption_length = max_caption_length

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            image_features_h5path=_C.DATA.TRAIN_FEATURES,
            captions_jsonpath=_C.DATA.TRAIN_CAPTIONS,
            max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
            in_memory=kwargs.pop("in_memory"),
        )

    def __len__(self) -> int:
        # Number of training examples are number of captions, not number of images.
        return len(self._captions_reader)

    def __getitem__(self, index: int) -> TrainingInstance:
        image_id, caption = self._captions_reader[index]
        (num_detections,
         detection_boxes,
         detection_scores,
         detection_labels,
         detection_classes,
         detection_features) = self._image_features_reader[image_id]

        # Tokenize caption.
        caption_tokens: List[int] = [self._vocabulary.get_token_index(c) for c in caption]

        # Pad upto max_caption_length.
        caption_tokens = caption_tokens[: self._max_caption_length]
        caption_tokens.extend(
            [self._vocabulary.get_token_index("@@UNKNOWN@@")]
            * (self._max_caption_length - len(caption_tokens))
        )

        # Tokenize the detection labels

        detection_labels = [self._vocabulary.get_token_index(label)
                            for label in detection_labels]
    
        # (num_detections x num_units,) to (num_detections, num_units)
        # Note that `num_detections` could potentially be zero
        detection_features = detection_features.reshape(
            [num_detections, constants.IMAGE_FEATURE_DIM])

        item: TrainingInstance = {
            "image_id": image_id,
            "num_detections": num_detections,
            # "detection_boxes": detection_boxes,
            # "detection_scores": detection_scores,
            "detection_labels": detection_labels,
            "detection_classes": detection_classes,
            "detection_features": detection_features,
            "caption_tokens": caption_tokens,
        }
        return item

    def collate_fn(self, batch_list: List[TrainingInstance]) -> TrainingBatch:
        # Convert lists of ``image_id``s and ``caption_tokens``s as tensors.
        image_id = torch.tensor([instance["image_id"] for instance in batch_list]).long()
        num_detections = torch.tensor([instance["num_detections"] for instance in batch_list]).long()
        caption_tokens = torch.tensor(
            [instance["caption_tokens"] for instance in batch_list]
        ).long()

        # Pad adaptive image features in the batch.
        (detection_features,
         detection_labels,
         detection_classes) = _custom_collate_image_features(
            detection_features_list=[instance["detection_features"] for instance in batch_list],
            detection_labels_list=[instance["detection_labels"] for instance in batch_list],
            detection_classes_list=[instance["detection_classes"] for instance in batch_list])

        detection_features = torch.from_numpy(detection_features)
        detection_labels = torch.from_numpy(detection_labels)
        detection_classes = torch.from_numpy(detection_classes)

        batch: TrainingBatch = {
            "image_id": image_id,
            "num_detections": num_detections,
            # "detection_boxes": detection_boxes,
            # "detection_scores": detection_scores,
            "detection_labels": detection_labels,
            "detection_classes": detection_classes,
            "detection_features": detection_features,
            "caption_tokens": caption_tokens,
        }
        return batch


class EvaluationDataset(Dataset):
    r"""
    A PyTorch :class:`~torch.utils.data.Dataset` providing image features for inference. When
    wrapped with a :class:`~torch.utils.data.DataLoader`, it provides batches of image features.

    .. note::

        Use :mod:`collate_fn` when wrapping with a :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    image_features_h5path: str
        Path to an H5 file containing pre-extracted features from nocaps val/test images.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        image_features_h5path: str,
        in_memory: bool = True
    ) -> None:
        # `in_memory` is left unused
        self._vocabulary = vocabulary
        self._image_features_reader = ImageFeaturesReader(image_features_h5path)
        self._image_ids = sorted(list(self._image_features_reader._map.keys()))

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            image_features_h5path=_C.DATA.INFER_FEATURES,
            in_memory=kwargs.pop("in_memory"))

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int) -> EvaluationInstance:
        image_id = self._image_ids[index]
        (num_detections,
         detection_boxes,
         detection_scores,
         detection_labels,
         detection_classes,
         detection_features) = self._image_features_reader[image_id]

        # Tokenize the detection labels
        detection_labels = [self._vocabulary.get_token_index(label)
                            for label in detection_labels]
        # (num_detections x num_units,) to (num_detections, num_units)
        # Note that `num_detections` could potentially be zero
        detection_features = detection_features.reshape(
            [num_detections, constants.IMAGE_FEATURE_DIM])

        item: EvaluationInstance = {
            "image_id": image_id,
            "num_detections": num_detections,
            # "detection_boxes": detection_boxes,
            # "detection_scores": detection_scores,
            "detection_labels": detection_labels,
            "detection_classes": detection_classes,
            "detection_features": detection_features,
        }
        return item

    def collate_fn(self, batch_list: List[EvaluationInstance]) -> EvaluationBatch:
        # Convert lists of ``image_id``s and ``caption_tokens``s as tensors.
        image_id = torch.tensor([instance["image_id"] for instance in batch_list]).long()
        num_detections = torch.tensor([instance["num_detections"] for instance in batch_list]).long()

        # Pad adaptive image features in the batch.
        (detection_features,
         detection_labels,
         detection_classes) = _custom_collate_image_features(
            detection_features_list=[instance["detection_features"] for instance in batch_list],
            detection_labels_list=[instance["detection_labels"] for instance in batch_list],
            detection_classes_list=[instance["detection_classes"] for instance in batch_list])

        detection_features = torch.from_numpy(detection_features)
        detection_labels = torch.from_numpy(detection_labels)
        detection_classes = torch.from_numpy(detection_classes)

        batch: EvaluationBatch = {
            "image_id": image_id,
            "num_detections": num_detections,
            # "detection_boxes": detection_boxes,
            # "detection_scores": detection_scores,
            "detection_labels": detection_labels,
            "detection_classes": detection_classes,
            "detection_features": detection_features,
        }
        return batch


class EvaluationDatasetWithConstraints(EvaluationDataset):
    r"""
    A PyTorch :class:`~torch.utils.data.Dataset` providing image features for inference, along
    with constraints for :class:`~updown.modules.cbs.ConstrainedBeamSearch`. When wrapped with a
    :class:`~torch.utils.data.DataLoader`, it provides batches of image features, Finite State
    Machines built (per instance) from constraints, and number of constraints used to make these.

    Extended Summary
    ----------------
    Finite State Machines as represented as adjacency matrices (Tensors) with state transitions
    corresponding to specific constraint (word) occurrence while decoding). We return the number
    of constraints used to make an FSM because it is required while selecting which decoded beams
    satisfied constraints. Refer :func:`~updown.utils.constraints.select_best_beam_with_constraints`
    for more details.

    .. note::

        Use :mod:`collate_fn` when wrapping with a :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    image_features_h5path: str
        Path to an H5 file containing pre-extracted features from nocaps val/test images.
    boxes_jsonpath: str
        Path to a JSON file containing bounding box detections in COCO format (nocaps val/test
        usually).
    wordforms_tsvpath: str
        Path to a TSV file containing two fields: first is the name of Open Images object class
        and second field is a comma separated list of words (possibly singular and plural forms
        of the word etc.) which could be CBS constraints.
    hierarchy_jsonpath: str
        Path to a JSON file containing a hierarchy of Open Images object classes as
        `here <https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html>`_.
    nms_threshold: float, optional (default = 0.85)
        NMS threshold for suppressing generic object class names during constraint filtering,
        for two boxes with IoU higher than this threshold, "dog" suppresses "animal".
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which can be specified for CBS decoding. Constraints are
        selected based on the prediction confidence score of their corresponding bounding boxes.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        image_features_h5path: str,
        boxes_jsonpath: str,
        wordforms_tsvpath: str,
        hierarchy_jsonpath: str,
        nms_threshold: float = 0.85,
        max_given_constraints: int = 3,
        in_memory: bool = True,
    ):
        super().__init__(image_features_h5path, in_memory=in_memory)

        self._vocabulary = vocabulary
        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")

        self._boxes_reader = ConstraintBoxesReader(boxes_jsonpath)

        self._constraint_filter = ConstraintFilter(
            hierarchy_jsonpath, nms_threshold, max_given_constraints
        )
        self._fsm_builder = FiniteStateMachineBuilder(vocabulary, wordforms_tsvpath)

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            image_features_h5path=_C.DATA.INFER_FEATURES,
            boxes_jsonpath=_C.DATA.CBS.INFER_BOXES,
            wordforms_tsvpath=_C.DATA.CBS.WORDFORMS,
            hierarchy_jsonpath=_C.DATA.CBS.CLASS_HIERARCHY,
            in_memory=kwargs.pop("in_memory"),
        )

    def __getitem__(self, index: int) -> EvaluationInstanceWithConstraints:
        item: EvaluationInstance = super().__getitem__(index)

        # Apply constraint filtering to object class names.
        constraint_boxes = self._boxes_reader[item["image_id"]]

        candidates: List[str] = self._constraint_filter(
            constraint_boxes["boxes"], constraint_boxes["class_names"], constraint_boxes["scores"]
        )
        fsm, nstates = self._fsm_builder.build(candidates)

        return {"fsm": fsm, "num_states": nstates, "num_constraints": len(candidates), **item}

    def collate_fn(
        self, batch_list: List[EvaluationInstanceWithConstraints]
    ) -> EvaluationBatchWithConstraints:

        batch = super().collate_fn(batch_list)

        max_state = max([s["num_states"] for s in batch_list])
        fsm = torch.stack([s["fsm"][:max_state, :max_state, :] for s in batch_list])
        num_candidates = torch.tensor([s["num_constraints"] for s in batch_list]).long()

        batch.update({"fsm": fsm, "num_constraints": num_candidates})
        return batch


def _collate_image_features(image_features_list: List[np.ndarray]) -> np.ndarray:
    num_boxes = [instance.shape[0] for instance in image_features_list]
    image_feature_size = image_features_list[0].shape[-1]

    image_features = np.zeros(
        (len(image_features_list), max(num_boxes), image_feature_size), dtype=np.float32
    )
    for i, (instance, dim) in enumerate(zip(image_features_list, num_boxes)):
        image_features[i, :dim] = instance
    return image_features


def _custom_collate_image_features(
        detection_features_list: List[np.ndarray],
        detection_labels_list: List[List[int]],
        detection_classes_list: List[np.ndarray]) -> Tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray]:
    num_boxes = [instance.shape[0] for instance in detection_features_list]
    image_feature_size = detection_features_list[0].shape[-1]

    detection_features = np.zeros(
        (len(detection_features_list),
         max(num_boxes),
         image_feature_size),
        dtype=np.float32)

    detection_labels = np.zeros(
        (len(detection_features_list),
         max(num_boxes)),
        dtype=np.int64)

    detection_classes = np.zeros(
        (len(detection_features_list),
         max(num_boxes)),
        dtype=np.int64)

    for i, (instance_features,
            instance_labels,
            instance_classes, dim) in enumerate(zip(detection_features_list,
                                                    detection_labels_list,
                                                    detection_classes_list,
                                                    num_boxes)):
        detection_features[i, :dim] = instance_features
        detection_labels[i, :dim] = instance_labels
        detection_classes[i, :dim] = instance_classes

    return detection_features, detection_labels, detection_classes
