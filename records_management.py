# -*- coding: utf-8 -*-


import os, random
import torch
import numpy as np


class TrainingEnvironment:
    def __init__(self, data_dir, save_dir, run_config, train_config):

        self.save_dir = save_dir
        self.data_dir = data_dir
        self.run_config = run_config
        self.train_config = train_config
        self.records = None
        self.metas = None
        self.ckpts = None

    def setup(self):
        self.records = Archive(
            os.path.join(self.save_dir, "records", "trainings_resnet")
        )
        self.metas = Archive(os.path.join(self.save_dir, "metas", "trainings_resnet"))
        self.ckpts = Archive(
            os.path.join(self.save_dir, "ckpts", "trainings_resnet"),
            f_name_len=6,
        )

    def to_train(self, run_config, r_training_id):

        # skip training on certain conditions
        if run_config["ignore_existing"]:
            if self.metas.has_id(r_training_id):
                print("meta data exists, will be overwritten")
            if self.ckpts.has_id(r_training_id):
                print("checkpoint exists, will be overwritten")
        else:
            to_train = False
            if not (
                self.ckpts.has_id(r_training_id) and self.metas.has_id(r_training_id)
            ):
                to_train = True
            else:
                meta = self.metas.fetch_record(r_training_id)
                if not meta["finished"]:
                    to_train = True
        if not to_train:
            print("a completed training already exists")

        return to_train


class Archive:
    def __init__(self, save_dir, r_id_len=8, f_name_len=2):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.r_id_len = r_id_len
        self.f_name_len = f_name_len

    def _record_file(self, r_id):
        return os.path.join(self.save_dir, r_id[: self.f_name_len] + ".pt")

    def _random_id(self):
        return "".join(
            ["{:X}".format(random.randrange(16)) for _ in range(self.r_id_len)]
        )

    def _new_id(self):
        r_id = None
        while True:
            r_id = self._random_id()
            if not self.has_id(r_id):
                break
        return r_id

    def _r_file_names(self):
        return [
            f
            for f in os.listdir(self.save_dir)
            if f.endswith(".pt") and len(f) == (self.f_name_len + 3)
        ]

    def _r_files(self):
        return [os.path.join(self.save_dir, f) for f in self._r_file_names()]

    def has_id(self, r_id):
        r_file = self._record_file(r_id)
        return os.path.exists(r_file) and r_id in torch.load(r_file)

    def assign(self, r_id, record):
        r_file = self._record_file(r_id)
        if os.path.exists(r_file):
            records = torch.load(r_file)
        else:
            records = {}
        records[r_id] = record
        torch.save(records, r_file)

    def update(self, r_id, record):
        r_file = self._record_file(r_id)
        if os.path.exists(r_file):
            records = torch.load(r_file)
        else:
            raise ValueError("Record not found")
        records[r_id].update(record)
        torch.save(records, r_file)

    def remove(self, r_id):
        if self.has_id(r_id):
            r_file = self._record_file(r_id)
            records = torch.load(r_file)
            records.pop(r_id)
            if records:
                torch.save(records, r_file)
            else:
                os.remove(r_file)

    def fetch_id(self, record):
        r_id, r_files = None, self._r_files()
        for r_file in r_files:
            records = torch.load(r_file)
            r_id = next((r_id for r_id, val in records.items() if val == record), None)
            if r_id is not None:
                break

        if r_id is None:
            r_id = self._new_id()
            self.assign(r_id, record)
        return r_id

    def fetch_record(self, r_id):
        if not self.has_id(r_id):
            return None
        else:
            r_file = self._record_file(r_id)
            records = torch.load(r_file)
            return records[r_id]

    def fetch_matched(self, matcher, mode="random"):
        r_files = self._r_files()

        if mode == "random":
            random.shuffle(r_files)
            for r_file in r_files:
                records = torch.load(r_file)
                matched_ids = [
                    r_id for r_id, record in records.items() if matcher(record)
                ]
                if matched_ids:
                    return random.choice(matched_ids)
            return None
        if mode == "all":
            matched_ids = []
            for r_file in r_files:
                records = torch.load(r_file)
                matched_ids += [
                    r_id for r_id, record in records.items() if matcher(record)
                ]
            return matched_ids

    def sync_from(self, master_archive):
        r_file_names = self._r_file_names()
        if self.f_name_len == master_archive.f_name_len:
            for r_file_name in r_file_names:
                source_file = os.path.join(master_archive.save_dir, r_file_name)
                if os.path.exists(source_file):
                    source = torch.load(source_file)
                else:
                    source = {}

                destination_file = os.path.join(self.save_dir, r_file_name)
                destination = torch.load(destination_file)
                r_ids = list(destination.keys())
                for r_id in r_ids:
                    if r_id in source:
                        destination[r_id] = source[r_id]
                    else:
                        destination.pop(r_id)
                if destination:
                    torch.save(destination, destination_file)
                else:
                    os.remove(destination_file)
        else:
            for r_file_name in r_file_names:
                destination_file = os.path.join(self.save_dir, r_file_name)
                destination = torch.load(destination_file)
                r_ids = list(destination.keys())
                for r_id in r_ids:
                    record = master_archive.fetch_record(r_id)
                    if record is not None:
                        destination[r_id] = record
                    else:
                        destination.pop(r_id)
                if destination:
                    torch.save(destination, destination_file)
                else:
                    os.remove(destination_file)


# return keys corresponding to all existing response data file, as a list of
# (scan, area) tuples
def valid_keys(data_dir, data_config):
    return [
        (scan, area)
        for scan in data_config["scans"]
        for area in data_config["areas"]
        if os.path.exists(
            os.path.join(data_dir, "neural_datasets", "{}_{}.pt".format(scan, area))
        )
    ]


# register images to the pool
def register_images(data_dir, image_ids, images):
    to_save = False
    pool_name = os.path.join(data_dir, "image.pool.pt")
    if os.path.exists(pool_name):
        pool = torch.load(pool_name)
    else:
        print("no image pool detected, {} created".format(pool_name))
        pool = {"ids": [], "images": []}
        to_save = True
    for image_id, image in zip(image_ids, images):
        if image_id in pool["ids"]:
            idx = pool["ids"].index(image_id)
            assert np.all(pool["images"][idx] == image)
        else:
            pool["ids"].append(image_id)
            pool["images"].append(image)
            to_save = True
    if to_save:
        torch.save(pool, pool_name)


# fetch images from the pool
def fetch_images(data_dir, image_ids):
    pool = torch.load(os.path.join(data_dir, "image.pool.pt"))
    return np.stack(
        [pool["images"][pool["ids"].index(image_id)] for image_id in image_ids]
    )
