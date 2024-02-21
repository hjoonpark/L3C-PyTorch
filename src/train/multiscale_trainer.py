"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os, glob, collections
import time

from fjcommon import config_parser
from fjcommon import functools_ext as ft
from fjcommon import timer

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

import vis.summarizable_module
from helpers import logdir_helpers
import pytorch_ext as pe
import vis.safe_summary_writer
from blueprints.multiscale_blueprint import MultiscaleBlueprint
from dataloaders import images_loader
from helpers.global_config import global_config
from helpers.paths import CKPTS_DIR_NAME
from helpers.saver import Saver
from train import lr_schedule
from train.train_restorer import TrainRestorer
from train.trainer import LogConfig, Trainer
from test.image_saver import ImageSaver
from PIL import Image

from helpers.testset import Testset
import auto_crop
import numpy as np
# --------------------------------------------- #
# JP
pil_to_tensor = transforms.Compose([images_loader.IndexImagesDataset.to_tensor_uint8_transform()])

class TestResult(object):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.per_img_results = {}

    def __setitem__(self, filename, result):
        self.per_img_results[filename] = result.item()  # unpack Torch Tensors

    def mean(self):
        return np.mean(list(self.per_img_results.values()))
# --------------------------------------------- #
class MultiscaleTrainer(Trainer):
    def __init__(self,
                 ms_config_p, dl_config_p,
                 log_dir_root, log_config: LogConfig,
                 num_workers,
                 saver: Saver, restorer: TrainRestorer=None,
                 sw_cls=vis.safe_summary_writer.SafeSummaryWriter):
        """
        :param ms_config_p: Path to the multiscale config file, see README
        :param dl_config_p: Path to the dataloader config file, see README
        :param log_dir_root: All outputs (checkpoints, tensorboard) will be saved here.
        :param log_config: Instance of train.trainer.LogConfig, contains intervals.
        :param num_workers: Number of workers to use for DataLoading, see train.py
        :param saver: Saver instance to use.
        :param restorer: Instance of TrainRestorer, if we need to restore
        """

        # Read configs
        # config_ms = config for the network (ms = multiscale)
        # config_dl = config for data loading
        (self.config_ms, self.config_dl), rel_paths = ft.unzip(map(config_parser.parse, [ms_config_p, dl_config_p]))
        # Update config_ms depending on global_config
        global_config.update_config(self.config_ms)
        # Create data loaders
        dl_train, dl_val = self._get_dataloaders(num_workers)
        # Create blueprint. A blueprint collects the network as well as the losses in one class, for easy reuse
        # during testing.
        self.blueprint = MultiscaleBlueprint(self.config_ms)
        # Setup optimizer
        optim_cls = {'RMSprop': optim.RMSprop,
                     'Adam': optim.Adam,
                     'SGD': optim.SGD,
                     }[self.config_ms.optim]
        net = self.blueprint.net
        self.optim = optim_cls(net.parameters(), self.config_ms.lr.initial,
                               weight_decay=self.config_ms.weight_decay)
        # Calculate a rough estimate for time per batch (does not take into account that CUDA is async,
        # but good enought to get a feeling during training).
        self.time_accumulator = timer.TimeAccumulator()
        # Restore network if requested
        skip_to_itr = self.maybe_restore(restorer)
        if skip_to_itr is not None:  # i.e., we have a restorer
            print('Skipping to {}...'.format(skip_to_itr))
        # Create LR schedule to update parameters
        self.lr_schedule = lr_schedule.from_spec(
                self.config_ms.lr.schedule, self.config_ms.lr.initial, [self.optim], epoch_len=len(dl_train))

        # --- All nn.Modules are setup ---
        print('-' * 80)

        # create log dir and summary writer
        self.log_dir = Trainer.get_log_dir(log_dir_root, rel_paths, restorer)
        self.log_date = logdir_helpers.log_date_from_log_dir(self.log_dir)
        self.ckpt_dir = os.path.join(self.log_dir, CKPTS_DIR_NAME)
        print(f'Checkpoints will be saved to {self.ckpt_dir}')
        saver.set_out_dir(self.ckpt_dir)

        # Create summary writer
        sw = sw_cls(self.log_dir)
        self.summarizer = vis.summarizable_module.Summarizer(sw)
        net.register_summarizer(self.summarizer)
        self.blueprint.register_summarizer(self.summarizer)
        # superclass setup
        super(MultiscaleTrainer, self).__init__(dl_train, dl_val, [self.optim], net, sw,
                                                max_epochs=self.config_dl.max_epochs,
                                                log_config=log_config, saver=saver, skip_to_itr=skip_to_itr)

        # JP
        self.recursive = 0
        self.image_saver = ImageSaver(os.path.join(self.log_dir, "train_sampled"))
        self.flags = {"crop": None, "sample": self.log_dir, "write_to_files": None}
        # JP: validation dataset
        testset = Testset("/nobackup/joon/1_Projects/L3C-PyTorch/data/val_oi", None, None)
        self.ds_test = self.get_test_dataset(testset)

        # n_p, p_val = 0, 0
        # for name, p in self.blueprint.net.named_parameters():
        #     if p.requires_grad:
        #         d = p.data
        #         p_val += p.sum()
        #         n_p += 1
        #         # print(f"  {name}:", p.data.shape, "\n\t({:.2f}, {:.2f}) {:.2f}, {:.2f}".format(d.min().item(), d.max().item(), d.mean().item(), d.std().item()))
        # print("p_val:", p_val, ", n_p:", n_p)
        # assert 0
    def get_test_dataset(self, testset):
        to_tensor_transform = [images_loader.IndexImagesDataset.to_tensor_uint8_transform()]
        if self.flags["crop"]:
            print('*** WARN: Cropping to {}'.format(self.flags["crop"]))
            to_tensor_transform.insert(0, transforms.CenterCrop(self.flags["crop"]))
        return images_loader.IndexImagesDataset(
                images=testset,
                to_tensor_transform=transforms.Compose(to_tensor_transform))

    def modules_to_save(self):
        return {'net': self.blueprint.net,
                'optim': self.optim}

    def _get_dataloaders(self, num_workers, shuffle_train=True):
        assert self.config_dl.train_imgs_glob is not None
        print('Cropping to {}'.format(self.config_dl.crop_size))
        # JP:
        to_tensor_transform = transforms.Compose([
                transforms.RandomCrop(self.config_dl.crop_size),
                transforms.RandomHorizontalFlip(),
                images_loader.IndexImagesDataset.to_tensor_uint8_transform()])
        
        # NOTE: if there are images in your training set with dimensions <128, training will abort at some point,
        # because the cropper failes. See REAME, section about data preparation.
        min_size = self.config_dl.crop_size
        ds_train = images_loader.IndexImagesDataset(
                images=images_loader.ImagesCached(
                        self.config_dl.train_imgs_glob,
                        self.config_dl.image_cache_pkl,
                        min_size=min_size),
                to_tensor_transform=to_tensor_transform)

        dl_train = DataLoader(ds_train, self.config_dl.batchsize_train, shuffle=shuffle_train,
                              num_workers=num_workers)
        print('Created DataLoader [train] {} batches -> {} imgs'.format(
                len(dl_train), self.config_dl.batchsize_train * len(dl_train)))

        ds_val = self._get_ds_val(
                self.config_dl.val_glob,
                crop=self.config_dl.crop_size,
                truncate=self.config_dl.num_val_batches * self.config_dl.batchsize_val)
        dl_val = DataLoader(
                ds_val, self.config_dl.batchsize_val, shuffle=False,
                num_workers=num_workers, drop_last=True)
        print('Created DataLoader [val] {} batches -> {} imgs'.format(
                len(dl_val), self.config_dl.batchsize_train * len(dl_val)))

        return dl_train, dl_val

    def _get_ds_val(self, images_spec, crop=False, truncate=False):
        img_to_tensor_t = [images_loader.IndexImagesDataset.to_tensor_uint8_transform()]
        if crop:
            img_to_tensor_t.insert(0, transforms.CenterCrop(crop))
        img_to_tensor_t = transforms.Compose(img_to_tensor_t)

        fixed_first = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixedimg.jpg')
        if not os.path.isfile(fixed_first):
            print(f'INFO: No file found at {fixed_first}')
            fixed_first = None

        ds = images_loader.IndexImagesDataset(
                images=images_loader.ImagesCached(
                        images_spec, self.config_dl.image_cache_pkl,
                        min_size=self.config_dl.val_glob_min_size),
                to_tensor_transform=img_to_tensor_t,
                fixed_first=fixed_first)  # fix a first image to have consistency in tensor board

        if truncate:
            ds = pe.TruncatedDataset(ds, num_elemens=truncate)

        return ds

    def train_step(self, i, batch, log, log_heavy, load_time=None):
        """
        :param i: current step
        :param batch: dict with 'idx', 'raw'
        """
        self.lr_schedule.update(i)
        self.net.zero_grad()

        values = Values('{:.3e}', ' | ')

        if i % 100 == 0:
            print("iter {}".format(i))

        with self.time_accumulator.execute():
            idxs, img_batch, s = self.blueprint.unpack(batch)

            with self.summarizer.maybe_enable(prefix='train', flag=log, global_step=i):
                out = self.blueprint.forward(img_batch)

            with self.summarizer.maybe_enable(prefix='train', flag=log_heavy, global_step=i):
                loss_out = self.blueprint.get_loss(out)
                loss_pc = loss_out.loss_pc
                nonrecursive_bpsps = loss_out.nonrecursive_bpsps

            total_loss = loss_pc
            total_loss.backward()
            self.optim.step()

            values['loss'] = loss_pc
            values['bpsp'] = sum(nonrecursive_bpsps)

            if i % 100 == 0:
                with open("output_plots/losses.txt", "a+") as f:
                    f.write("step={} loss={} bpsp={}\n".format(i, values.values['loss'], values.values['bpsp']))

            # SAMPLE training data crop (3, 128, 128)
            self.blueprint.set_eval()
            if i % 1000 == 0:
                sample = 1
                img_batch = img_batch[0].unsqueeze(0)
                if sample:
                    # -----------------------------------------------                
                    # JP: save ground truth
                    save_folder = "{:06d}".format(i)
                    sample_save_dir = os.path.join(self.image_saver.out_dir, save_folder)
                    os.makedirs(sample_save_dir, exist_ok=True)
                    print(">> saving sampled to: {}".format(sample_save_dir))

                    # JP: save ground truth
                    self.image_saver.save_img(img_batch, os.path.join(save_folder, '{:06d}_gt.png'.format(i)))

                    # sample crop
                    for style, sample_scales in (('rgb', []),               # Sample RGB scale (final scale)
                                                # ('rgb+bn0', [0]),          # Sample RGB + z^(1)
                                                # ('rgb+bn0+bn1', [0, 1])
                                                ):  # Sample RGB + z^(1) + z^(2)
                        sampled = self.blueprint.sample_forward(img_batch, sample_scales, name_prefix=f"crop_tr_{i}")
                        self.image_saver.save_img(sampled, os.path.join(save_folder, '{:06d}_{}.png'.format(i, style)))

                    # SAMPLE a full test image
                    with torch.no_grad():
                        result = self._test(self.ds_test)

                    # # -----------------------------------------------                
                    # # SAMPLE test data (3, full w, full h)
                    # val_path = glob.glob("../data/val_oi/*.png")[0]
                    # raw_img_uint8_crop = pil_to_tensor(Image.open(val_path)).unsqueeze(0)
                    # img, _ = self.blueprint.unpack_batch_pad(raw_img_uint8_crop, fac=8)
                    # self.image_saver.save_img(img, os.path.join(save_folder, 'val_{:06d}_gt.png'.format(i)))

                    # # sample
                    # for style, sample_scales in (('rgb', []),               # Sample RGB scale (final scale)
                    #                             # ('rgb+bn0', [0]),          # Sample RGB + z^(1)
                    #                             # ('rgb+bn0+bn1', [0, 1])
                    #                             ):  # Sample RGB + z^(1) + z^(2)
                    #     self.blueprint.net.training = False
                    #     sampled = self.blueprint.sample_forward(img, sample_scales, name_prefix=f"full_val_{i}")
                    #     self.blueprint.net.training = True
                    #     self.image_saver.save_img(sampled, os.path.join(save_folder, 'val_{:06d}_{}.png'.format(i, style)))
            self.blueprint.set_train()

        if not log:
            return

        mean_time_per_batch = self.time_accumulator.mean_time_spent()
        imgs_per_second = self.config_dl.batchsize_train / mean_time_per_batch

        print('{} {: 6d}: {} // {:.3f} img/s '.format(
                self.log_date, i, values.get_str(), imgs_per_second) + (load_time or ''))

        values.write(self.sw, i)

        # Gradients
        params = [('all', self.net.parameters())]
        for name, ps in params:
            tot = pe.get_total_grad_norm(ps)
            self.sw.add_scalar('grads/{}/total'.format(name), tot, i)

        # log LR
        lrs = list(self.get_lrs())
        assert len(lrs) == 1
        self.sw.add_scalar('train/lr', lrs[0], i)

        if not log_heavy:
            return

        self.blueprint.add_image_summaries(self.sw, out, i, 'train')

    def validation_loop(self, i):
        bs = pe.BatchSummarizer(self.sw, i)
        val_start = time.time()
        for j, batch in enumerate(self.dl_val):
            idxs, img_batch, s = self.blueprint.unpack(batch)

            # Only log TB summaries for first batch
            with self.summarizer.maybe_enable(prefix='val', flag=j == 0, global_step=i):
                out = self.blueprint.forward(img_batch)
                loss_out = self.blueprint.get_loss(out)

            bs.append('val/bpsp', sum(loss_out.nonrecursive_bpsps))

            if j > 0:
                continue

            self.blueprint.add_image_summaries(self.sw, out, i, 'val')

        val_duration = time.time() - val_start
        num_imgs = len(self.dl_val.dataset)
        time_per_img = val_duration/num_imgs

        output_strs = bs.output_summaries()
        output_strs = ['{: 6d}'.format(i)] + output_strs + ['({:.3f} s/img)'.format(time_per_img)]
        output_str = ' | '.join(output_strs)
        sep = '-' * len(output_str)
        print('\n'.join([sep, output_str, sep]))

    def _test(self, ds):
        print("-"*10, "self._test")

        metric_name = 'bpsp recursive' if self.recursive else 'bpsp'
        test_result = TestResult(metric_name)

        # If we sample, we store the result with a ImageSaver.
        if self.flags["sample"]:
            image_saver = ImageSaver(os.path.join(self.flags["sample"], "sampled_testset"))
            print('  Will store samples in {}.'.format(image_saver.out_dir))
        else:
            image_saver = None

        log = ''
        one_line_output = not self.flags["sample"]
        number_of_crops = collections.defaultdict(int)

        for i, img in enumerate(ds):
            filename = os.path.splitext(os.path.basename(ds.files[img['idx']]))[0]

            # Use arithmetic coding to write to real files, and save time statistics.
            assert self.flags["write_to_files"] is None
            if self.flags["write_to_files"]:
                print('  ***', filename)
                img_raw = img["raw"].long().unsqueeze(0).to(pe.DEVICE)  # 1CHW
                with self.times.skip(i == 0):
                    out_dir = self.flags["write_to_files"]
                    os.makedirs(out_dir, exist_ok=True)
                    out_p = os.path.join(out_dir, filename + _FILE_EXT)
                    # As a side effect, create a time report if --time_report given.
                    bpsp = self._write_to_file(img_raw, out_p)
                    test_result[filename] = bpsp
                    log = (f'{self.log_date}: {filename} ({i: 10d}): '
                           f'mean {test_result.metric_name}={test_result.mean()}')
                    print(log)
                    continue

            raw_img_uint8 = img["raw"].unsqueeze(0)  # Full resolution image, !CHW

            # Make sure we count bpsp of different crops of an image correctly.
            combinator = auto_crop.CropLossCombinator()

            num_crops_img = 0

            for raw_img_uint8_crop in auto_crop.iter_crops(raw_img_uint8):
                # We have to pad images not divisible by (2 ** num_scales), because we downsample num_scales-times.
                # To get the correct bpsp, we have to use, num_subpixels_before_pad,
                #   see `get_loss` in multiscale_blueprint.py
                num_subpixels_before_pad = np.prod(raw_img_uint8_crop.shape)
                img_batch, _ = self.blueprint.unpack_batch_pad(raw_img_uint8_crop, fac=self._padding_fac())

                out = self.blueprint.forward(img_batch, self.recursive)
                
                loss_out: MultiscaleLoss = self.blueprint.get_loss(
                    out, num_subpixels_before_pad=num_subpixels_before_pad)

                if self.flags["sample"]:  # TODO not tested with multiple crops
                    self._sample(loss_out.nonrecursive_bpsps, img_batch, image_saver, '{}_{}'.format(i, filename))

                if self.recursive:
                    bpsp = sum(loss_out.recursive_bpsps)
                else:
                    bpsp = sum(loss_out.nonrecursive_bpsps)

                combinator.add(bpsp.item(), num_subpixels_before_pad)
                num_crops_img += 1

            number_of_crops[num_crops_img] += 1
            test_result[filename] = combinator.get_bpsp()

            log = f'{self.log_date}: {filename} ({i: 10d}): mean {test_result.metric_name}={test_result.mean()}'
            number_of_crops_str = '|'.join(
                f'{count}:{freq}' for count, freq in sorted(number_of_crops.items(), reverse=True))
            log += ' crops:freq -> ' + number_of_crops_str
        if self.flags["write_to_files"]:
            return None
        return test_result

    def _sample(self, bpsps, img_batch, image_saver, save_prefix):
        # Make sure folder does not already contain samples for this file.
        # if image_saver.file_starting_with_exists(save_prefix):
        #     raise FileExistsError('  Previous sample outputs found in {}. Please remove.'.format(
        #             image_saver.out_dir))
        # Store ground truth for comparison
        image_saver.save_img(img_batch, '{}_{:.3f}_gt.png'.format(save_prefix, sum(bpsps)))
        for style, sample_scales in (('rgb', []),               # Sample RGB scale (final scale)
                                    #  ('rgb+bn0', [0]),          # Sample RGB + z^(1)
                                    #  ('rgb+bn0+bn1', [0, 1])
                                     ):  # Sample RGB + z^(1) + z^(2)
            sampled = self.blueprint.sample_forward(img_batch, sample_scales, name_prefix="test")
            bpsp_sample = sum(bpsps[len(sample_scales) + 1:])
            image_saver.save_img(sampled, '{}_{}_{:.3f}.png'.format(save_prefix, style, bpsp_sample))

    def _padding_fac(self):
        if self.recursive:
            return 2 ** (self.recursive + 1)
        return 2 ** self.config_ms.num_scales

class Values(object):
    """
    Stores values during one training step. Essentially a thin wrapper around dict with support to get a nicely
    formatted string and write to a SummaryWriter.
    """
    def __init__(self, fmt_str='{:.3f}', joiner=' / ', prefix='train/'):
        self.fmt_str = fmt_str
        self.joiner = joiner
        self.prefix = prefix
        self.values = {}

    def __setitem__(self, key, value):
        self.values[key] = value

    def get_str(self):
        """ :return pretty printed version of all values, using default_fmt_str """
        return self.joiner.join('{} {}'.format(k, self.fmt_str.format(v))
                                for k, v in sorted(self.values.items()))

    def write(self, sw, i):
        """ Writes to summary writer `sw`. """
        for k, v in self.values.items():
            sw.add_scalar(self.prefix + k, v, i)
