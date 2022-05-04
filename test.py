import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import torch
from data.thermal_dataset import ThermalDataset
from data.flir_dataset import FlirDataset

#import pydevd_pycharm
#pydevd_pycharm.settrace('10.201.182.31', port=2525, stdoutToServer=True, stderrToServer=True)


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

mode="test"
if opt.dataset_mode == 'VEDAI':
    dataset = ThermalDataset()
    dataset.initialize(opt, mode="test")
elif opt.dataset_mode == 'KAIST':
    dataset = ThermalDataset()
    # mode = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/test-all-20.txt'
    dataset.initialize(opt, mode=mode)
elif opt.dataset_mode == 'FLIR':
    dataset = FlirDataset()
    dataset.initialize(opt, test=True)
elif opt.dataset_mode == 'BUTR':
    dataset = ThermalDataset()
    # mode = 'test'
    dataset.initialize(opt, mode=mode)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1, # opt.batchSize,
    shuffle=False,
    num_workers=int(opt.nThreads))
model = create_model(opt)
#opt.no_html = True
#opt.display_id = 0
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()
