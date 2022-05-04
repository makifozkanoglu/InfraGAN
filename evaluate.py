import os
from options.test_options import TestOptions
from models.models import create_model
from ssim import MSSSIM, SSIM
import torch
from lpips.lpips import LPIPS
from util import util
from data.thermal_dataset import ThermalDataset
from data.flir_dataset import FlirDataset
import tensorflow as tf
from tqdm import tqdm

class AverageMeter(object):
    def __init__(self):
        self.n = 0
        self.val = 0
        self.total = 0
        self.avg = 0

    def update(self, val, n):
        self.total += val*n
        self.n += n
        self.avg = self.total / self.n

vedai_class = {
    1: "car",
    2: "truck",
    11: "pickup",
    4: "tractor",
    5: "campingcar",
    23: "boat",
    7: "motorcycle",
    8: "bus",
    9: "van",
    10: "other",
    #11: "smallcar",
    #12: "largecar",
    31: "largevehicles",
    #23: "board"
    # unique array([ 1,  2,  4,  5,  7,  8,  9, 10, 11, 23, 31])
}
kaist_class = {
    1:"cyclist",
    2:"people",
    3:"person",
    4:"other"
}

opt = TestOptions().parse()
opt.nThreads = 0  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


mode="test"
print(opt.dataset_mode)
if opt.dataset_mode == 'VEDAI':
    dataset = ThermalDataset()
    dataset.initialize(opt, mode="test")
elif opt.dataset_mode == 'KAIST':
    dataset = ThermalDataset()
    dataset.initialize(opt, mode=mode)
elif opt.dataset_mode == 'FLIR':
    dataset = FlirDataset()
    dataset.initialize(opt, test=True)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.nThreads))
model = create_model(opt)
# opt.no_html = True
# opt.display_id = 0
# create website
# test
mssim_obj = MSSSIM(channel=1)
ssim_obj = SSIM()
lpips_obj = LPIPS()
L1_obj = torch.nn.L1Loss()
metrics = {}
visualize = False
class_agn = False
class_agn = False if opt.dataset_mode == 'FLIR' else class_agn
dataset_class = vedai_class if opt.dataset_mode == 'VEDAI' else kaist_class

for k in dataset_class.values():
    metrics[k] = {
        "ssim": AverageMeter(),
        "mssim": AverageMeter(),
        "lpips": AverageMeter(),
        #"mi": AverageMeter(),
        "l1": AverageMeter(),
        "psnr": AverageMeter()
    }
overall_metric = {
    "ssim": AverageMeter(),
    "mssim": AverageMeter(),
    "lpips": AverageMeter(),
    #"mi": AverageMeter(),
    "l1": AverageMeter(),
    "psnr": AverageMeter()
}

t = 0.
for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader))):

    model.set_input(data)
    batch_size = model.input_A.shape[0]
    t += model.test()

    cls = []
    if class_agn and os.path.exists(data['annotation_file'][0]):
        with open(data['annotation_file'][0]) as f:
            if opt.dataset_mode == 'VEDAI':
                cls = [int(line.split(" ")[3]) for line in f.read().split("\n") if len(line.split(" ")) > 3]
            elif opt.dataset_mode == 'KAIST':
                lines = f.read().split("\n")
                for line in lines:
                    for k in ['cyclist', 'people', 'person']:
                        if k in line.split(" ")[0]:
                            if k=="cyclist":
                                cls += [1]
                            elif k == "people":
                                cls += [2]
                            elif k == "person":
                                cls += [3]
                        else:
                            cls += [4]

    l1 = L1_obj(model.real_B, model.fake_B).item()
    mssim = mssim_obj(model.real_B, model.fake_B).item()
    ssim = ssim_obj(model.real_B, model.fake_B).item()
    lpips = lpips_obj(model.real_B.cpu(), model.fake_B.cpu()).mean().item()
    psnr = tf.image.psnr(tf.convert_to_tensor(model.real_B.cpu().numpy()),
                         tf.convert_to_tensor(model.fake_B.cpu().numpy()), 2).numpy()
    if visualize and i in [0]:# False and (ssim > 0.4 or i == 179):
        image_dir = 'samples'
        visuals = model.get_current_visuals()
        for label, im in visuals.items():
            image_name = '{}_{}.png' .format(i, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            util.save_image(im, save_path)
    if class_agn:
        for k in dataset_class.keys():
            if k in cls:
                metrics[dataset_class[k]]['l1'].update(l1, batch_size)
                metrics[dataset_class[k]]['mssim'].update(mssim, batch_size)
                metrics[dataset_class[k]]['ssim'].update(ssim, batch_size)
                metrics[dataset_class[k]]['lpips'].update(lpips, batch_size)
                metrics[dataset_class[k]]['psnr'].update(psnr, batch_size)

    overall_metric['l1'].update(l1, batch_size)
    overall_metric['mssim'].update(mssim, batch_size)
    overall_metric['ssim'].update(ssim, batch_size)
    overall_metric['lpips'].update(lpips, batch_size)
    overall_metric['psnr'].update(psnr, batch_size)

print("fps:", (i+1)/t)

if class_agn:
    for k in dataset_class.keys():
        print("\n{}------>".format(dataset_class[k]))
        for m in metrics[dataset_class[k]].keys():
            print("{}={},".format(m, metrics[dataset_class[k]][m].avg), end="")

print("\nOverall results------>")
for m in overall_metric.keys():
    print("{}={},".format(m, overall_metric[m].avg), end="")
print("")
