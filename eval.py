from ssim import MSSSIM, SSIM
from data.thermal_dataset import ThermalDataset
from data.flir_dataset import FlirDataset
import torch.utils.data

class Evalulate:
    def __init__(self, opt):
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
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1, # opt.batchSize,
            shuffle=False,
            num_workers=int(opt.nThreads))
        # TODO: No flip

    def eval(self,  model, visualizer):
        # model = create_model(opt)
        # opt.no_html = True
        # opt.display_id = 0
        # create website
        # test
        # mssim_obj = MSSSIM(channel=1)
        ssim_obj = SSIM()
        mssim, ssim, i = 0, 0, 0
        for i, data in enumerate(self.dataloader):
            model.set_input(data)
            model.test()
            visualizer.add_errors(model.get_current_errors())
            #mssim = (mssim*i + mssim_obj(model.real_B, model.fake_B).item())/(i+1)
            ssim = (ssim*i + ssim_obj(model.real_B, model.fake_B).item())/(i+1)
        visualizer.append_error_hist(i, val=True)

        return ssim# , mssim

#mssim /= (i + 1)
#ssim /= (i + 1)
# print("ok,mssim:{},ssim{}".format(mssim, ssim))
