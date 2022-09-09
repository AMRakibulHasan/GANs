from dataset.CelebA import CelebAData
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class DL:
    def __init__(self, args):
        data = None
        if 'CelebA' in args.data_path:
            data = CelebAData(args.data_path, args.img_size)

        self.sampler = DistributedSampler(data, shuffle=True)

        self.dl = DataLoader(data,
                             shuffle=False,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             sampler=self.sampler,
                             drop_last=True,
                             )

