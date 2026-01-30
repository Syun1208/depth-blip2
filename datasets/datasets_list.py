import os
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils import data
from torchvision import transforms

from utils.transform_list import EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensorNumpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def save_tensor_image(tensor, path, is_depth=False):
    import numpy as np
    from PIL import Image
    import torch

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

    arr = tensor.numpy()

    # -------- FIX DIMENSIONS --------
    if arr.ndim == 3:

        # if CHW → convert to HWC
        if arr.shape[0] in [1, 3]:
            arr = np.transpose(arr, (1, 2, 0))

        # squeeze grayscale channel
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]

    # -------- DEPTH VISUALIZATION --------
    if is_depth:
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    # convert to uint8
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(path)


class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                # RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1))],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])
            self.test_transform = EnhancedCompose([
                # CropNumpy((args.height, args.width)),
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])
        elif args.dataset == 'NYU':
            self.train_transform = EnhancedCompose([
                # RandomCropNumpy((args.height, args.width)),
                # RandomHorizontalFlip(),
                # [RandomColor(multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25))],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])

    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)

def has_no_header(df: pd.DataFrame) -> bool:
    """Check if DataFrame columns are auto-generated integers."""
    return list(df.columns) == list(range(len(df.columns)))


def sample_unique_classes(df, n, class_col="class", random_state=None):
    unique_classes = df[class_col].drop_duplicates().sample(
        n=min(n, df[class_col].nunique()),
        random_state=random_state
    )

    result = (
        df[df[class_col].isin(unique_classes)]
        .groupby(class_col, group_keys=False)
        .sample(1, random_state=random_state)
        .reset_index(drop=True)
    )

    return result

def extract_class(s):
  l = s.split('/')[-2].split('_')
  n_remove = 2
  for i in range(n_remove):
    l.pop()
  return "_".join(l)

class NYUDataset(data.Dataset):
    """NYU Depth V2 dataloader"""

    def __init__(
        self,
        args,
        vis_processors,
        train=True,
        val=False,
        return_filename=False,
    ):
        self.args = args
        self.train = train
        self.return_filename = return_filename

        self.blip2_image_processor = vis_processors["eval"]
        self.depth_scale = 1000.0
        self.transform = Transformer(args)

        # --------------------------------------------------
        # Select CSV file
        # --------------------------------------------------
        if train and val:
            file_path = self.args.val_file
        elif train:
            file_path = self.args.train_file
        else:
            file_path = self.args.test_file

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset CSV not found: {file_path}")

        # --------------------------------------------------
        # Load CSV
        # --------------------------------------------------
        self.fileset = pd.read_csv(file_path)

        # if has_no_header(self.fileset):
        self.fileset.columns = ["rgb", "depth"]
        self.fileset['class'] = self.fileset['rgb'].apply(extract_class)

        print(self.fileset.head())
        # --------------------------------------------------
        # Apply limits
        # --------------------------------------------------
        if train and hasattr(args, "train_limit") and args.train_limit != -1:
            # self.fileset = self.fileset.iloc[: args.train_limit]
            self.fileset = sample_unique_classes(self.fileset, args.train_limit)

        if val and hasattr(args, "val_limit") and args.val_limit != -1:
            # self.fileset = self.fileset.iloc[: args.val_limit]
            self.fileset = sample_unique_classes(self.fileset, args.val_limit)

        # --------------------------------------------------
        # Data root
        # --------------------------------------------------
        self.data_root_path = self.args.data_root_path


    def __len__(self):
        return len(self.fileset)

    def __getitem__(self, idx):
        row = self.fileset.iloc[idx]

        rgb_path = row["rgb"]
        depth_path = row["depth"]

        filename = os.path.splitext(rgb_path)[0]

        # --------------------------------------------------
        # Load RGB image
        # --------------------------------------------------
        rgb_img_file = os.path.join(self.data_root_path, rgb_path)
        if not os.path.isfile(rgb_img_file):
            raise FileNotFoundError(f"RGB image not found: {rgb_img_file}")
        # Load RGB
        input_rgb_img = Image.open(rgb_img_file).convert("RGB")
        input_rgb_img = transforms.Resize((224, 224))(input_rgb_img)
        input_rgb_img = transforms.ToTensor()(input_rgb_img)
        input_rgb_img = input_rgb_img.permute(2, 1, 0)

        # Convert to numpy float32 in HWC
        preprocess_rgb_img = np.asarray(input_rgb_img, dtype=np.float32) / 255.0

        # IMPORTANT — transform will convert to CHW internally
        preprocess_rgb_img = self.transform(
            [preprocess_rgb_img], train=self.train
        )[0]

        # --------------------------------------------------
        # Load depth map
        # --------------------------------------------------
        if self.args.dataset != "NYU":
            raise NotImplementedError("Only NYU dataset is supported.")

        gt_file = os.path.join(self.data_root_path, depth_path)
        if not os.path.isfile(gt_file):
            raise FileNotFoundError(f"Depth image not found: {gt_file}")

        input_gt_img = Image.open(gt_file)

        if not _is_pil_image(input_gt_img):
            raise ValueError(f"Invalid depth image: {gt_file}")

        input_gt_img = np.asarray(input_gt_img, dtype=np.float32)
        input_gt_img = np.expand_dims(
            input_gt_img / self.depth_scale, axis=2
        )
        input_gt_img = np.clip(
            input_gt_img, 0, self.args.max_depth
        )

        preprocess_gt_img = torch.from_numpy(
            input_gt_img.transpose((2, 0, 1))
        )

        # --------------------------------------------------
        # Return
        # --------------------------------------------------
        # debug_dir = "./debug_outputs"
        # os.makedirs(debug_dir, exist_ok=True)

        # save_tensor_image(
        #     input_rgb_img,
        #     os.path.join(debug_dir, f"{idx}_input_rgb.png")
        # )

        # save_tensor_image(
        #     preprocess_rgb_img,
        #     os.path.join(debug_dir, f"{idx}_preprocess_rgb.png")
        # )

        # save_tensor_image(
        #     preprocess_gt_img,
        #     os.path.join(debug_dir, f"{idx}_depth.png"),
        #     is_depth=True
        # )
        return (
            input_rgb_img,
            preprocess_rgb_img,
            preprocess_gt_img,
            filename,
        )


class NYUDatasetTXT(data.Dataset):
    """NYU Depth V2 dataloader for txt file"""

    def __init__(self, args, vis_processors, train=True, val=False, return_filename=False):
        self.args = args

        # read dataset mapping relation
        self.blip2_image_processor = vis_processors["eval"]
        self.depth_scale = 1000.0
        try:
            file_path = self.args.val_file if train and val else self.args.train_file if train else self.args.test_file
            with open(file_path, 'r') as f:
                fileset = f.readlines()
            fileset = sorted(fileset)
            self.fileset = [file for file in fileset
                            if file.split()[0].rsplit('/', 1)[0] == self.args.class_name
                            or self.args.class_name == 'all']
            self.data_root_path = os.path.join(self.args.data_root_path, 'train') if train else os.path.join(
                self.args.data_root_path, 'test')
        except FileNotFoundError as e:
            print(e.__context__)

        self.train = train
        self.transform = Transformer(args)
        self.return_filename = return_filename

    def __getitem__(self, idx):
        divided_file = self.fileset[idx].split()
        # 1-Opening image files.
        # rgb: input color image, gt: sparse depth map
        # rgb: range:(0, 1),  depth range: (0, max_depth)
        class_name, filename = divided_file[0].rsplit('/', 1)
        filename = filename.rsplit('.', 1)[0]
        # 1.1-load rgb and process rgb
        rgb_img_file = ''.join([self.data_root_path, '/', divided_file[0]])
        input_rgb_img = Image.open(rgb_img_file)
        input_rgb_img_crop = input_rgb_img.crop((40 + 20, 42 + 14, 616 - 12, 474 - 2))
        input_rgb_img = np.asarray(input_rgb_img, dtype=np.int32)
        input_rgb_img_crop = transforms.Resize((224, 224))(input_rgb_img_crop)
        preprocess_rgb_img = np.asarray(input_rgb_img_crop, dtype=np.float32) / 255.0
        preprocess_rgb_img = self.transform([preprocess_rgb_img], train=self.train)[0]
        # preprocess_rgb_img = preprocess_rgb_img.permute(2, 0, 1)
        # 1.2-load gt
        if self.args.dataset == 'NYU':
            gt_file = ''.join([self.data_root_path, '/', divided_file[1]])
            input_gt_img = Image.open(gt_file)
        else:
            print('other dataset is not supported now!')
            exit()

        # 1.3-process depth map
        if _is_pil_image(input_gt_img):
            # process gt
            # input_gt_img = input_gt_img.crop((40 + 20, 42 + 14, 616 - 12, 474 - 2))
            # input_gt_img = transforms.Resize((self.args.height, self.args.width))(input_gt_img)
            input_gt_img = np.expand_dims(np.asarray(input_gt_img, dtype=np.float32) / self.depth_scale, 2)
            input_gt_img = np.clip(input_gt_img, 0, self.args.max_depth)
            preprocess_gt_img = torch.from_numpy(input_gt_img.transpose((2, 0, 1)))
        else:
            print('location: {} is not legal image!'.format(gt_file))
            exit()

        # 2-return result
        return input_rgb_img, preprocess_rgb_img, preprocess_gt_img, filename

    def __len__(self):
        return len(self.fileset)


class KITTIDataset(data.Dataset):
    """KITTI dataloader"""

    def __init__(self, args, vis_processors, train=True, val=False, return_filename=False):
        self.args = args

        # read dataset mapping relation
        self.blip2_image_processor = vis_processors["eval"]
        self.depth_scale = 256.0
        try:
            file_path = self.args.val_file if train and val else self.args.train_file if train else self.args.test_file
            with open(file_path, 'r') as f:
                fileset = f.readlines()
            fileset = sorted(fileset)
            self.fileset = [file for file in fileset
                            if file.split()[-1] != 'None']
            self.data_root_path = os.path.join(self.args.data_root_path, 'train') if train else os.path.join(
                self.args.data_root_path, 'test')
        except FileNotFoundError as e:
            print(e.__context__)

        self.train = train
        self.transform = Transformer(args)
        self.return_filename = return_filename

    def __getitem__(self, idx):
        divided_file = self.fileset[idx].split()
        # 1-Opening image files.
        # rgb: input color image, gt: sparse depth map
        # rgb: range:(0, 1),  depth range: (0, max_depth)
        _, filename = divided_file[0].rsplit('/', 1)
        filename = filename.rsplit('.', 1)[0]
        # 1.1-load rgb and process rgb
        rgb_img_file = ''.join([self.data_root_path, '/', divided_file[0]])
        input_rgb_img = Image.open(rgb_img_file)
        # using bounding
        bound_left = (input_rgb_img.width - 1216) // 2
        bound_right = bound_left + 1216
        bound_top = input_rgb_img.height - 352
        bound_bottom = bound_top + 352
        input_rgb_img_crop = input_rgb_img.crop((bound_left, bound_top, bound_right, bound_bottom))
        input_rgb_img = transforms.Compose([
            transforms.Resize((375, 1242)),
        ])(input_rgb_img)
        input_rgb_img = np.asarray(input_rgb_img, dtype=np.int32)
        input_rgb_img_crop = transforms.Resize((224, 224))(input_rgb_img_crop)
        preprocess_rgb_img = np.asarray(input_rgb_img_crop, dtype=np.float32) / 255.0
        preprocess_rgb_img = self.transform([preprocess_rgb_img], train=self.train)[0]
        # preprocess_rgb_img = preprocess_rgb_img.permute(2, 0, 1)
        # 1.2-load gt
        gt_file = ''.join([self.data_root_path, '/', divided_file[1]])
        input_gt_img = Image.open(gt_file)

        # 1.3-process depth map
        if _is_pil_image(input_gt_img):
            # process gt
            # input_gt_img = transforms.Resize((self.args.height, self.args.width))(input_gt_img)
            # input_gt_img = input_gt_img.crop((bound_left, bound_top, bound_right, bound_bottom))
            input_gt_img = transforms.Compose([
                transforms.Resize((375, 1242)),
            ])(input_gt_img)
            input_gt_img = np.expand_dims(np.asarray(input_gt_img, dtype=np.float32) / self.depth_scale, 2)
            input_gt_img = np.clip(input_gt_img, 0, self.args.max_depth)
            preprocess_gt_img = torch.from_numpy(input_gt_img.transpose((2, 0, 1)))
        else:
            print('location: {} is not legal image!'.format(gt_file))
            exit()

        # 2-return result
        return input_rgb_img, preprocess_rgb_img, preprocess_gt_img, filename

    def __len__(self):
        return len(self.fileset)


class AllDataset(data.Dataset):
    """dataloader: supporting NYU Depth V2, KITTI"""

    def __init__(self, args, vis_processors, train=True, val=False, return_filename=False):
        if args.dataset == 'NYU':
            self.dataset = NYUDataset(args, vis_processors, train=train, val=val, return_filename=return_filename)
        elif args.dataset == 'KITTI':
            self.dataset = KITTIDataset(args, vis_processors, train=train, val=val, return_filename=return_filename)
        else:
            print('other dataset is not supported now!')
            exit()

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
