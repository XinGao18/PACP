from __future__ import annotations

from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_clip_transform(image_size: int = 224, is_train: bool = False, normalization: str = "clip") -> transforms.Compose:
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if normalization == "imagenet" else (CLIP_MEAN, CLIP_STD)
    if is_train:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        ops = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)
