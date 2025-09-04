import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

def base_preproc(img_size=224):
    return A.Compose([
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2()
    ])

def anomaly_bank(severity=3):
    s = max(1, min(5, severity))
    return A.OneOf([
        A.MotionBlur(blur_limit=(3, 3 + 2*s), p=0.5),
        A.GaussianBlur(blur_limit=(3, 3 + 2*s), p=0.5),
        A.ISONoise(intensity=(0.02*s, 0.04*s), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1*s/5,
                                   contrast_limit=0.1*s/5, p=0.7),
        A.RandomFog(fog_coef_lower=0.02*s/10, fog_coef_upper=0.05*s/10, p=0.5),
        A.RandomRain(blur_value=3 + s, brightness_coefficient=1.0 - 0.05*s, p=0.5),
        A.ImageCompression(quality_lower=90-10*s, quality_upper=95-10*s, p=0.6),
        A.CoarseDropout(max_holes=1+s, max_height=int(0.1*s*224),
                        max_width=int(0.1*s*224), fill_value=0, p=0.7),
        A.RandomSnow(brightness_coeff=1.0, snow_point_lower=0.1*s/10,
                     snow_point_upper=0.2*s/10, p=0.3),
        A.Defocus(radius=(1, 1+s), alias_blur=(0.1, 0.2*s), p=0.4),
    ], p=1.0)

def make_pipeline(img_size=224, apply_anomaly=False, severity=3):
    if apply_anomaly:
        return A.Compose([
            A.LongestMaxSize(img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0),
            anomaly_bank(severity),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2()
        ])
    return base_preproc(img_size)
