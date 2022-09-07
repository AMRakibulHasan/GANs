from torchvision.transforms import transforms as tf


class Transform:
    def __init__(self, img_size):
        self.transform = tf.Compose([
            tf.Resize((img_size, img_size)),
            tf.CenterCrop(img_size),
            tf.ToTensor(),
            tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, img):
        return self.transform(img)

