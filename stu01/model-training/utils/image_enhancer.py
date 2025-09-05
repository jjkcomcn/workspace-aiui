import tensorflow as tf

def enhance_image(image, augment=True):
    """图像增强处理"""
    # 归一化
    image = tf.cast(image, tf.float32) / 255.0
    
    if augment:
        # 随机水平翻转
        image = tf.image.random_flip_left_right(image)
        # 随机旋转
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        # 随机亮度调整
        image = tf.image.random_brightness(image, max_delta=0.1)
        # 随机对比度调整
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    return image
