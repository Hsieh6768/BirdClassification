import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Dense, Input, ZeroPadding2D
from tensorflow.keras.models import Model
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 设置随机种子以确保可重复性
tf.random.set_seed(42)
np.random.seed(42)

# 数据集路径
image_dir = r"D:\Data\02_02_Office\25夏 计算机视觉导论\archive_200\CUB_200_2011\CUB_200_2011\images"
segmentation_dir = r"D:\Data\02_02_Office\25夏 计算机视觉导论\archive_200\segmentations\segmentations"

# 图像参数
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 200


def load_and_preprocess_data(image_dir, segmentation_dir):
    """加载图像和分割掩码，准备数据集"""
    image_paths = []
    segmentation_paths = []
    labels = []

    # 获取所有类别文件夹
    class_folders = sorted([f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))])

    # 遍历每个类别文件夹
    for label, class_folder in enumerate(class_folders):
        class_image_dir = os.path.join(image_dir, class_folder)
        class_segmentation_dir = os.path.join(segmentation_dir, class_folder)

        if not os.path.exists(class_segmentation_dir):
            print(f"警告: 分割目录 {class_segmentation_dir} 不存在，跳过此类")
            continue

        # 获取图像文件
        image_files = [f for f in os.listdir(class_image_dir) if f.endswith(('.jpg'))]

        print(f"类别 {class_folder} 有 {len(image_files)} 张图像")

        for image_file in image_files:
            image_path = os.path.join(class_image_dir, image_file)

            # 构建对应的分割文件路径（分割文件与图像文件同名但扩展名不同）
            seg_file_base = os.path.splitext(image_file)[0]
            seg_file = seg_file_base + ".png"
            segmentation_path = os.path.join(class_segmentation_dir, seg_file)

            if os.path.exists(segmentation_path):
                image_paths.append(image_path)
                segmentation_paths.append(segmentation_path)
                labels.append(label)
            else:
                print(f"警告: 未找到 {seg_file} 的分割文件，跳过此图像")

    print(f"总共加载了 {len(image_paths)} 张图像和对应的分割掩码")
    return image_paths, segmentation_paths, labels, class_folders


def load_and_preprocess_image(image_path, segmentation_path, label):
    """加载和预处理图像和分割掩码"""
    # 读取图像
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0  # 归一化到[0,1]

    # 读取分割掩码
    # segmentation = tf.io.read_file(segmentation_path)
    # segmentation = tf.image.decode_image(segmentation, channels=1, expand_animations=False)
    # segmentation = tf.image.resize(segmentation, [IMG_HEIGHT, IMG_WIDTH])
    # segmentation = tf.cast(segmentation, tf.float32) / 255.0

    # 将标签转换为one-hot编码
    label = tf.one_hot(label, depth=NUM_CLASSES)

    # return (image, segmentation), label
    return image, label


# 创建TensorFlow数据集
def create_dataset(image_paths, segmentation_paths, labels, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, segmentation_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()  # 训练时重复数据集

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def identity_block(input_ten, kernel_size, filters):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1))(input_ten)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = layers.add([x, input_ten])
    x = Activation('relu')(x)
    return x


def conv_block(input_ten, kernel_size, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    x = Conv2D(filters1, (1, 1), strides=strides)(input_ten)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_ten)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


# 效仿 ResNet50 构建网络模型
def ResNetModel(nb_class, input_shape):
    input_ten = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(input_ten)

    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])

    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])

    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])

    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])

    x = AveragePooling2D((7, 7))(x)
    x = tf.keras.layers.Flatten()(x)

    output_ten = Dense(nb_class, activation='softmax')(x)
    model = Model(input_ten, output_ten)

    return model


def main():
    # 加载数据路径和标签
    image_paths, segmentation_paths, labels, class_names = load_and_preprocess_data(image_dir, segmentation_dir)

    # 划分训练集和测试集
    train_image_paths, test_image_paths, train_segmentation_paths, test_segmentation_paths, train_labels, test_labels = train_test_split(
        image_paths, segmentation_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 创建训练和测试数据集
    train_dataset = create_dataset(train_image_paths, train_segmentation_paths, train_labels, is_training=True)
    test_dataset = create_dataset(test_image_paths, test_segmentation_paths, test_labels, is_training=False)

    # 创建模型
    model = ResNetModel(NUM_CLASSES, (IMG_HEIGHT, IMG_WIDTH, 3))

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 打印模型概要
    model.summary()

    # 计算训练和验证的步骤数
    steps_per_epoch = len(train_image_paths) // BATCH_SIZE
    validation_steps = max(1, len(test_image_paths) // BATCH_SIZE)

    # 训练模型
    history = model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps
    )

    # 评估模型
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=validation_steps)
    print(f"测试准确率: {test_accuracy:.4f}")

    # 保存模型
    model.save('bird_classification_resnet.h5')
    print("模型已保存为 'bird_classification_resnet.h5'")


if __name__ == "__main__":
    main()