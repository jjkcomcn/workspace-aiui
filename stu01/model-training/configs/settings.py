# 模型训练项目配置

# 数据路径配置
DATA_DIR = "data/"
INPUT_DIR = f"{DATA_DIR}input/"
OUTPUT_DIR = f"{DATA_DIR}output/"
PROCESSED_DIR = f"{DATA_DIR}processed/"

# 模型训练参数
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
IMG_SIZE = (224, 224)  # 图像输入尺寸

# 日志配置
LOG_DIR = "logs/"
LOG_LEVEL = "INFO"

# 模型保存配置
MODEL_SAVE_DIR = "models/saved/"
MODEL_SAVE_FORMAT = "h5"
