import tensorflow_datasets as tfds
import tensorflow as tf
import os

# --- 配置 ---
# 要下载的数据集官方名称列表
# 注意：在tfds中，Open-X数据集通常需要 'oxe/' 前缀
DATASETS_TO_DOWNLOAD = {
    'Berkeley Bridge': 'oxe/bridge_data',
    'Language Table': 'oxe/language_table',
    'Maniskill': 'oxe/maniskill_dataset'
}

# 您想下载的每个数据集的样本数量
NUM_SAMPLES_PER_DATASET = 1000

# 数据将被下载到这个目录下，以保持项目整洁
DOWNLOAD_DIR = 'dataset/tfds_data'

# --- 脚本 ---

def download_and_verify_subset(friendly_name: str, tfds_name: str, num_samples: int, data_dir: str):
    """
    下载并验证指定数据集的子集。
    """
    print(f"\n{'='*50}")
    print(f"准备下载数据集: '{friendly_name}' (TFDS name: '{tfds_name}')")
    print(f"目标样本数: {num_samples}")
    print(f"下载目录: {data_dir}")
    print(f"{'='*50}")

    try:
        # 定义要加载的数据分割和数量，例如 'train[:1000]'
        split_selection = f'train[:{num_samples}]'

        # 使用 tfds.load 进行流式加载和下载
        # TFDS 会在后台智能处理，只下载和准备所需的最少文件
        ds = tfds.load(
            tfds_name,
            split=split_selection,
            data_dir=data_dir,
            try_gcs=True,  # 尝试从 Google Cloud Storage 直接访问
        )

        print(f"\n🎉 成功！'{friendly_name}' 的数据加载器已准备就绪。")
        
        # --- 验证下载的数据 ---
        print("正在验证数据...")
        episode_count = 0
        for episode in ds.take(1):  # 只取一个样本进行快速验证
            episode_count += 1
            steps = episode['steps']
            first_step = next(iter(steps))
            
            # 尝试获取语言指令，如果存在的话
            if 'language_instruction' in first_step:
                instruction = first_step['language_instruction'].numpy().decode('utf-8')
                print(f"  - 样本轨迹的语言指令: '{instruction}'")
            else:
                print("  - 样本轨迹没有语言指令。")
            
            num_steps = tf.data.experimental.cardinality(steps)
            print(f"  - 样本轨迹的长度: {num_steps} 步")

        if episode_count == 0:
            print("  - 警告: 验证时未能从数据加载器中获取任何样本。")
            
    except Exception as e:
        print(f"\n❌ 下载或处理 '{friendly_name}' 时发生严重错误: {e}")
        print("  可能的原因包括:")
        print("  1. 网络连接问题或无法访问GCS。")
        print("  2. 数据集名称不正确。请确认TFDS中的官方名称。")
        print(f"  3. 磁盘空间不足或 '{data_dir}' 目录没有写入权限。")

def main():
    """
    主函数，用于执行数据集下载流程。
    """
    print("--- Open X-Embodiment 数据集子集下载脚本 ---")
    
    # 确保下载目录存在
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"创建下载目录: {DOWNLOAD_DIR}")
        os.makedirs(DOWNLOAD_DIR)

    # 循环下载所有指定的数据集
    for friendly_name, tfds_name in DATASETS_TO_DOWNLOAD.items():
        download_and_verify_subset(
            friendly_name=friendly_name,
            tfds_name=tfds_name,
            num_samples=NUM_SAMPLES_PER_DATASET,
            data_dir=DOWNLOAD_DIR
        )
    
    print(f"\n{'='*50}")
    print("所有指定的下载任务已执行完毕。")
    print(f"请检查 '{DOWNLOAD_DIR}' 目录查看下载的数据。")
    print("="*50)


if __name__ == '__main__':
    # 禁用 TensorFlow 的一些内存增长行为，这对于使用 GPU 是一个好习惯
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # 必须在程序启动时设置内存增长
            print(e)
            
    main() 