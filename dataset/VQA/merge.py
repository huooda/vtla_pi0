import json
import ijson
from tqdm import tqdm
import os

# --- 使用说明 ---
# 1. 确保已安装 ijson 和 tqdm: pip install ijson tqdm
# 2. 将此脚本放置在 VQA 数据集所在的目录中 (例如 dataset/VQA/)
# 3. 根据你的文件名修改下面的 QUESTION_FILE, ANNOTATION_FILE 和 OUTPUT_FILE 变量。
#    这里使用了 VQA v2 训练集的标准文件名作为示例。
# 4. 运行脚本: python merge.py
# -----------------

# --- 文件路径配置 ---
# 请根据你的实际文件名修改
QUESTION_FILE = 'v2_OpenEnded_mscoco_train2014_questions.json'
ANNOTATION_FILE = 'v2_mscoco_train2014_annotations.json'
OUTPUT_FILE = 'merged_vqa.json'

def merge_vqa_files():
    """
    合并 VQA 的问题和答案文件。
    该函数以内存高效的方式处理大型JSON文件。
    """
    if not os.path.exists(QUESTION_FILE):
        print(f"错误: 问题文件未找到 -> {QUESTION_FILE}")
        print("请确保 QUESTION_FILE 的路径和文件名正确。")
        return

    if not os.path.exists(ANNOTATION_FILE):
        print(f"错误: 答案文件未找到 -> {ANNOTATION_FILE}")
        print("请确保 ANNOTATION_FILE 的路径和文件名正确。")
        return

    print("第一步: 正在从答案文件加载答案...")
    # 使用流式解析读取答案文件，构建 question_id -> answer 的映射
    # 这比一次性加载整个文件更节省内存
    answers_map = {}
    with open(ANNOTATION_FILE, 'rb') as f:
        annotations_parser = ijson.items(f, 'annotations.item')
        for annotation in tqdm(annotations_parser, desc="解析答案"):
            answers_map[annotation['question_id']] = annotation['multiple_choice_answer']
    
    print(f"加载了 {len(answers_map)} 个答案。")
    print("-" * 20)
    print("第二步: 正在合并问题与答案...")

    # 流式读取问题文件，并与答案合并
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile, \
         open(QUESTION_FILE, 'rb') as f:
        
        questions_parser = ijson.items(f, 'questions.item')
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(questions_parser, desc="合并数据")
        
        for i, question_data in enumerate(progress_bar):
            question_id = question_data['question_id']
            
            # 从字典中查找对应的答案
            answer = answers_map.get(question_id)
            
            if answer:
                # 构建新的数据结构
                merged_item = {
                    'id': i + 1,
                    'image_id': question_data['image_id'],
                    'question': f"[text] {question_data['question']}",
                    'answer': answer
                }
                
                # 将合并后的条目写入输出文件，每行一个JSON对象
                outfile.write(json.dumps(merged_item, ensure_ascii=False) + '\n')
    
    print("-" * 20)
    print(f"合并完成！数据已保存到 {OUTPUT_FILE}")
    print(f"总共处理了 {i + 1} 个问答对。")


if __name__ == '__main__':
    merge_vqa_files() 