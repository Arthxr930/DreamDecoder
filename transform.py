import json
import re

def parse_dreams_complete(text):
    dreams = []
    
    # 更灵活的正则表达式，处理各种格式
    pattern = r'#(\d+-\d+)\s+(.*?)\s*\((\d+)\s*words?\)\s*(.*?)(?=\s*#\d+-\d+|\s*$)'
    
    # 使用多行模式匹配
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    print(f"正则表达式匹配到 {len(matches)} 个梦境")
    
    for i, match in enumerate(matches):
        dream_id = match[0]
        title = match[1].strip()
        word_count = int(match[2])
        content = match[3].strip()
        
        # 清理内容中的多余空行
        content = re.sub(r'\n\s*\n', '\n', content)
        content = content.replace('\n', ' ').strip()
        
        dream = {
            'id': dream_id,
            'title': title,
            'word_count': word_count,
            'content': content
        }
        dreams.append(dream)
        
        # 打印调试信息（可选）
        if i < 3:  # 只打印前3个的调试信息
            print(f"梦境 {i+1}: ID={dream_id}, 字数={word_count}, 标题长度={len(title)}")
    
    return dreams

def debug_text_parsing(text):
    """调试函数：分析文本结构"""
    print("=== 文本分析 ===")
    print(f"文本总长度: {len(text)} 字符")
    
    # 查找所有梦境ID
    dream_ids = re.findall(r'#\d+-\d+', text)
    print(f"找到的梦境ID数量: {len(dream_ids)}")
    print(f"前5个ID: {dream_ids[:5]}")
    
    # 查找所有字数统计
    word_counts = re.findall(r'\(\d+\s*words?\)', text, re.IGNORECASE)
    print(f"找到的字数统计数量: {len(word_counts)}")
    
    # 分割文本查看结构
    lines = text.split('\n')
    print(f"总行数: {len(lines)}")
    
    # 显示前20行
    print("\n前20行内容:")
    for i, line in enumerate(lines[:20]):
        print(f"{i:2d}: {repr(line)}")

def save_dreams_to_json(dreams, output_file='dreams_complete.json'):
    """保存梦境数据到JSON文件"""
    output = {
        "metadata": {
            "total_dreams": len(dreams),
            "source": "College students 1997-1998",
            "description": "Complete dream collection from psychology study"
        },
        "dreams": dreams
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 成功保存 {len(dreams)} 个梦境到 {output_file}")

def main():
    # 读取文本文件
    try:
        with open('dreams.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        
        print(f"读取文件成功，文本长度: {len(text)} 字符")
        
        # 调试：分析文本结构
        debug_text_parsing(text)
        
        # 解析梦境
        print("\n=== 开始解析梦境 ===")
        dreams = parse_dreams_complete(text)
        
        if dreams:
            save_dreams_to_json(dreams)
            
            # 显示统计信息
            print(f"\n📊 最终统计:")
            print(f"总梦境数: {len(dreams)}")
            print(f"ID范围: {dreams[0]['id']} - {dreams[-1]['id']}")
            print(f"字数范围: {min(d['word_count'] for d in dreams)} - {max(d['word_count'] for d in dreams)}")
            
            # 显示前几个梦境
            print(f"\n前3个梦境详情:")
            for i, dream in enumerate(dreams[:3]):
                print(f"\n{i+1}. ID: {dream['id']}")
                print(f"   标题: {dream['title'][:50]}...")
                print(f"   字数: {dream['word_count']}")
                print(f"   内容: {dream['content'][:80]}...")
        else:
            print("❌ 没有解析到任何梦境")
            
    except FileNotFoundError:
        print("❌ 错误：找不到 dreams.txt 文件")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()