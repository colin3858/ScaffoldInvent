import pandas as pd

# 读取包含SMILES数据的CSV文件
input_csv_path = 'DOWNLOAD-Z1.csv'  # 替换为你的CSV文件路径
output_smi_path = 'smiles_output.smi'  # 指定输出的.smi文件路径

try:
    # 尝试读取CSV文件，忽略错误行
    df = pd.read_csv(input_csv_path, encoding='gbk', sep=';', on_bad_lines='skip')
except UnicodeDecodeError:
    print("Error: 文件不是gbk编码。请确认文件编码格式。")
    exit()
except pd.errors.ParserError as e:
    print(f"Error: 解析CSV文件时出错: {e}")
    exit()

# 检查CSV文件中是否有数据
if not df.empty:
    # 提取'Smiles'列，假设该列名为'Smiles'
    if 'Smiles' in df.columns:
        smiles_column = df['Smiles']
        
        # 去除空白数据
        smiles_column = smiles_column.dropna()
        
        # 将SMILES数据写入.smi文件
        with open(output_smi_path, 'w', encoding='utf-8') as smi_file:
            for smile in smiles_column:
                smi_file.write(str(smile) + '\n')
        
        print(f"SMILES数据已经成功保存到{output_smi_path}")
    else:
        print("CSV文件中没有找到'Smiles'列")
    # print(df.head())
else:
    print("CSV文件中没有数据")
