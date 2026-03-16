# 在项目根目录下运行这个临时脚本
import sys
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
from oat.env.libero.factory import MT_TASKS

# 打印所有支持的多任务名称
print("支持的 multitask_name 列表：")
for key in MT_TASKS.keys():
    print(f"- {key}")

# 若包含 libero_spatial，进一步查看其对应的子任务
if "libero_spatial" in MT_TASKS:
    print(f"\nlibero_spatial 对应的子任务：{MT_TASKS['libero_spatial']}")
else:
    print("\nMT_TASKS 中无 libero_spatial 键，无法直接使用")