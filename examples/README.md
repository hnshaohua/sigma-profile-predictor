# example / 示例文件

This directory contains example input files and corresponding prediction results.

本目录包含示例输入文件及对应的预测结果。

---

## Input Files / 输入文件

| File / 文件 | Description / 说明 |
|---|---|
| `example_input_with_area.csv` | 带分子总表面积的输入示例 / Example input with molecular total surface area |
| `example_input_without_area.csv` | 不带分子总表面积的输入示例 / Example input without molecular total surface area |

### Input Format / 输入格式

`example_input_with_area.csv` 包含两列：

| smiles | area |
|---|---|
| CCO | 123.45 |
| c1ccccc1 | 156.78 |

- 第一列：`smiles`
- 第二列：`area`，分子总表面积

`example_input_without_area.csv` 仅包含一列：

| smiles |
|---|
| CCO |
| c1ccccc1 |

- 第一列：`smiles`
- 分子总表面积将由模型预测

---

## Output Files / 输出文件

| File / 文件 | Input / 输入 | Model / 模型 |
|---|---|---|
| `output_with_area_mse.csv` | `example_input_with_area.csv` | MSE |
| `output_with_area_composite.csv` | `example_input_with_area.csv` | Composite |
| `output_without_area_mse.csv` | `example_input_without_area.csv` | MSE |
| `output_without_area_composite.csv` | `example_input_without_area.csv` | Composite |

### Output Format / 输出格式

每个输出 CSV 文件包含 53 列：

| 列 / Column | 内容 / Content |
|---|---|
| 第 1 列 / Column 1 | `smiles` |
| 第 2 列 / Column 2 | `area`，分子总表面积 |
| 第 3–53 列 / Column 3–53 | 预测的 Sigma-Profile 数值 / Predicted sigma-profile values |

`area` 列的取值规则：

- 若输入文件包含 `area` 列，输出中直接保留输入值
- 若输入文件不包含 `area` 列，输出中为模型预测的分子总表面积

Sigma-Profile 数值说明：

- 共 51 个数值
- X 轴范围：`-0.25` 到 `0.25`
- X 轴间隔：`0.01`
- X 轴物理意义：单位表面积的电荷密度
- Y 轴单位：平方埃（Å²）

---

## Usage / 使用方法

带分子总表面积的输入：

```bash
python predict.py --input example/example_input_with_area.csv --output result.csv --model mse
python predict.py --input example/example_input_with_area.csv --output result.csv --model composite


不带分子总表面积的输入：

python predict.py --input example/example_input_without_area.csv --output result.csv --model mse
python predict.py --input example/example_input_without_area.csv --output result.csv --model composite


详细参数说明请参见项目根目录下的 USAGE.md。
For detailed parameter descriptions, see USAGE.md in the project root directory.