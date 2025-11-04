import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

#改动测试

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数显示问题
# 数据读取与预处理
def load_and_preprocess_data():
    # 示例数据，这些数据需要替换成实际数据
    market_size_data = {
        'indicator_names': ['市值(亿美元)', '市场主导份额(%)', '年内增长率(%)', '支持链数量(条)'],
        'USDT': [1680, 60, 32, 15],
        'USDC': [720, 35, 72, 8]
    }

    liquidity_data = {
        'indicator_names': ['月度交易量(亿美元)', 'Volume/MCap比率', '交易对数量(个)', '链上活跃度(万笔)'],
        'USDT': [607.98, 0.65, 10000, 240],
        'USDC': [669.15, 0.28, 8000, 28.4]
    }

    compliance_data = {
        'indicator_names': ['合规交易所占比(%)', '机构投资者占比(%)', '企业合作数量(家)', '储备透明度评分'],
        'USDT': [59.90, 30, 127, 65],
        'USDC': [75, 65, 345, 95]
    }

    defi_data = {
        'indicator_names': ['DeFi交易占比(%)', 'DeFi锁仓量(亿美元)', '年化收益率(%)'],
        'USDT': [32, 75, 2.8],
        'USDC': [48, 382, 4.5]
    }

    risk_data = {
        'indicator_names': ['储备资产波动率(%)', '脱钩率(%)'],
        'USDT': [8, 0.3],
        'USDC': [1.2, 0.05]
    }

    # 合并所有数据
    all_indicator_names = (market_size_data['indicator_names'] +
                           liquidity_data['indicator_names'] +
                           compliance_data['indicator_names'] +
                           defi_data['indicator_names'] +
                           risk_data['indicator_names'])

    all_USDT = np.array(market_size_data['USDT'] + liquidity_data['USDT'] +
                        compliance_data['USDT'] + defi_data['USDT'] + risk_data['USDT'])

    all_USDC = np.array(market_size_data['USDC'] + liquidity_data['USDC'] +
                        compliance_data['USDC'] + defi_data['USDC'] + risk_data['USDC'])

    return all_indicator_names, all_USDT, all_USDC, market_size_data, liquidity_data, compliance_data, defi_data, risk_data

# Min-Max标准化
def normalize_minmax(x):
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0

# 标准化数据
def normalize_group(data_dict, reverse=False):
    USDT_vals = np.array(data_dict['USDT'])
    USDC_vals = np.array(data_dict['USDC'])

    norm_result = np.zeros((2, len(USDT_vals)))

    for i in range(len(USDT_vals)):
        values = np.array([USDT_vals[i], USDC_vals[i]])
        norm_values = normalize_minmax(values)

        if reverse:
            norm_values = 1 - norm_values

        norm_result[0, i] = norm_values[0]  # USDT
        norm_result[1, i] = norm_values[1]  # USDC

    return norm_result

# 综合评分计算
def compute_total_scores():
    # 加载并预处理数据
    all_indicator_names, all_USDT, all_USDC, market_size_data, liquidity_data, compliance_data, defi_data, risk_data = load_and_preprocess_data()

    # 数据标准化
    market_norm = normalize_group(market_size_data)
    liquidity_norm = normalize_group(liquidity_data)
    compliance_norm = normalize_group(compliance_data)
    defi_norm = normalize_group(defi_data)
    risk_norm = normalize_group(risk_data, reverse=True)  # 风险指标反向

    # 权重设定
    weights = {
        'market_size': 0.25,
        'liquidity': 0.20,
        'compliance': 0.20,
        'defi': 0.20,
        'risk': 0.15
    }

    # 计算各维度得分
    USDT_market_score = np.mean(market_norm[0, :]) * 100
    USDC_market_score = np.mean(market_norm[1, :]) * 100

    USDT_liquidity_score = np.mean(liquidity_norm[0, :]) * 100
    USDC_liquidity_score = np.mean(liquidity_norm[1, :]) * 100

    USDT_compliance_score = np.mean(compliance_norm[0, :]) * 100
    USDC_compliance_score = np.mean(compliance_norm[1, :]) * 100

    USDT_defi_score = np.mean(defi_norm[0, :]) * 100
    USDC_defi_score = np.mean(defi_norm[1, :]) * 100

    USDT_risk_score = np.mean(risk_norm[0, :]) * 100
    USDC_risk_score = np.mean(risk_norm[1, :]) * 100

    # 综合得分
    USDT_total_score = (weights['market_size'] * USDT_market_score +
                        weights['liquidity'] * USDT_liquidity_score +
                        weights['compliance'] * USDT_compliance_score +
                        weights['defi'] * USDT_defi_score +
                        weights['risk'] * USDT_risk_score)

    USDC_total_score = (weights['market_size'] * USDC_market_score +
                        weights['liquidity'] * USDC_liquidity_score +
                        weights['compliance'] * USDC_compliance_score +
                        weights['defi'] * USDC_defi_score +
                        weights['risk'] * USDC_risk_score)

    return USDT_total_score, USDC_total_score, USDT_market_score, USDC_market_score, USDT_liquidity_score, USDC_liquidity_score, USDT_compliance_score, USDC_compliance_score, USDT_defi_score, USDC_defi_score, USDT_risk_score, USDC_risk_score

# 可视化
def plot_results(USDT_total_score, USDC_total_score, USDT_market_score, USDC_market_score, USDT_liquidity_score, USDC_liquidity_score, USDT_compliance_score, USDC_compliance_score, USDT_defi_score, USDC_defi_score, USDT_risk_score, USDC_risk_score):
    # 雷达图显示
    labels = ['市场规模', '流动性', '合规性', 'DeFi生态', '风险控制']
    usdt_values = [USDT_market_score, USDT_liquidity_score, USDT_compliance_score, USDT_defi_score, USDT_risk_score]
    usdc_values = [USDC_market_score, USDC_liquidity_score, USDC_compliance_score, USDC_defi_score, USDC_risk_score]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    usdt_values += usdt_values[:1]
    usdc_values += usdc_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, usdt_values, color='blue', linewidth=2, linestyle='solid', label='USDT')
    ax.fill(angles, usdt_values, color='blue', alpha=0.25)
    ax.plot(angles, usdc_values, color='orange', linewidth=2, linestyle='solid', label='USDC')
    ax.fill(angles, usdc_values, color='orange', alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    plt.title("USDT vs USDC 综合评分对比", fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()

    # 各维度柱状图对比
    categories = ['市场规模', '流动性', '合规性', 'DeFi生态', '风险控制']
    USDT_scores = [USDT_market_score, USDT_liquidity_score, USDT_compliance_score, USDT_defi_score, USDT_risk_score]
    USDC_scores = [USDC_market_score, USDC_liquidity_score, USDC_compliance_score, USDC_defi_score, USDC_risk_score]

    x = np.arange(len(categories))  # 横坐标
    width = 0.35  # 每个柱子的宽度

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, USDT_scores, width, label='USDT', color='blue')
    ax.bar(x + width / 2, USDC_scores, width, label='USDC', color='orange')

    ax.set_ylabel('得分')
    ax.set_title('USDT与USDC各维度得分对比')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    plt.tight_layout()
    plt.style.use("ggplot")
    plt.show()

# 主函数
def main():
    USDT_total_score, USDC_total_score, USDT_market_score, USDC_market_score, USDT_liquidity_score, USDC_liquidity_score, USDT_compliance_score, USDC_compliance_score, USDT_defi_score, USDC_defi_score, USDT_risk_score, USDC_risk_score = compute_total_scores()
    print(f"USDT 总评分: {USDT_total_score}")
    print(f"USDC 总评分: {USDC_total_score}")
    plot_results(USDT_total_score, USDC_total_score, USDT_market_score, USDC_market_score, USDT_liquidity_score, USDC_liquidity_score, USDT_compliance_score, USDC_compliance_score, USDT_defi_score, USDC_defi_score, USDT_risk_score, USDC_risk_score)

# 执行主函数
if __name__ == "__main__":
    main()
