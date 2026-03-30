
# import os
# import requests
# from dotenv import load_dotenv
# load_dotenv()

# api_key = os.getenv('ARK_API_KEY')
# base_url = os.getenv('BASE_URL')
# embedder = os.getenv('EMBEDDER')

# print('API Key:', api_key[:15] + '...')
# print('Base URL:', base_url)
# print('Embedder:', embedder)

# headers = {
#     'Authorization': f'Bearer {api_key}',
#     'Content-Type': 'application/json'
# }
# data = {
#     'model': embedder,
#     'input': ['测试'],
#     'encoding_format': 'float'
# }

# # 先检查模型是否存在
# resp = requests.get(f'{base_url}/models', headers={'Authorization': f'Bearer {api_key}'}, timeout=30)
# models = [m['id'] for m in resp.json()['data']]
# print('bge-m3 in models:', 'BAAI/bge-m3' in models)

# # 尝试调用 embedding
# resp = requests.post(f'{base_url}/embeddings', headers=headers, json=data, timeout=30)
# print('Status:', resp.status_code)
# print('Response:', resp.text[:300])



# 简易退休计算器逻辑
def retirement_calculator(age_now, annual_expense, inflation, current_savings, return_rate):
    years_to_retire = 35 - age_now  # 假设60岁退休
    future_expense = annual_expense * (1 + inflation) ** years_to_retire
    required_capital = future_expense / 0.04  # 4%法则
    gap = required_capital - current_savings * (1 + return_rate) ** years_to_retire
    
    if gap <= 0:
        return "已具备退休条件！"
    else:
        return f"需年储蓄：{gap / years_to_retire:.2f}万元"

# 示例：35岁年花20万，通胀3%，现有200万，收益5%
print(retirement_calculator(30, 3, 0.03, 50, 0.10)) 
# 输出：需年储蓄18.3万元