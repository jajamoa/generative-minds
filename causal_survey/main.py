import os
from dotenv import load_dotenv
from dialogue.dialogue_manager import DialogueManager
from inference.causal_model import CausalModel
from visualization.graph_viz import CausalGraphVisualizer

def main():
    # 加载环境变量
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请在.env文件中设置OPENAI_API_KEY")
        
    # 创建输出目录
    os.makedirs("logs", exist_ok=True)
    
    # 初始化对话管理器
    dialogue_manager = DialogueManager(api_key)
    
    print("\n=== 智能问卷对话系统 ===")
    print("这个系统将通过对话收集信息，并分析其中的因果关系。")
    print("您可以描述多个决策场景，每个场景我们都会分析其中的因果关系。")
    print("请尽可能详细地回答问题。\n")
    
    # 开始对话并收集多个场景的因果关系
    scenarios = dialogue_manager.start_conversation()
    
    if not scenarios:
        print("\n未收集到任何决策场景。")
        return
        
    # 为每个场景创建因果模型和可视化
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n=== 场景 {i}: {scenario['description'][:50]}... ===")
        
        if not scenario['causal_relations']:
            print("未能识别出任何因果关系。")
            continue
            
        print("\n识别出的因果关系：")
        for relation in scenario['causal_relations']:
            print(f"- {relation['cause']} -> {relation['effect']}")
            
        # 创建因果模型
        causal_model = CausalModel()
        marginals = causal_model.update_from_dialogue(scenario['causal_relations'])
        
        # 可视化结果
        visualizer = CausalGraphVisualizer(causal_model.graph)
        
        # 为每个场景创建单独的文件名
        scenario_id = f"{dialogue_manager.chat_id}_scenario_{i}"
        
        # 保存静态图像
        image_path = visualizer.save_graph_image(
            output_path=f"logs/{scenario_id}_graph.png",
            marginals=marginals,
            title=f"因果关系图 - 场景 {i}"
        )
        
        # 创建交互式HTML图
        html_path = visualizer.create_html_graph(
            output_path=f"logs/{scenario_id}_graph.html",
            marginals=marginals
        )
        
        print(f"\n场景 {i} 的分析结果已保存：")
        print(f"- 静态图像：{image_path}")
        print(f"- 交互式图表：{html_path}")
        
    print(f"\n完整对话记录已保存至：logs/{dialogue_manager.chat_id}.json")
    
if __name__ == "__main__":
    main() 