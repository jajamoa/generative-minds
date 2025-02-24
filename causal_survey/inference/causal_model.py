import pyro
import torch
import pyro.distributions as dist
from typing import List, Dict, Any
import networkx as nx

class CausalModel:
    def __init__(self):
        """初始化因果模型"""
        pyro.clear_param_store()
        self.graph = nx.DiGraph()
        self.observed_data = {}
        
    def _create_variable_name(self, text: str) -> str:
        """创建变量名"""
        return text.lower().replace(" ", "_")
        
    def add_causal_relation(self, cause: str, effect: str):
        """添加因果关系到图中"""
        cause_var = self._create_variable_name(cause)
        effect_var = self._create_variable_name(effect)
        
        self.graph.add_edge(cause_var, effect_var)
        
    def model(self):
        """定义概率图模型"""
        # 为图中的每个节点创建变量
        node_samples = {}
        
        # 首先处理没有父节点的节点（根节点）
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                node_samples[node] = pyro.sample(
                    node,
                    dist.Beta(torch.tensor(2.0), torch.tensor(2.0))
                )
                
        # 然后处理有父节点的节点
        for node in self.graph.nodes():
            if self.graph.in_degree(node) > 0:
                parents = list(self.graph.predecessors(node))
                parent_values = torch.stack([node_samples[p] for p in parents])
                
                # 使用父节点的值计算当前节点的概率
                alpha = 0.2 + 0.5 * torch.mean(parent_values)
                node_samples[node] = pyro.sample(
                    node,
                    dist.Beta(
                        alpha * torch.ones(1),
                        (1 - alpha) * torch.ones(1)
                    )
                )
                
        return node_samples
        
    def infer(self, observed_data: Dict[str, Any]):
        """执行贝叶斯推断"""
        self.observed_data = {
            self._create_variable_name(k): torch.tensor(float(v))
            for k, v in observed_data.items()
        }
        
        # 条件化模型
        conditioned_model = pyro.condition(self.model, data=self.observed_data)
        
        # 使用重要性采样进行推断
        importance = pyro.infer.Importance(conditioned_model, num_samples=1000)
        importance.run()
        
        # 获取后验分布
        marginals = {}
        for node in self.graph.nodes():
            if node not in self.observed_data:
                marginal = pyro.infer.EmpiricalMarginal(importance, sites=node)
                marginals[node] = {
                    "mean": float(marginal.mean),
                    "std": float(marginal.variance.sqrt())
                }
                
        return marginals
        
    def update_from_dialogue(self, causal_relations: List[Dict[str, str]]):
        """从对话中更新因果模型"""
        for relation in causal_relations:
            self.add_causal_relation(relation["cause"], relation["effect"])
            
        # 为观察到的变量设置简单的二值观测
        observed_data = {
            relation["effect"]: 1.0
            for relation in causal_relations
        }
        
        return self.infer(observed_data) 