# -*- coding: utf-8 -*-

import dgl
import dgl.nn as dglnn
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

# n_max_clicks = 10 #
student_data = np.genfromtxt("{}.csv".format("student_24_data"), delimiter=',', dtype=np.dtype(str))
course_data = np.genfromtxt("{}.csv".format("course_data"), delimiter=',', dtype=np.dtype(str))
scholarship_data = np.genfromtxt("{}.csv".format("scholarship_data"), delimiter=',', dtype=np.dtype(str),
                                 encoding='utf-8')
competition_data = np.genfromtxt("{}.csv".format("competition_data"), delimiter=',', dtype=np.dtype(str),
                                 encoding='utf-8')

rel_student_competition = np.genfromtxt("{}.csv".format("rel_student_24_competition"), delimiter=',',
                                        dtype=np.dtype(str), encoding='utf-8')
rel_student_scholarship = np.genfromtxt("{}.csv".format("rel_student_24_scholarship"), delimiter=',',
                                        dtype=np.dtype(str), encoding='utf-8')
rel_student_course = np.genfromtxt("{}.csv".format("rel_student_24_course"), delimiter=',', dtype=np.dtype(str))

# 学生id映射
student_idx = np.array(student_data[:, 0], dtype=np.dtype(str))
student_idx_map = {j: i for i, j in enumerate(student_idx)}
# 课程id映射
course_idx = np.array(course_data[:, 0], dtype=np.dtype(str))
course_idx_map = {j: i for i, j in enumerate(course_idx)}
# 奖学金名称映射
scholarship_name = np.array(scholarship_data[:, 0], dtype=np.dtype(str))
scholarship_name_map = {j: i for i, j in enumerate(scholarship_name)}
# 竞赛名称映射
competition_name = np.array(competition_data[:, 0], dtype=np.dtype(str))
competition_name_map = {j: i for i, j in enumerate(competition_name)}

# stu_rel_des = np.array(list(follow_dst))


edges_stu_comp = np.array(
    [(student_idx_map.get(edge[0]), competition_name_map.get(edge[1]), int(float(edge[2]))) for edge in
     rel_student_competition], dtype=np.int32)
edges_stu_scholarship = np.array(
    [(student_idx_map.get(edge[0]), scholarship_name_map.get(edge[1]), int(float(edge[2]))) for edge in
     rel_student_scholarship], dtype=np.int32)
edges_stu_course = np.array(
    [(student_idx_map.get(edge[0]), course_idx_map.get(edge[1]), edge[2]) for edge in rel_student_course if
     edge[0] in student_idx and edge[1] in course_idx], dtype=np.int32)

hetero_graph = dgl.heterograph({  # 正反两个方向构边
    ('学生', '获取', '奖学金'): (edges_stu_scholarship[:, 0], edges_stu_scholarship[:, 1]),
    ('奖学金', '被获取', '学生'): (edges_stu_scholarship[:, 1], edges_stu_scholarship[:, 0]),

    ('学生', '选则', '课程'): (edges_stu_course[:, 0], edges_stu_course[:, 1]),
    ('课程', '被选则', '学生'): (edges_stu_course[:, 1], edges_stu_course[:, 0]),

    ('学生', '参加', '竞赛'): (edges_stu_comp[:, 0], edges_stu_comp[:, 1]),
    ('竞赛', '被参加', '学生'): (edges_stu_comp[:, 1], edges_stu_comp[:, 0])
})

print(hetero_graph)

# 特征构造

# 学生特征，5644个学生特征18维
hetero_graph.nodes['学生'].data['feature'] = torch.from_numpy(np.array(student_data[:, 1:-1], dtype=np.float32))
# 奖学金特征，9个奖学金特征2维
hetero_graph.nodes['奖学金'].data['feature'] = torch.from_numpy(np.array(scholarship_data[:, 1:], dtype=np.float32))
# 竞赛特征，30个竞赛特征1维
hetero_graph.nodes['竞赛'].data['feature'] = torch.from_numpy(np.array(competition_data[:, 1:], dtype=np.float32))
# 课程特征，4091个课程特征18维
hetero_graph.nodes['课程'].data['feature'] = torch.from_numpy(np.array(course_data[:, 1:], dtype=np.float32))
# 学生类型标签,5644维向量
hetero_graph.nodes['学生'].data['label'] = torch.from_numpy(np.array(student_data[:, -1], dtype=np.int64))

# 边标签
# hetero_graph.edges['获取'].data['label'] = torch.from_numpy(np.array(edges[:,-1]))
# hetero_graph.edges['获取'].data['label'] = torch.from_numpy(np.array(edges[:,-1]))
hetero_graph.nodes['学生'].data['train_mask'] = torch.zeros(5644, dtype=torch.bool).bernoulli(0.6)
hetero_graph.nodes['学生'].data['test_mask'] = ~hetero_graph.nodes['学生'].data['train_mask']


# hetero_graph.edges['点击'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)


# 定义特征聚合模块
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats, hid_feats)
                                            for rel in rel_names}, aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_feats, out_feats)
                                            for rel in rel_names}, aggregate='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


# 构造模型
model = RGCN(in_feats=18, hid_feats=36,
             out_feats=9, rel_names=hetero_graph.etypes)

linear_layer_scho = nn.Linear(2, 18)
linear_layer_comp = nn.Linear(1, 18)
student_feats = hetero_graph.nodes['学生'].data['feature']  # 用户特征
course_feats = hetero_graph.nodes['课程'].data['feature']  # 课程特征
scholarship_feats = hetero_graph.nodes['奖学金'].data['feature']  # 项目特征
scholarship_feats = linear_layer_scho(scholarship_feats)
competition_feats = hetero_graph.nodes['竞赛'].data['feature']  # 项目特征
competition_feats = linear_layer_comp(competition_feats)
labels = hetero_graph.nodes['学生'].data['label']  # 用户类型标签
train_mask = hetero_graph.nodes['学生'].data['train_mask']
test_mask = hetero_graph.nodes['学生'].data['test_mask']
# 特征归一化
scholarship_feats = F.normalize(scholarship_feats, dim=0)
student_feats = F.normalize(student_feats, dim=0)
competition_feats = F.normalize(competition_feats, dim=0)
course_feats = F.normalize(course_feats, dim=0)

node_features = {'奖学金': scholarship_feats, '学生': student_feats, '竞赛': competition_feats, '课程': course_feats}
# 特征字典
h_dict = model(hetero_graph, node_features)

# 模型优化器
opt = torch.optim.Adam(model.parameters())

best_train_acc = 0
loss_list = []
train_score_list = []

# 迭代训练
for epoch in range(5000):
    model.train()
    # 输入图和节点特征，提取出user的特征
    logits = model(hetero_graph, node_features)['学生']
    # 计算损失
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    # 预测user
    pred = logits.argmax(1)
    # 计算准确率
    train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    if best_train_acc < train_acc:
        best_train_acc = train_acc
    train_score_list.append(train_acc)

    # 反向优化
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_list.append(loss.item())
    # 输出训练结果
    print('Loss %.4f, Train Acc %.4f  Test Acc %.4f (Best %.4f)' % (
        loss.item(),
        train_acc.item(),
        test_acc.item(),
        best_train_acc.item(),))
